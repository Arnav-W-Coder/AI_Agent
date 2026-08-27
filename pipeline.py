"""
pipeline.py — Wires every component into one production-grade system.

Query flow (deterministic — not agent-based, more reliable for production):
  1.  Embed query           → for semantic cache lookup
  2.  Answer cache check    → return immediately on semantic hit
  3.  Rewrite query         → QueryRewriter (trainable few-shot)
  4.  Retrieval cache check → skip steps 5-6 on exact-hash hit
  5.  Hybrid retrieval      → BM25 + dense ANN → RRF → cross-encoder rerank
  6.  Web fallback          → DDG + scrape if retrieval score below threshold
  7.  LLM generation        → direct RAG prompt (not agent — deterministic)
  8.  Critic check          → CriticAndRepair.check()
  9.  Repair if needed      → CriticAndRepair.repair() — no tool re-runs
  10. Final critic check    → after repair
  11. Polish                → strip hedging language
  12. Store in caches       → answer cache + retrieval cache
  13. Record metrics        → QueryTrace → MetricsRecorder
  14. Drift check           → every drift_window queries
"""
import json
import logging
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import RAGConfig
from db import Database
from cache import CacheLayer
from metrics import MetricsRecorder, QueryTrace
from ingestion import AsyncIngestionPipeline
from retrieval import BM25Index, CrossEncoderReranker, HybridRetriever
from rewriter import QueryRewriter
from critic import CriticAndRepair

log = logging.getLogger(__name__)

# ── RAG generation prompt ─────────────────────────────────────────────────────

_RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a precise research assistant.
Answer the question using ONLY the context below.
Each context chunk is labeled with its source file and page.

If the answer is not present in the context, respond with exactly:
"I don't have enough information to answer this confidently."

Never guess. Never add facts not present in the context.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:""")

# ── Domain reliability scoring (for web fallback) ─────────────────────────────

_DOMAIN_SCORES: dict[str, int] = {
    "wikipedia.org": 95,   "britannica.com": 90,
    "*.gov": 95,           "*.edu": 90,
    "arxiv.org": 92,       "pubmed.ncbi.nlm.nih.gov": 95,
    "docs.python.org": 95, "developer.mozilla.org": 92,
    "aws.amazon.com": 85,  "cloud.google.com": 85,
    "learn.microsoft.com": 85, "research.ibm.com": 88,
    "openai.com": 82,      "anthropic.com": 85,
    "langchain.com": 80,   "huggingface.co": 80,
    "stackoverflow.com": 75, "github.com": 78,
    "towardsdatascience.com": 72, "medium.com": 65,
    "pinterest.com": 0,    "facebook.com": 0,
    "twitter.com": 0,      "x.com": 0,
    "tiktok.com": 0,       "grokipedia.com": 15,
}


def _score_domain(url: str) -> int:
    try:
        host = urlparse(url).netloc.lower().lstrip("www.")
    except Exception:
        return 0
    if host in _DOMAIN_SCORES:
        return _DOMAIN_SCORES[host]
    parts = host.split(".")
    for i in range(len(parts) - 1):
        if ".".join(parts[i:]) in _DOMAIN_SCORES:
            return _DOMAIN_SCORES[".".join(parts[i:])]
    tld_key = f"*.{parts[-1]}"
    if tld_key in _DOMAIN_SCORES:
        return _DOMAIN_SCORES[tld_key]
    return 60   # unknown domain — allow but score neutrally


# ── Pipeline ──────────────────────────────────────────────────────────────────

class ProductionRAGPipeline:
    """
    Production RAG system. Call await setup() once, then query() for every request.

    Example:
        cfg      = RAGConfig()
        pipeline = ProductionRAGPipeline(cfg)
        await pipeline.setup()

        result = pipeline.query("What caused westernization in Japan?")
        print(result["answer"])
        print(result["sources"])

        pipeline.rate_answer(result["query_id"], rating=5,
                             rewrite_id=result["rewrite_id"])
    """

    def __init__(self, cfg: RAGConfig) -> None:
        self.cfg = cfg

        # ── Core infrastructure ───────────────────────────────────────────────
        self.db = Database(cfg.db_path)

        # ── Models ────────────────────────────────────────────────────────────
        self.embeddings = OllamaEmbeddings(model=cfg.embed_model)
        self.llm        = ChatOllama(model=cfg.llm_model, num_ctx=cfg.ctx_window)

        # ── Vector store ──────────────────────────────────────────────────────
        self.vectorstore = Chroma(
            persist_directory=str(cfg.chroma_dir),
            embedding_function=self.embeddings,
        )

        # ── Component stack ───────────────────────────────────────────────────
        self.cache    = CacheLayer(self.db, cfg)
        self.metrics  = MetricsRecorder(self.db, cfg)
        self.rewriter = QueryRewriter(self.db, cfg, self.llm)
        self.critic   = CriticAndRepair(self.llm, cfg=cfg)

        # Retrieval components — wired in setup()
        self.bm25:      Optional[BM25Index]            = None
        self.reranker:  Optional[CrossEncoderReranker] = None
        self.retriever: Optional[HybridRetriever]      = None

        # Generation chain
        self._rag_chain = _RAG_PROMPT | self.llm | StrOutputParser()

        # Query counter — drives drift-check cadence
        self._query_count = 0

    # ── Setup ─────────────────────────────────────────────────────────────────

    async def setup(self) -> dict:
        """
        1. Run async PDF ingestion (skips unchanged files).
        2. Rebuild BM25 index from SQLite chunks.
        3. Load cross-encoder reranker (downloads ~80 MB model on first run).
        Must be awaited before any calls to query().
        """
        log.info("=" * 60)
        log.info("[Pipeline] Starting setup...")

        # ── Ingestion ─────────────────────────────────────────────────────────
        ingestion = AsyncIngestionPipeline(
            self.db, self.cfg, self.vectorstore, self.embeddings
        )
        summaries = await ingestion.run()
        self.cache.invalidate_index_cache()

        # ── BM25 ──────────────────────────────────────────────────────────────
        log.info("[Pipeline] Building BM25 index...")
        self.bm25 = BM25Index(self.db)

        # ── Cross-encoder ─────────────────────────────────────────────────────
        log.info("[Pipeline] Loading cross-encoder reranker...")
        self.reranker = CrossEncoderReranker(self.cfg.rerank_model)

        # ── Hybrid retriever ──────────────────────────────────────────────────
        self.retriever = HybridRetriever(
            self.vectorstore, self.bm25, self.reranker, self.cfg
        )

        log.info("[Pipeline] Setup complete — ready to query.")
        log.info("=" * 60)
        return {"ingested_files": summaries}

    # ── Main query entrypoint ─────────────────────────────────────────────────

    def query(
        self,
        question:         str,
        metadata_filter:  Optional[dict] = None,
        use_web_fallback: bool = True,
    ) -> dict:
        """
        Run the full production RAG pipeline for one question.

        Args:
            question:        Raw user question.
            metadata_filter: Optional ChromaDB `where` clause, e.g.
                             {"filename": {"$eq": "report.pdf"}}
            use_web_fallback: If True, scrape the web when retrieval score
                              is below cfg.min_retrieval_score.

        Returns a dict with keys:
            answer, sources, query_id, rewrite_id, rewritten_query,
            from_cache, drift_alert, metrics
        """
        assert self.retriever is not None, "Call await setup() before query()."

        trace = QueryTrace(query_text=question)
        log.info(f"\n[Query] '{question[:80]}'")

        # ── 1. Embed query for semantic cache lookup ───────────────────────────
        query_emb = self._embed_query(question)

        # ── 2. Answer cache check ─────────────────────────────────────────────
        cached = self.cache.get_answer(question, query_emb)
        if cached:
            answer, sources = cached
            trace.answer_cache_hit = True
            trace.t_end = time.time()
            self.metrics.record(trace)
            log.info("[Query] Returning from answer cache.")
            return {
                "answer":          answer,
                "sources":         sources,
                "query_id":        trace.query_id,
                "rewrite_id":      None,
                "rewritten_query": question,
                "from_cache":      True,
                "drift_alert":     None,
                "metrics":         {"answer_cache_hit": True,
                                    "total_ms": trace.total_ms()},
            }

        # ── 3. Rewrite query ──────────────────────────────────────────────────
        rewrite_id, rewritten = self.rewriter.rewrite(question)
        trace.t_rewrite       = time.time()
        trace.rewritten_query = rewritten

        # ── 4. Retrieval cache check ──────────────────────────────────────────
        cached_chunks = self.cache.get_retrieval(rewritten)
        bm25_ids: set  = set()
        dense_ids: set = set()

        if cached_chunks:
            chunks = cached_chunks
            trace.retrieval_cache_hit = True
            trace.t_retrieval = time.time()
            trace.t_rerank    = time.time()
            log.info(f"[Query] Retrieval cache hit — {len(chunks)} chunks")
        else:
            # ── 5 + 6. Parallel PDF candidates + web scraping ─────────────────
            # Both sources run concurrently in a thread pool.
            # If always_scrape_web=False, web runs only when PDF score is low.
            log.info("[Query] Starting parallel PDF + web retrieval...")
            with ThreadPoolExecutor(max_workers=2) as pool:
                pdf_fut = pool.submit(
                    self.retriever.retrieve_candidates, rewritten, metadata_filter
                )
                web_fut = pool.submit(
                    self._web_scrape_chunks, rewritten
                ) if (use_web_fallback and self.cfg.always_scrape_web) else None

                pdf_candidates, bm25_ids, dense_ids = pdf_fut.result()
                web_chunks = web_fut.result() if web_fut else []

            trace.t_retrieval = time.time()

            # If not always-on, check PDF quality and run web only if needed
            if (
                use_web_fallback
                and not self.cfg.always_scrape_web
                and not web_chunks
            ):
                top_pdf_score = (
                    pdf_candidates[0].get("rrf_score", 0.0)
                    if pdf_candidates else 0.0
                )
                if top_pdf_score < self.cfg.min_retrieval_score:
                    log.info("[Query] PDF score low — triggering web fallback")
                    web_chunks = self._web_scrape_chunks(rewritten)

            # Merge all candidates (PDF + web), deduplicate by chroma_id / text
            seen_ids: set = set()
            all_candidates: list[dict] = []
            for c in pdf_candidates:
                cid = c.get("chroma_id", "")
                if cid not in seen_ids:
                    seen_ids.add(cid); all_candidates.append(c)

            for wc in web_chunks[:self.cfg.web_top_k]:
                # Web chunks have no chroma_id — deduplicate on text prefix
                key = wc.get("text", "")[:80]
                if key not in seen_ids:
                    seen_ids.add(key); all_candidates.append(wc)

            log.info(
                f"[Query] Merged candidates: {len(pdf_candidates)} PDF + "
                f"{len(web_chunks)} web = {len(all_candidates)} total"
            )

            # Single cross-encoder rerank pass over the merged set
            chunks = self.retriever.reranker.rerank(
                rewritten, all_candidates,
                self.cfg.top_k_rerank, self.cfg.min_rerank_score
            )
            trace.t_rerank = time.time()
            log.info(
                f"[Query] After rerank: {len(chunks)} chunks | "
                f"top score: {chunks[0]['rerank_score'] if chunks else 'n/a'}"
            )

            # Serialise and store in retrieval cache
            safe_chunks = [
                {k: v for k, v in c.items()
                 if isinstance(v, (str, int, float, bool, type(None)))}
                for c in chunks
            ]
            self.cache.set_retrieval(rewritten, safe_chunks)

        # ── Retrieval quality trace ────────────────────────────────────────────
        rerank_scores = [
            c.get("rerank_score", 0.0) for c in chunks if "rerank_score" in c
        ]
        trace.num_chunks_retrieved = len(chunks)
        trace.mean_rerank_score = (
            round(sum(rerank_scores) / len(rerank_scores), 4)
            if rerank_scores else 0.0
        )
        trace.top_rerank_score = max(rerank_scores) if rerank_scores else 0.0
        trace.bm25_overlap     = len(bm25_ids & dense_ids)

        # ── 7. Format context and generate ────────────────────────────────────
        context = self._format_context(chunks)
        log.info(f"[Query] Generating answer over {len(chunks)} chunks...")
        answer  = self._rag_chain.invoke(
            {"context": context, "question": rewritten}
        ).strip()
        trace.t_generation = time.time()

        # ── 8. Critic check ───────────────────────────────────────────────────
        verdict, claims, faith_score = self.critic.check(context, answer)
        log.info(
            f"[Query] Critic: {verdict} | faithfulness={faith_score:.2f}"
        )

        # ── 9. Repair if hallucinated ─────────────────────────────────────────
        if verdict == "HALLUCINATED":
            answer = self.critic.repair(context, answer, claims)
            verdict2, claims2, faith_score = self.critic.check(context, answer)
            log.info(f"[Query] Post-repair critic: {verdict2} | score={faith_score:.2f}")

        # ── 10. Polish ────────────────────────────────────────────────────────
        answer = self.critic.polish(answer)
        trace.answer_faithfulness = faith_score

        # ── 11. Store in answer cache ─────────────────────────────────────────
        sources = [
            {
                "filename":     Path(c.get("filename", "web")).name,
                "page":         c.get("page_number", 0),
                "rerank_score": round(c.get("rerank_score", 0.0), 3),
            }
            for c in chunks
        ]
        self.cache.set_answer(question, query_emb, answer, sources)

        # ── 12. Record metrics ────────────────────────────────────────────────
        trace.t_end = time.time()
        self.metrics.record(trace)

        # ── 13. Auto-label rewrite from faithfulness score ────────────────────
        self.rewriter.record_answer_score(rewrite_id, faith_score)
        if faith_score >= self.cfg.rewriter_helpful_min_score:
            self.rewriter.record_feedback(rewrite_id, helpful=True)
        elif faith_score < self.cfg.rewriter_unhelpful_max_score:
            self.rewriter.record_feedback(rewrite_id, helpful=False)
        # Scores between the two thresholds get no auto-label — awaits user rating

        # ── 14. Drift check every drift_window queries ────────────────────────
        self._query_count += 1
        drift_alert = None
        if self._query_count % self.cfg.drift_window == 0:
            drift_alert = self.metrics.check_drift()

        log.info(f"[Query] Done in {trace.total_ms():.0f}ms")
        return {
            "answer":          answer,
            "sources":         sources,
            "query_id":        trace.query_id,
            "rewrite_id":      rewrite_id,
            "rewritten_query": rewritten,
            "from_cache":      False,
            "drift_alert":     drift_alert,
            "metrics": {
                "total_ms":         trace.total_ms(),
                "rewrite_ms":       trace.latency_ms(trace.t_start,     trace.t_rewrite),
                "retrieval_ms":     trace.latency_ms(trace.t_rewrite,   trace.t_retrieval),
                "generation_ms":    trace.latency_ms(trace.t_rerank,    trace.t_generation),
                "top_rerank_score": trace.top_rerank_score,
                "mean_rerank_score":trace.mean_rerank_score,
                "faithfulness":     faith_score,
                "chunks_used":      len(chunks),
                "bm25_overlap":     trace.bm25_overlap,
                "retrieval_cached": trace.retrieval_cache_hit,
            },
        }

    # ── User feedback ─────────────────────────────────────────────────────────

    def rate_answer(
        self,
        query_id:   str,
        rating:     int,
        rewrite_id: Optional[int] = None,
    ) -> None:
        """
        Record a 1–5 user rating.
        Also marks the query rewrite as helpful (rating >= 3) or not.
        """
        assert 1 <= rating <= 5, "Rating must be 1–5"
        self.metrics.record_user_rating(query_id, rating)
        if rewrite_id is not None:
            self.rewriter.record_feedback(rewrite_id, helpful=(rating >= 3))
        log.info(f"[Pipeline] Rating: {rating}/5 for query_id={query_id}")

    # ── Monitoring ────────────────────────────────────────────────────────────

    def monitoring_report(self, last_n: int = 50) -> dict:
        """
        Combined report across all subsystems.
        Returns metrics, cache stats, and rewriter pool health.
        """
        return {
            "performance": self.metrics.report(last_n),
            "cache":       self.cache.stats(),
            "rewriter":    self.rewriter.rewrite_stats(),
        }

    def trigger_drift_check(self) -> Optional[str]:
        """Manually run the drift detector. Returns alert string or None."""
        return self.metrics.check_drift()

    def rebuild_bm25(self) -> None:
        """
        Rebuild BM25 index from SQLite — call after manual document additions
        or after a re-embedding job completes.
        """
        log.info("[Pipeline] Rebuilding BM25 index...")
        if self.bm25:
            self.bm25.rebuild()
        log.info("[Pipeline] BM25 rebuild complete.")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _embed_query(self, text: str) -> np.ndarray:
        return np.array(self.embeddings.embed_query(text), dtype=np.float32)

    def _format_context(self, chunks: list[dict]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = Path(chunk.get("filename", "unknown")).name
            page   = chunk.get("page_number", "?")
            score  = chunk.get("rerank_score", 0.0)
            text   = chunk.get("text", chunk.get("text_preview", ""))
            parts.append(
                f"[Source {i} | {source} | page {page} | score {score:.2f}]\n{text}"
            )
        return "\n\n---\n\n".join(parts)

    def _web_scrape_chunks(self, query: str) -> list[dict]:
        """
        DDG search → domain-score → scrape approved URLs →
        embed into temp Chroma → similarity-search for best chunks.
        Returns a list of chunk dicts compatible with the main pipeline.
        """
        log.info(f"[WebFallback] Searching DDG for: '{query}'")
        raw = []
        for attempt in range(self.cfg.ddg_retries):
            try:
                with DDGS() as ddgs:
                    raw = list(ddgs.text(query, max_results=self.cfg.max_scrape_urls))
                if raw:
                    break
                time.sleep(2 ** attempt)
            except Exception as e:
                log.warning(f"[WebFallback] DDG attempt {attempt+1} failed: {e}")
                time.sleep(2 ** attempt)

        if not raw:
            log.warning("[WebFallback] No DDG results.")
            return []

        # Filter by domain score
        approved = sorted(
            [r for r in raw if _score_domain(r["href"]) >= self.cfg.min_domain_score],
            key=lambda r: _score_domain(r["href"]),
            reverse=True,
        )
        if not approved:
            log.warning("[WebFallback] All URLs below domain score threshold.")
            return []

        log.info(f"[WebFallback] Approved {len(approved)}/{len(raw)} URLs")

        # Scrape each URL
        scraped_docs: list[Document] = []
        for r in approved:
            url, title = r["href"], r.get("title", "Web")
            text = self._scrape_url(url)
            if text and not text.startswith("Error:"):
                scraped_docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": url, "title": title},
                    )
                )
                log.info(f"[WebFallback] Scraped: {url[:60]}")

        if not scraped_docs:
            return []

        # Chunk scraped content
        splitter    = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
        web_chunks  = splitter.split_documents(scraped_docs)

        # Embed into a temporary in-memory Chroma collection
        try:
            temp_store = Chroma.from_documents(
                documents=web_chunks,
                embedding=self.embeddings,
            )
            results = temp_store.similarity_search_with_score(query, k=4)
        except Exception as e:
            log.error(f"[WebFallback] Temp store error: {e}")
            return []

        # Convert to pipeline chunk format
        return [
            {
                "text":         doc.page_content,
                "filename":     doc.metadata.get("source", "web"),
                "page_number":  0,
                "rerank_score": round(float(1 - score), 3),
                "source_type":  "web",
            }
            for doc, score in results
        ]

    def _scrape_url(self, url: str, char_limit: int = 2500) -> str:
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"},
                timeout=10,
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script","style","nav","footer","header","aside"]):
                tag.decompose()
            main = soup.find("main") or soup.find("article") or soup.body
            text = main.get_text(separator="\n", strip=True) if main else ""
            return text[:char_limit] + ("\n[truncated]" if len(text) > char_limit else "")
        except requests.exceptions.Timeout:
            return "Error: timeout"
        except requests.exceptions.HTTPError as e:
            return f"Error: HTTP {e.response.status_code}"
        except Exception as e:
            return f"Error: {e}"