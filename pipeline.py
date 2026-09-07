"""
pipeline.py — Wires every component into one production-grade system.

Query flow:
  1. Embed query → answer-cache lookup
  2. Rewrite query
  3. Retrieval-cache lookup
  4. Hybrid PDF retrieval → BM25 + dense ANN → RRF
  5. Web retrieval runs concurrently when configured (always-on remains supported)
  6. Merge PDF + web candidates → ONE cross-encoder rerank
  7. Expand winning PDF children to parent/neighbor context
  8. Grounded LLM generation
  9. Claim-level critic → optional surgical repair
  10. Deterministic cleanup and cache/metrics storage
"""
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings

from config import RAGConfig
from db import Database
from cache import CacheLayer
from metrics import MetricsRecorder, QueryTrace
from ingestion import AsyncIngestionPipeline
from retrieval import BM25Index, CrossEncoderReranker, HybridRetriever
from rewriter import QueryRewriter
from critic import CriticAndRepair
from web_store import WebChunkStore

log = logging.getLogger(__name__)

_RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a precise research assistant. Answer the QUESTION using only the
retrieved CONTEXT. The context may contain several sources, so synthesize the
relevant evidence instead of copying a single sentence.

Answer requirements:
- Directly answer the question first.
- For questions asking for components, causes, steps, comparisons, or features,
  cover each major item supported by the context and explain its role briefly.
- Use short headings or numbered/bulleted lists when they improve clarity.
- Explain relationships between components when the context supports them.
- Prefer a complete, useful answer over a one-sentence summary.
- Every factual claim must be supported by the context.
- You may paraphrase and synthesize information that is explicitly supported by
  multiple context chunks.
- Do not invent facts, examples, citations, numbers, or terminology.
- If the context genuinely does not contain enough information, say exactly:
  I don't have enough information to answer this confidently.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:
""")

_DOMAIN_SCORES: dict[str, int] = {
    "wikipedia.org": 95, "britannica.com": 90,
    "*.gov": 95, "*.edu": 90,
    "arxiv.org": 92, "pubmed.ncbi.nlm.nih.gov": 95,
    "docs.python.org": 95, "developer.mozilla.org": 92,
    "aws.amazon.com": 85, "cloud.google.com": 85,
    "learn.microsoft.com": 85, "research.ibm.com": 88,
    "openai.com": 82, "anthropic.com": 85,
    "langchain.com": 80, "huggingface.co": 80,
    "stackoverflow.com": 75, "github.com": 78,
    "towardsdatascience.com": 72, "medium.com": 65,
    "pinterest.com": 0, "facebook.com": 0,
    "twitter.com": 0, "x.com": 0, "tiktok.com": 0,
    "grokipedia.com": 15,
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
        suffix = ".".join(parts[i:])
        if suffix in _DOMAIN_SCORES:
            return _DOMAIN_SCORES[suffix]
    tld_key = f"*.{parts[-1]}" if parts else ""
    return _DOMAIN_SCORES.get(tld_key, 60)


class ProductionRAGPipeline:
    """Deterministic production RAG pipeline."""

    def __init__(self, cfg: RAGConfig) -> None:
        self.cfg = cfg
        self.db = Database(cfg.db_path)
        self.embeddings = OllamaEmbeddings(model=cfg.embed_model)
        self.llm = ChatOllama(model=cfg.llm_model, num_ctx=cfg.ctx_window)
        self.vectorstore = Chroma(
            persist_directory=str(cfg.chroma_dir),
            embedding_function=self.embeddings,
        )

        self.cache = CacheLayer(self.db, cfg)
        self.metrics = MetricsRecorder(self.db, cfg)
        self.rewriter = QueryRewriter(self.db, cfg, self.llm)
        self.critic = CriticAndRepair(self.llm, cfg=cfg)

        self.bm25: Optional[BM25Index] = None
        self.reranker: Optional[CrossEncoderReranker] = None
        self.retriever: Optional[HybridRetriever] = None
        self.web_store: Optional[WebChunkStore] = None
        self._rag_chain = _RAG_PROMPT | self.llm | StrOutputParser()
        self._query_count = 0

    async def setup(self) -> dict:
        log.info("=" * 60)
        log.info("[Pipeline] Starting setup...")

        ingestion = AsyncIngestionPipeline(
            self.db, self.cfg, self.vectorstore, self.embeddings
        )
        summaries = await ingestion.run()
        self.cache.invalidate_index_cache()

        self.bm25 = BM25Index(self.db)
        self.reranker = CrossEncoderReranker(self.cfg.rerank_model)
        self.retriever = HybridRetriever(
            self.vectorstore, self.bm25, self.reranker, self.cfg
        )
        self.web_store = WebChunkStore(self.db, self.cfg, self.embeddings)

        log.info("[Pipeline] Setup complete — ready to query.")
        log.info("=" * 60)
        return {"ingested_files": summaries}

    def query(
        self,
        question: str,
        metadata_filter: Optional[dict] = None,
        use_web_fallback: bool = True,
    ) -> dict:
        assert self.retriever is not None, "Call await setup() before query()."
        trace = QueryTrace(query_text=question)
        log.info("\n[Query] '%s'", question[:80])

        query_emb = self._embed_query(question)
        cached = self.cache.get_answer(question, query_emb)
        if cached:
            answer, sources = cached
            trace.answer_cache_hit = True
            trace.t_end = time.time()
            self.metrics.record(trace)
            return {
                "answer": answer,
                "sources": sources,
                "query_id": trace.query_id,
                "rewrite_id": None,
                "rewritten_query": question,
                "from_cache": True,
                "drift_alert": None,
                "metrics": {"answer_cache_hit": True, "total_ms": trace.total_ms()},
            }

        rewrite_id, rewritten = self.rewriter.rewrite(question)
        trace.t_rewrite = time.time()
        trace.rewritten_query = rewritten

        cached_chunks = self.cache.get_retrieval(rewritten)
        bm25_ids: set = set()
        dense_ids: set = set()

        if cached_chunks:
            chunks = cached_chunks
            trace.retrieval_cache_hit = True
            trace.t_retrieval = time.time()
            trace.t_rerank = time.time()
            log.info("[Query] Retrieval cache hit — %d chunks", len(chunks))
        else:
            log.info("[Query] Starting PDF + web retrieval...")
            with ThreadPoolExecutor(max_workers=2) as pool:
                pdf_fut = pool.submit(
                    self.retriever.retrieve_candidates, rewritten, metadata_filter
                )
                web_fut = (
                    pool.submit(self._web_scrape_chunks, rewritten)
                    if (use_web_fallback and self.cfg.always_scrape_web)
                    else None
                )
                pdf_candidates, bm25_ids, dense_ids = pdf_fut.result()
                web_chunks = web_fut.result() if web_fut else []

            trace.t_retrieval = time.time()

            if use_web_fallback and not self.cfg.always_scrape_web and not web_chunks:
                top_pdf_score = pdf_candidates[0].get("rrf_score", 0.0) if pdf_candidates else 0.0
                if top_pdf_score < self.cfg.min_retrieval_score:
                    log.info("[Query] PDF score low — triggering web fallback")
                    web_chunks = self._web_scrape_chunks(rewritten)

            seen_ids: set[str] = set()
            all_candidates: list[dict] = []
            for candidate in pdf_candidates:
                cid = candidate.get("chroma_id", "")
                key = f"pdf:{cid}" if cid else f"pdf:{candidate.get('text', '')[:80]}"
                if key not in seen_ids:
                    seen_ids.add(key)
                    all_candidates.append(candidate)

            for web_chunk in web_chunks[:self.cfg.web_top_k]:
                key = f"web:{web_chunk.get('text', '')[:120]}"
                if key not in seen_ids:
                    seen_ids.add(key)
                    all_candidates.append(web_chunk)

            log.info(
                "[Query] Merged candidates: %d PDF + %d web = %d total",
                len(pdf_candidates), len(web_chunks), len(all_candidates),
            )

            # Exactly one cross-encoder pass, after all sources are merged.
            chunks = self.reranker.rerank(
                rewritten,
                all_candidates,
                self.cfg.top_k_rerank,
                self.cfg.min_rerank_score,
            )

            # Expand only after reranking so parent context does not distort
            # candidate ranking and the cross-encoder scores remain child-level.
            pdf_winners = [c for c in chunks if c.get("source_type") != "web" and c.get("parent_id")]
            web_winners = [c for c in chunks if c.get("source_type") == "web" or not c.get("parent_id")]
            expanded_pdf = self.retriever.expand_to_context(pdf_winners)

            # Preserve reranked web chunks alongside expanded PDF context.
            chunks = expanded_pdf + web_winners
            chunks = sorted(
                chunks,
                key=lambda c: c.get("rerank_score", 0.0),
                reverse=True,
            )[:self.cfg.top_k_rerank]
            trace.t_rerank = time.time()
            log.info(
                "[Query] After rerank/context expansion: %d chunks | top score: %s",
                len(chunks),
                chunks[0].get("rerank_score") if chunks else "n/a",
            )

            safe_chunks = [
                {k: v for k, v in c.items()
                 if isinstance(v, (str, int, float, bool, type(None)))}
                for c in chunks
            ]
            self.cache.set_retrieval(rewritten, safe_chunks)

        rerank_scores = [c.get("rerank_score", 0.0) for c in chunks if "rerank_score" in c]
        trace.num_chunks_retrieved = len(chunks)
        trace.mean_rerank_score = (
            round(sum(rerank_scores) / len(rerank_scores), 4) if rerank_scores else 0.0
        )
        trace.top_rerank_score = max(rerank_scores) if rerank_scores else 0.0
        trace.bm25_overlap = len(bm25_ids & dense_ids)

        context = self._format_context(chunks)
        log.info("[Query] Generating answer over %d chunks...", len(chunks))
        answer = self._rag_chain.invoke({"context": context, "question": rewritten}).strip()
        trace.t_generation = time.time()

        verdict, claims, faith_score = self.critic.check(context, answer)
        log.info("[Query] Critic: %s | faithfulness=%.2f", verdict, faith_score)
        if verdict == "HALLUCINATED":
            answer = self.critic.repair(context, answer, claims)
            # The pre-repair score remains the honest measured score. We do not
            # make a second critic call just to increase latency.
            log.info("[Query] Surgical repair applied")

        answer = self.critic.polish(answer)
        trace.answer_faithfulness = faith_score

        sources = [
            {
                "filename": Path(c.get("filename", "web")).name,
                "page": c.get("page_number", 0),
                "rerank_score": round(c.get("rerank_score", 0.0), 3),
            }
            for c in chunks
        ]
        self.cache.set_answer(question, query_emb, answer, sources)

        trace.t_end = time.time()
        self.metrics.record(trace)

        # ── 13. Auto-label rewrite from faithfulness score ────────────────────
        self.rewriter.record_answer_score(rewrite_id, faith_score)
        if faith_score >= self.cfg.rewriter_helpful_min_score:
            self.rewriter.record_feedback(rewrite_id, helpful=True)
        elif faith_score < self.cfg.rewriter_unhelpful_max_score:
            self.rewriter.record_feedback(rewrite_id, helpful=False)
        # Scores between the two thresholds get no auto-label — awaits user rating

        self._query_count += 1
        drift_alert = None
        if self._query_count % self.cfg.drift_window == 0:
            drift_alert = self.metrics.check_drift()

        log.info("[Query] Done in %.0fms", trace.total_ms())
        return {
            "answer":          answer,
            "sources":         sources,
            "query_id":        trace.query_id,
            "rewrite_id":      rewrite_id,
            "rewritten_query": rewritten,
            "from_cache": False,
            "drift_alert": drift_alert,
            "metrics": {
                "total_ms": trace.total_ms(),
                "rewrite_ms": trace.latency_ms(trace.t_start, trace.t_rewrite),
                "retrieval_ms": trace.latency_ms(trace.t_rewrite, trace.t_retrieval),
                "generation_ms": trace.latency_ms(trace.t_rerank, trace.t_generation),
                "top_rerank_score": trace.top_rerank_score,
                "mean_rerank_score": trace.mean_rerank_score,
                "faithfulness": faith_score,
                "chunks_used": len(chunks),
                "bm25_overlap": trace.bm25_overlap,
                "retrieval_cached": trace.retrieval_cache_hit,
            },
        }

    def rate_answer(self, query_id: str, rating: int, rewrite_id: Optional[int] = None) -> None:
        assert 1 <= rating <= 5, "Rating must be 1–5"
        self.metrics.record_user_rating(query_id, rating)
        if rewrite_id is not None:
            self.rewriter.record_feedback(rewrite_id, helpful=(rating >= 3))
        log.info("[Pipeline] Rating: %d/5 for query_id=%s", rating, query_id)

    def monitoring_report(self, last_n: int = 50) -> dict:
        report = {
            "performance": self.metrics.report(last_n),
            "cache": self.cache.stats(),
            "rewriter": self.rewriter.rewrite_stats(),
        }
        if self.web_store is not None:
            report["web_store"] = self.web_store.stats()
        return report

    def trigger_drift_check(self) -> Optional[str]:
        return self.metrics.check_drift()

    def rebuild_bm25(self) -> None:
        log.info("[Pipeline] Rebuilding BM25 index...")
        if self.bm25:
            self.bm25.rebuild()
        log.info("[Pipeline] BM25 rebuild complete.")

    def _embed_query(self, text: str) -> np.ndarray:
        # Use the same embedding endpoint as ingestion so cache vectors and
        # Chroma document vectors always share one dimensionality.
        vector = self.embeddings.embed_documents([text])[0]
        return np.asarray(vector, dtype=np.float32)

    def _format_context(self, chunks: list[dict]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = Path(chunk.get("filename", "unknown")).name
            page = chunk.get("page_number", "?")
            score = chunk.get("rerank_score", 0.0)
            text = chunk.get("text", chunk.get("text_preview", ""))
            section = chunk.get("section_path", "")
            section_label = f" | section {section}" if section else ""
            parts.append(
                f"[Source {i} | {source} | page {page}{section_label} | score {score:.2f}]\n{text}"
            )
        return "\n\n---\n\n".join(parts)

    def _web_scrape_chunks(self, query: str) -> list[dict]:
        assert self.web_store is not None, "Call await setup() before query()."
        log.info("[WebScrape] Querying DDG: '%s'", query)
        raw = []
        for attempt in range(self.cfg.ddg_retries):
            try:
                with DDGS() as ddgs:
                    raw = list(ddgs.text(query, max_results=self.cfg.max_scrape_urls))
                if raw:
                    break
                time.sleep(2 ** attempt)
            except Exception as exc:
                log.warning("[WebScrape] DDG attempt %d failed: %s", attempt + 1, exc)
                time.sleep(2 ** attempt)

        if raw:
            approved = sorted(
                [r for r in raw if _score_domain(r.get("href", "")) >= self.cfg.min_domain_score],
                key=lambda r: _score_domain(r.get("href", "")),
                reverse=True,
            )
            new_chunks = 0
            for result in approved:
                url = result.get("href", "")
                title = result.get("title", "Web")
                if not url or self.web_store.is_fresh(url):
                    continue
                text = self._scrape_url(url)
                new_chunks += self.web_store.upsert(url, title, text)
            log.info("[WebScrape] %d new chunks added to persistent store", new_chunks)
        else:
            log.warning("[WebScrape] No DDG results — searching existing cache only.")

        return self.web_store.search(query, k=self.cfg.web_top_k)

    def _scrape_url(self, url: str, char_limit: int = 2500) -> str:
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"},
                timeout=10,
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            main = soup.find("main") or soup.find("article") or soup.body
            text = main.get_text(separator="\n", strip=True) if main else ""
            return text[:char_limit] + ("\n[truncated]" if len(text) > char_limit else "")
        except requests.exceptions.Timeout:
            return "Error: timeout"
        except requests.exceptions.HTTPError as exc:
            return f"Error: HTTP {exc.response.status_code}"
        except Exception as exc:
            return f"Error: {exc}"
