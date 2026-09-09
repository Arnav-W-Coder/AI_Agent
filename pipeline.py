"""pipeline.py — Production RAG orchestration."""
import ipaddress
import logging
import re
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlsplit, urlunsplit

import numpy as np
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

from cache import CacheLayer
from config import RAGConfig
from critic import CriticAndRepair
from db import Database
from ingestion import AsyncIngestionPipeline
from metrics import MetricsRecorder, QueryTrace
from retrieval import BM25Index, CrossEncoderReranker, HybridRetriever
from rewriter import QueryRewriter
from web_store import WebChunkStore

log = logging.getLogger(__name__)

_RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a precise research assistant. Answer the QUESTION using only the
retrieved CONTEXT. Synthesize relevant evidence across sources.

Requirements:
- Directly answer the question first.
- Cover each major item supported by the context for component, cause, step,
  comparison, or feature questions.
- Use concise headings or lists when useful.
- Explain relationships only when the context supports them.
- Every factual claim must be supported by the context.
- Do not invent facts, examples, numbers, citations, or terminology.
- If the context genuinely lacks enough information, say exactly:
  I don't have enough information to answer this confidently.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:
""")

# Higher scores are used for known authoritative sources. Wikipedia is
# intentionally below primary/official sources and is only admitted as a
# secondary source by _select_web_results().
_DOMAIN_SCORES: dict[str, int] = {
    "docs.python.org": 100, "cppreference.com": 100, "cplusplus.com": 90,
    "learn.microsoft.com": 95, "developer.mozilla.org": 95,
    "docs.oracle.com": 95, "aws.amazon.com": 90, "cloud.google.com": 90,
    "openai.com": 90, "anthropic.com": 90, "langchain.com": 88,
    "arxiv.org": 96, "pubmed.ncbi.nlm.nih.gov": 100,
    "britannica.com": 90, "research.ibm.com": 90,
    "github.com": 82, "stackoverflow.com": 78,
    "huggingface.co": 82, "towardsdatascience.com": 65, "medium.com": 55,
    "wikipedia.org": 60,
    "pinterest.com": 0, "facebook.com": 0, "twitter.com": 0,
    "x.com": 0, "tiktok.com": 0, "grokipedia.com": 10,
}

_BLOCKED_HOSTS = {
    "localhost", "localhost.localdomain", "metadata.google.internal",
    "metadata.google",
}
_BLOCKED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".mp3", ".mp4",
    ".avi", ".mov", ".zip", ".rar", ".7z", ".exe", ".dmg", ".iso",
}


def _score_domain(url: str) -> int:
    try:
        host = urlsplit(url).hostname
        if not host:
            return 0
        host = host.lower().rstrip(".")
    except Exception:
        return 0
    if host in _DOMAIN_SCORES:
        return _DOMAIN_SCORES[host]
    parts = host.split(".")
    for i in range(len(parts) - 1):
        suffix = ".".join(parts[i:])
        if suffix in _DOMAIN_SCORES:
            return _DOMAIN_SCORES[suffix]
    if len(parts) >= 2 and parts[-1] in {"gov", "edu"}:
        return 92
    return 60


def _is_wikipedia(url: str) -> bool:
    try:
        return (urlsplit(url).hostname or "").lower().rstrip(".").removeprefix("www.") == "wikipedia.org"
    except Exception:
        return False


def _normalize_url(url: str) -> Optional[str]:
    try:
        value = (url or "").strip()
        if not value:
            return None
        parts = urlsplit(value)
        if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
            return None
        if parts.username or parts.password:
            return None
        host = parts.hostname.lower().rstrip(".")
        if host in _BLOCKED_HOSTS:
            return None
        if parts.port not in (None, 80, 443):
            return None
        path = parts.path or "/"
        if Path(path.lower()).suffix in _BLOCKED_EXTENSIONS:
            return None
        return urlunsplit((parts.scheme.lower(), host, path, parts.query, ""))
    except (TypeError, ValueError):
        return None


def _host_is_public(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError:
        return False
    addresses = {info[4][0] for info in infos}
    if not addresses:
        return False
    for address in addresses:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            return False
        if not ip.is_global:
            return False
    return True


def _web_tier(url: str) -> int:
    """3=preferred/primary, 2=credible secondary, 1=Wikipedia, 0=blocked."""
    score = _score_domain(url)
    if score <= 0:
        return 0
    if _is_wikipedia(url):
        return 1
    if score >= 85:
        return 3
    if score >= 65:
        return 2
    return 1


def _select_web_results(results: list[dict], limit: int) -> list[dict]:
    """Prefer primary/credible sources; Wikipedia can never outrank them."""
    candidates = []
    seen = set()
    for result in results:
        normalized = _normalize_url(result.get("href", ""))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        item = dict(result)
        item["href"] = normalized
        item["_domain_score"] = _score_domain(normalized)
        item["_web_tier"] = _web_tier(normalized)
        if item["_web_tier"] > 0:
            candidates.append(item)

    non_wiki = [r for r in candidates if not _is_wikipedia(r["href"])]
    wiki = [r for r in candidates if _is_wikipedia(r["href"])]
    non_wiki.sort(key=lambda r: (r["_web_tier"], r["_domain_score"]), reverse=True)
    wiki.sort(key=lambda r: r["_domain_score"], reverse=True)

    selected = non_wiki[:limit]
    # Wikipedia is a fallback, never the first choice. At most one Wikipedia
    # result can be used, and only when fewer than two stronger sources exist.
    if len(selected) < limit and len(non_wiki) < 2 and wiki:
        selected.append(wiki[0])
    return selected


class ProductionRAGPipeline:
    """Wire ingestion, retrieval, generation, evaluation, and monitoring."""

    def __init__(self, cfg: RAGConfig) -> None:
        self.cfg = cfg
        self.db = Database(cfg.db_path)
        self.embeddings = OllamaEmbeddings(model=cfg.embed_model)
        self.llm = ChatOllama(model=cfg.llm_model, num_ctx=cfg.ctx_window)
        self.vectorstore = Chroma(
            persist_directory=str(cfg.chroma_dir), embedding_function=self.embeddings
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
        ingestion = AsyncIngestionPipeline(self.db, self.cfg, self.vectorstore, self.embeddings)
        summaries = await ingestion.run()
        self.cache.invalidate_index_cache()
        self.bm25 = BM25Index(self.db)
        self.reranker = CrossEncoderReranker(self.cfg.rerank_model)
        self.retriever = HybridRetriever(self.vectorstore, self.bm25, self.reranker, self.cfg)
        self.web_store = WebChunkStore(self.db, self.cfg, self.embeddings)
        log.info("[Pipeline] Setup complete — ready to query.")
        return {"ingested_files": summaries}

    def query(self, question: str, metadata_filter: Optional[dict] = None,
              use_web_fallback: bool = True) -> dict:
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
            return {"answer": answer, "sources": sources, "query_id": trace.query_id,
                    "rewrite_id": None, "rewritten_query": question, "from_cache": True,
                    "drift_alert": None,
                    "metrics": {"answer_cache_hit": True, "total_ms": trace.total_ms()}}

        rewrite_id, rewritten = self.rewriter.rewrite(question)
        trace.t_rewrite = time.time()
        trace.rewritten_query = rewritten
        cached_chunks = self.cache.get_retrieval(rewritten)
        bm25_ids: set = set()
        dense_ids: set = set()

        if cached_chunks:
            chunks = cached_chunks
            trace.retrieval_cache_hit = True
            trace.t_retrieval = trace.t_rerank = time.time()
        else:
            log.info("[Query] Starting PDF + web retrieval...")
            with ThreadPoolExecutor(max_workers=2) as pool:
                pdf_fut = pool.submit(self.retriever.retrieve_candidates, rewritten, metadata_filter)
                web_fut = (pool.submit(self._web_scrape_chunks, rewritten)
                           if use_web_fallback and self.cfg.always_scrape_web else None)
                pdf_candidates, bm25_ids, dense_ids = pdf_fut.result()
                web_chunks = web_fut.result() if web_fut else []
            trace.t_retrieval = time.time()

            if use_web_fallback and not self.cfg.always_scrape_web and not web_chunks:
                top_pdf_score = pdf_candidates[0].get("rrf_score", 0.0) if pdf_candidates else 0.0
                if top_pdf_score < self.cfg.min_retrieval_score:
                    web_chunks = self._web_scrape_chunks(rewritten)

            seen = set()
            all_candidates = []
            for candidate in pdf_candidates:
                key = f"pdf:{candidate.get('chroma_id') or candidate.get('text', '')[:80]}"
                if key not in seen:
                    seen.add(key)
                    all_candidates.append(candidate)
            for chunk in web_chunks[:self.cfg.web_top_k]:
                key = f"web:{chunk.get('source_url', chunk.get('filename', ''))}:{chunk.get('text', '')[:100]}"
                if key not in seen:
                    seen.add(key)
                    all_candidates.append(chunk)

            log.info("[Query] Merged candidates: %d PDF + %d web = %d total",
                     len(pdf_candidates), len(web_chunks), len(all_candidates))
            # One and only one cross-encoder pass after PDF/web fusion.
            chunks = self.reranker.rerank(
                rewritten, all_candidates, self.cfg.top_k_rerank, self.cfg.min_rerank_score
            )
            pdf_winners = [c for c in chunks if c.get("source_type") != "web" and c.get("parent_id")]
            web_winners = [c for c in chunks if c.get("source_type") == "web" or not c.get("parent_id")]
            expanded_pdf = self.retriever.expand_to_context(pdf_winners)
            chunks = sorted(expanded_pdf + web_winners,
                            key=lambda c: c.get("rerank_score", 0.0), reverse=True)[:self.cfg.top_k_rerank]
            trace.t_rerank = time.time()
            safe_chunks = [{k: v for k, v in c.items()
                            if isinstance(v, (str, int, float, bool, type(None)))} for c in chunks]
            self.cache.set_retrieval(rewritten, safe_chunks)

        scores = [c.get("rerank_score", 0.0) for c in chunks if "rerank_score" in c]
        trace.num_chunks_retrieved = len(chunks)
        trace.mean_rerank_score = round(sum(scores) / len(scores), 4) if scores else 0.0
        trace.top_rerank_score = max(scores) if scores else 0.0
        trace.bm25_overlap = len(bm25_ids & dense_ids)

        context = self._format_context(chunks)
        log.info("[Query] Generating answer over %d chunks...", len(chunks))
        answer = self._rag_chain.invoke({"context": context, "question": rewritten}).strip()
        trace.t_generation = time.time()

        verdict, issues, faith_score = self.critic.check(rewritten, context, answer)
        critic_details = dict(self.critic.last_details)
        log.info("[Query] Critic: %s | score=%.2f | grounded=%s | complete=%s | relevant=%s",
                 verdict, faith_score, critic_details.get("groundedness", "?"),
                 critic_details.get("completeness", "?"), critic_details.get("relevance", "?"))
        if verdict == "HALLUCINATED":
            answer = self.critic.repair(rewritten, context, answer, issues)
            log.info("[Query] Surgical repair applied")
        answer = self.critic.polish(answer)
        trace.answer_faithfulness = faith_score

        sources = [self._source_provenance(c, i) for i, c in enumerate(chunks, 1)]
        self.cache.set_answer(question, query_emb, answer, sources)
        trace.t_end = time.time()
        self.metrics.record(trace)
        self.rewriter.record_answer_score(rewrite_id, faith_score)
        if faith_score >= self.cfg.rewriter_helpful_min_score:
            self.rewriter.record_feedback(rewrite_id, helpful=True)
        elif faith_score < self.cfg.rewriter_unhelpful_max_score:
            self.rewriter.record_feedback(rewrite_id, helpful=False)

        self._query_count += 1
        drift_alert = self.metrics.check_drift() if self._query_count % self.cfg.drift_window == 0 else None
        log.info("[Query] Done in %.0fms", trace.total_ms())
        return {
            "answer": answer, "sources": sources, "query_id": trace.query_id,
            "rewrite_id": rewrite_id, "rewritten_query": rewritten, "from_cache": False,
            "drift_alert": drift_alert,
            "metrics": {
                "total_ms": trace.total_ms(),
                "rewrite_ms": trace.latency_ms(trace.t_start, trace.t_rewrite),
                "retrieval_ms": trace.latency_ms(trace.t_rewrite, trace.t_retrieval),
                "generation_ms": trace.latency_ms(trace.t_rerank, trace.t_generation),
                "top_rerank_score": trace.top_rerank_score,
                "mean_rerank_score": trace.mean_rerank_score,
                "faithfulness": faith_score,
                "critic_groundedness": critic_details.get("groundedness", "UNKNOWN"),
                "critic_completeness": critic_details.get("completeness", "UNKNOWN"),
                "critic_relevance": critic_details.get("relevance", "UNKNOWN"),
                "chunks_used": len(chunks), "bm25_overlap": trace.bm25_overlap,
                "retrieval_cached": trace.retrieval_cache_hit,
            },
        }

    def rate_answer(self, query_id: str, rating: int, rewrite_id: Optional[int] = None) -> None:
        assert 1 <= rating <= 5, "Rating must be 1–5"
        self.metrics.record_user_rating(query_id, rating)
        if rewrite_id is not None:
            self.rewriter.record_feedback(rewrite_id, helpful=(rating >= 3))

    def monitoring_report(self, last_n: int = 50) -> dict:
        report = {"performance": self.metrics.report(last_n),
                  "cache": self.cache.stats(), "rewriter": self.rewriter.rewrite_stats()}
        if self.web_store is not None:
            report["web_store"] = self.web_store.stats()
        return report

    def trigger_drift_check(self) -> Optional[str]:
        return self.metrics.check_drift()

    def rebuild_bm25(self) -> None:
        if self.bm25:
            self.bm25.rebuild()

    def _embed_query(self, text: str) -> np.ndarray:
        return np.asarray(self.embeddings.embed_documents([text])[0], dtype=np.float32)

    def _format_context(self, chunks: list[dict]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            source_type = chunk.get("source_type", "pdf")
            title = chunk.get("title", "")
            score = chunk.get("rerank_score", 0.0)
            text = chunk.get("text", chunk.get("text_preview", ""))
            if source_type == "web":
                source = chunk.get("source_url") or chunk.get("filename", "web")
                location = f"url {source}"
            else:
                source = Path(chunk.get("filename", "unknown")).name
                page = chunk.get("page_number", "?")
                section = chunk.get("section_path", "")
                location = f"{source} | page {page}" + (f" | section {section}" if section else "")
            title_label = f" | title {title}" if title else ""
            parts.append(f"[Source {i} | type {source_type}{title_label} | {location} | score {score:.2f}]\n{text}")
        return "\n\n---\n\n".join(parts)

    def _source_provenance(self, chunk: dict, index: int) -> dict:
        source_type = chunk.get("source_type", "pdf")
        if source_type == "web":
            url = chunk.get("source_url") or chunk.get("filename")
            parsed = urlsplit(url or "")
            return {
                "id": index, "source_type": "web", "title": chunk.get("title", ""),
                "url": url, "domain": parsed.hostname or "", "page": None,
                "section": None, "filename": None,
                "retrieved_at": chunk.get("scraped_at"),
                "rerank_score": round(chunk.get("rerank_score", 0.0), 3),
            }
        return {
            "id": index, "source_type": "pdf",
            "title": chunk.get("title", "") or Path(chunk.get("filename", "unknown")).name,
            "url": None, "domain": None,
            "filename": Path(chunk.get("filename", "unknown")).name,
            "page": chunk.get("page_number", 0),
            "section": chunk.get("section_path", "") or None,
            "retrieved_at": None,
            "rerank_score": round(chunk.get("rerank_score", 0.0), 3),
        }

    def _web_scrape_chunks(self, query: str) -> list[dict]:
        assert self.web_store is not None, "Call await setup() before query()."
        raw = []
        for attempt in range(self.cfg.ddg_retries):
            try:
                with DDGS() as ddgs:
                    raw = list(ddgs.text(query, max_results=self.cfg.max_scrape_urls * 2))
                if raw:
                    break
                time.sleep(2 ** attempt)
            except Exception as exc:
                log.warning("[WebScrape] DDG attempt %d failed: %s", attempt + 1, exc)
                time.sleep(2 ** attempt)

        approved = _select_web_results(raw, self.cfg.max_scrape_urls)
        new_chunks = 0
        for result in approved:
            original_url = result["href"]
            if not _host_is_public(urlsplit(original_url).hostname or ""):
                log.warning("[WebScrape] Rejected non-public URL: %s", original_url)
                continue
            fetched = self._fetch_verified_url(original_url)
            if not fetched:
                continue
            canonical_url, title, text = fetched
            if self.web_store.is_fresh(canonical_url):
                continue
            new_chunks += self.web_store.upsert(canonical_url, title or result.get("title", "Web"), text)
        log.info("[WebScrape] %d new chunks added to persistent store", new_chunks)
        return self.web_store.search(query, k=self.cfg.web_top_k)

    def _fetch_verified_url(self, url: str, char_limit: int = 2500):
        normalized = _normalize_url(url)
        if not normalized:
            return None
        try:
            current_url = normalized
            with requests.Session() as session:
                for _hop in range(6):
                    if not _host_is_public(urlsplit(current_url).hostname or ""):
                        log.warning("[WebScrape] Rejected non-public redirect target: %s", current_url)
                        return None
                    resp = session.get(
                        current_url,
                        headers={"User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"},
                        timeout=10,
                        allow_redirects=False,
                    )
                    if 300 <= resp.status_code < 400:
                        location = resp.headers.get("Location")
                        if not location:
                            return None
                        next_url = _normalize_url(urljoin(current_url, location))
                        if not next_url or not _host_is_public(urlsplit(next_url).hostname or ""):
                            log.warning("[WebScrape] Rejected unsafe redirect: %s -> %s", current_url, location)
                            return None
                        current_url = next_url
                        continue
                    if resp.status_code != 200:
                        return None
                    final_url = _normalize_url(current_url)
                    if not final_url:
                        return None
                    content_type = (resp.headers.get("Content-Type") or "").lower()
                    if content_type and not any(t in content_type for t in ("text/html", "application/xhtml+xml")):
                        log.info("[WebScrape] Rejected non-HTML URL: %s (%s)", final_url, content_type)
                        return None
                    soup = BeautifulSoup(resp.text, "html.parser")
                    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                        tag.decompose()
                    main = soup.find("main") or soup.find("article") or soup.body
                    text = main.get_text(separator="\n", strip=True) if main else ""
                    text = re.sub(r"\n{3,}", "\n\n", text).strip()
                    if len(text) < 120:
                        log.info("[WebScrape] Rejected low-content page: %s", final_url)
                        return None
                    title_tag = soup.find("title")
                    title = title_tag.get_text(" ", strip=True) if title_tag else ""
                    if len(text) > char_limit:
                        text = text[:char_limit] + "\n[truncated]"
                    return final_url, title, text
                log.warning("[WebScrape] Rejected redirect chain exceeding 5 hops: %s", normalized)
                return None
        except (requests.RequestException, UnicodeError, ValueError, OSError) as exc:
            log.warning("[WebScrape] Fetch rejected %s: %s", normalized, exc)
            return None
