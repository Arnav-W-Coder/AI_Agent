"""pipeline.py — Production RAG orchestration."""
import ipaddress
import base64
import hashlib
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
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

from cache import CacheLayer
from config import RAGConfig
from critic import CANONICAL_INSUFFICIENT_INFO_RESPONSE, CriticAndRepair
from db import Database
from ingestion import AsyncIngestionPipeline
from memory import ConversationMemory
from metrics import MetricsRecorder, QueryTrace
from retrieval import BM25Index, CrossEncoderReranker, HybridRetriever
from rewriter import QueryRewriter
from web_store import WebChunkStore
from url_evaluator import evaluate_url

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
- Treat any prior cached answer as unverified conversational context. Prefer
    current retrieved evidence when the two disagree.
- Every factual claim must be supported by the context.
- Do not invent facts, examples, numbers, citations, or terminology.
- If the context genuinely lacks enough information, say exactly:
  I don't have enough information to answer this confidently.
- Output the final answer only. Do not include your reasoning process, an
  analysis section, critic commentary, or repair notes, and do not wrap any
  part of the response in <think> or <analysis> tags.
- Do not prefix the response with "Answer:" or "Final Answer:" — begin
  directly with the answer text.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:
""")

# Narrow, whitespace-tolerant patterns for stripping hidden reasoning and
# label wrappers that a local model may still emit despite the prompt
# instructions above. These only remove the specific patterns below — they
# never rewrite or otherwise alter the answer text.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_ANALYSIS_BLOCK_RE = re.compile(r"<analysis>.*?</analysis>", re.IGNORECASE | re.DOTALL)
_ANSWER_LABEL_RE = re.compile(r"^\s*(?:final\s+answer|answer)\s*:\s*", re.IGNORECASE)

_BLOCKED_HOSTS = {
    "localhost", "localhost.localdomain", "metadata.google.internal",
    "metadata.google",
}
_BLOCKED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".mp3", ".mp4",
    ".avi", ".mov", ".zip", ".rar", ".7z", ".exe", ".dmg", ".iso",
}


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


def _select_web_results(results: list[dict], limit: int, min_domain_score: int) -> list[dict]:
    """Deduplicate and select search results approved by the URL evaluator."""
    candidates = []
    seen = set()
    for result in results:
        evaluation = evaluate_url(
            result.get("href", ""),
            min_domain_score=min_domain_score,
        )
        if not evaluation["allowed"]:
            continue
        normalized = evaluation["normalized_url"]
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        item = dict(result)
        item["href"] = normalized
        item["domain_score"] = evaluation["domain_score"]
        item["domain_score_reasons"] = evaluation["domain_score_reasons"]
        candidates.append(item)
    return candidates[:limit]


class ProductionRAGPipeline:
    """Wire ingestion, retrieval, generation, evaluation, and monitoring."""

    @staticmethod
    def _classify_query(question: str) -> str:
        """Classify query intent using inexpensive lexical signals."""
        text = question.strip().lower()
        words = set(re.findall(r"[a-z0-9_]+", text))
        if any(marker in text for marker in ("stack trace", "traceback", "error:",
                                              "exception", "not working", "fails", "failure")):
            return "troubleshooting"
        if any(marker in text for marker in ("compare ", "comparison", " versus ", " vs ",
                                              "differences between", "which is better")):
            return "comparison"
        if any(word in words for word in ("best", "recommend", "recommendation", "alternatives")):
            return "recommendation"
        if any(marker in text for marker in ("survey", "state of the art", "literature", "research",
                                              "evidence", "papers", "according to multiple")):
            return "research"
        if any(marker in text for marker in ("how do i", "how to ", "steps to", "guide to",
                                              "tutorial", "install", "configure", "set up")):
            return "how_to"
        if any(marker in text for marker in ("what is ", "what are ", "define ", "definition of ")):
            return "definition"
        if len(words) >= 18 or text.count(",") >= 2 or sum(
            word in words for word in ("with", "without", "including", "using", "for")
        ) >= 2:
            return "multi_constraint"
        if any(marker in text for marker in ("why ", "explain ", "how does ", "how do ")):
            return "explanation"
        return "explanation"

    @staticmethod
    def _is_generation_request(question: str) -> bool:
        """Detect requests to produce or transform an artifact or answer."""
        generation_verbs = {
            "write", "draft", "create", "generate", "implement", "build",
            "research", "summarize", "translate", "rewrite", "refactor",
            "design", "develop", "prepare", "solve",
        }
        words = re.findall(r"[a-z0-9]+", question.lower())
        return any(word in generation_verbs for word in words[:6])

    @staticmethod
    def _generation_retrieval_query(question: str) -> str:
        """Remove artifact instructions while preserving the evidence subject."""
        query = re.sub(r"^\s*(?:please\s+)?", "", question.strip(), flags=re.IGNORECASE)
        query = re.sub(
            r"^(?:write|draft|create|generate|implement|build|research|summarize|"
            r"translate|rewrite|refactor|design|develop|prepare|solve)\s+",
            "",
            query,
            flags=re.IGNORECASE,
        )
        query = re.sub(
            r"^(?:a|an|the)\s+(?:potential\s+)?"
            r"(?:introductory paragraph|introduction|essay|email|story|report|"
            r"implementation|study plan|architecture|summary|translation|"
            r"refactor|design|solution|piece of code)\s+"
            r"(?:for|about|on|of)\s+",
            "",
            query,
            flags=re.IGNORECASE,
        )
        query = re.sub(r"\s+", " ", query).strip(" .?!")
        return query or question.strip()

    @staticmethod
    def _named_options(question: str) -> list[str]:
        """Extract simple comma/and-separated options from comparison queries."""
        subject = re.split(r"\bfor\b", question, maxsplit=1, flags=re.IGNORECASE)[0]
        subject = re.sub(r"^.*?\b(?:compare|comparison of|differences between)\b", "", subject,
                         flags=re.IGNORECASE)
        parts = re.split(r",|\band\b|\bvs\.?\b|\bversus\b", subject, flags=re.IGNORECASE)
        options = []
        for part in parts:
            value = re.sub(r"[^A-Za-z0-9+#. -]", "", part).strip(" -")
            if 2 <= len(value.split()) <= 5 and value.lower() not in {"what", "which"}:
                options.append(value)
        return list(dict.fromkeys(options))

    @staticmethod
    def _recommendation_queries(question: str, rewritten_queries: list[str]) -> list[str]:
        """Build complementary retrieval queries for recommendation intents."""
        original = question.strip()
        rewritten = next(
            (query.strip() for query in rewritten_queries
             if query.strip().lower() != original.lower()),
            original,
        )
        topic = re.sub(
            r"^\s*(?:what are|what is|which are|which is|recommend|suggest)\b",
            "",
            original,
            flags=re.IGNORECASE,
        )
        topic = re.sub(r"\b(?:the )?(?:best|top|recommended)\b", "", topic, flags=re.IGNORECASE)
        topic = re.sub(r"\s+", " ", topic).strip(" ?.") or original
        return list(dict.fromkeys([
            rewritten,
            f"best options for {topic}",
            f"{topic} comparison",
            f"{topic} advantages disadvantages",
            f"{topic} production use cases",
        ]))

    def _route_query(self, question: str, query_type: str,
                     queries: list[str]) -> dict:
        """Build the retrieval plan for a classified query."""
        plan = {
            "queries": list(dict.fromkeys(queries or [question])),
            "top_k": self.cfg.top_k_rerank,
            "candidate_k": self.cfg.top_k_rerank,
            "always_web": self.cfg.always_scrape_web,
            "diversify": False,
        }
        if not self.cfg.query_routing_enabled:
            return plan
        if query_type == "definition":
            plan["queries"] = [question]
            plan["top_k"] = min(self.cfg.top_k_rerank, 3)
        elif query_type == "how_to":
            plan["queries"] = [f"{query} documentation reference guide" for query in plan["queries"]]
        elif query_type == "comparison":
            plan["queries"].extend(f"{option} {question}" for option in self._named_options(question))
            plan["diversify"] = True
            plan["candidate_k"] = max(self.cfg.top_k_rerank * 2, self.cfg.top_k_rerank + 2)
        elif query_type == "recommendation":
            if self.cfg.recommendation_query_expansion_enabled:
                plan["queries"] = self._recommendation_queries(question, plan["queries"])
            else:
                plan["queries"] = [question]
            plan["diversify"] = True
            plan["candidate_k"] = max(self.cfg.top_k_rerank * 2, self.cfg.top_k_rerank + 2)
        elif query_type == "troubleshooting":
            plan["queries"].append(f"{question} exact error message solution")
        elif query_type == "research":
            plan["always_web"] = True
            plan["diversify"] = True
            plan["top_k"] = max(self.cfg.top_k_rerank, 6)
            plan["candidate_k"] = max(plan["top_k"] * 2, plan["top_k"] + 2)
        elif query_type == "multi_constraint":
            plan["diversify"] = True
            plan["candidate_k"] = max(self.cfg.top_k_rerank * 2, self.cfg.top_k_rerank + 2)
        return plan

    @staticmethod
    def _diversify_sources(chunks: list[dict], query_type: str, limit: int,
                           max_per_source: int = 2, max_per_domain: int = 3) -> list[dict]:
        """Select relevant chunks while preserving independent sources."""
        if query_type not in {"comparison", "recommendation", "research", "multi_constraint"}:
            return chunks[:limit]

        remaining = sorted(
            chunks,
            key=lambda chunk: chunk.get("rerank_score", 0.0),
            reverse=True,
        )
        selected = []
        source_counts: dict[str, int] = {}
        domain_counts: dict[str, int] = {}
        score_tolerance = 0.5

        while remaining and len(selected) < limit:
            eligible = []
            for chunk in remaining:
                source = chunk.get("source_url") or chunk.get("filename") or chunk.get("chroma_id") or "unknown"
                domain = urlsplit(source).hostname if "://" in source else source
                if source_counts.get(source, 0) < max_per_source and domain_counts.get(domain, 0) < max_per_domain:
                    eligible.append((chunk, source, domain))
            if not eligible:
                break

            best_score = eligible[0][0].get("rerank_score", 0.0)
            unseen_domain = next(
                (item for item in eligible
                 if domain_counts.get(item[2], 0) == 0
                 and best_score - item[0].get("rerank_score", 0.0) <= score_tolerance),
                None,
            )
            chunk, source, domain = unseen_domain or eligible[0]
            selected.append(chunk)
            source_counts[source] = source_counts.get(source, 0) + 1
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            remaining.remove(chunk)

        return selected

    @staticmethod
    def _query_evidence_terms(question: str) -> set[str]:
        stop_words = {
            "a", "an", "and", "are", "be", "does", "do", "for", "how",
            "is", "me", "of", "please", "the", "to", "what", "which",
        }
        return {
            term for term in re.findall(r"[a-z0-9]+", question.lower())
            if len(term) > 2 and term not in stop_words
        }

    def _has_query_evidence(self, question: str, chunks: list[dict]) -> bool:
        terms = self._query_evidence_terms(question)
        if not terms:
            return bool(chunks)
        evidence = " ".join(
            str(chunk.get("text", chunk.get("text_preview", ""))).lower()
            for chunk in chunks
            if not chunk.get("context_only")
        )
        coverage = sum(term in evidence for term in terms) / len(terms)
        return coverage >= self.cfg.answerability_min_query_term_coverage

    def _answerability_debug(self, chunks: list[dict], question: str) -> dict:
        """Return the individual checks used by the answerability gate."""
        score_chunks = [
            chunk
            for chunk in chunks
            if "rerank_score" in chunk and not chunk.get("context_only")
        ]
        scores = sorted(
            (float(chunk.get("rerank_score", 0.0)) for chunk in score_chunks),
            reverse=True,
        )
        window = max(1, int(self.cfg.answerability_score_window))
        strongest_scores = scores[:window]
        mean_score = (
            sum(strongest_scores) / len(strongest_scores)
            if strongest_scores else 0.0
        )
        return {
            "chunk_count": len(chunks),
            "scoreable_chunk_count": len(score_chunks),
            "top_score": scores[0] if scores else None,
            "all_chunk_mean": sum(scores) / len(scores) if scores else None,
            "strongest_chunk_mean": mean_score,
            "top_score_pass": bool(
                scores
                and scores[0] >= self.cfg.answerability_min_top_score
            ),
            "mean_score_pass": (
                mean_score >= self.cfg.answerability_min_mean_score
            ),
            "query_evidence_pass": self._has_query_evidence(
                question, score_chunks
            ),
        }

    def _is_answerable(self, chunks: list[dict], question: str = "") -> bool:
        """Determine whether retrieved evidence is sufficient for generation."""
        score_chunks = [
            chunk
            for chunk in chunks
            if "rerank_score" in chunk and not chunk.get("context_only")
        ]

        if len(score_chunks) < self.cfg.answerability_min_chunks:
            return False

        scores = sorted(
            (float(chunk.get("rerank_score", 0.0)) for chunk in score_chunks),
            reverse=True,
        )
        top_score = scores[0]
        score_window = max(1, int(self.cfg.answerability_score_window))
        strongest_scores = scores[:score_window]
        mean_score = sum(strongest_scores) / len(strongest_scores)

        scoreable = (
            top_score >= self.cfg.answerability_min_top_score
            and mean_score >= self.cfg.answerability_min_mean_score
        )

        if not scoreable:
            return False

        return not question or self._has_query_evidence(question, score_chunks)

    def _web_evidence_is_relevant(self, question: str,
                                  web_candidates: list[dict]) -> bool:
        """Check web evidence independently before allowing web-backed generation."""
        usable = [chunk for chunk in web_candidates if not chunk.get("context_only")]
        if len(usable) < getattr(self.cfg, "web_min_candidates", 2):
            return False
        terms = self._query_evidence_terms(question)
        if not terms:
            return True
        evidence = " ".join(
            str(chunk.get("text", chunk.get("text_preview", ""))).lower()
            for chunk in usable
        )
        coverage = sum(term in evidence for term in terms) / len(terms)
        return coverage >= self.cfg.web_min_query_relevance

    def _combined_evidence_is_answerable(self, question: str,
                                         candidates: list[dict]) -> bool:
        """Evaluate all retriever sources as one evidence set."""
        return self._is_answerable(candidates, question)

    @staticmethod
    def _retrieval_debug_summary(question: str, rewritten_queries: list[str],
                                 retrieval_calls: list[dict], unique_candidates: int,
                                 chunks: list[dict], answerable: bool,
                                 web_fallback: bool) -> str:
        scores = [
            chunk.get("rerank_score") for chunk in chunks
            if chunk.get("rerank_score") is not None and not chunk.get("context_only")
        ]
        top_scores = sorted((float(score) for score in scores), reverse=True)[:5]
        filenames = [
            Path(str(chunk.get("filename", "unknown"))).name
            for chunk in chunks[:5]
            if chunk.get("filename")
        ]
        return (
            "Retrieval debug summary | "
            f"Original question: {question!r} | "
            f"Rewritten queries: {rewritten_queries!r} | "
            f"Retrieval calls: {retrieval_calls!r} | "
            f"Candidates per query: {[call.get('candidates', 0) for call in retrieval_calls]!r} | "
            f"Unique candidates: {unique_candidates} | "
            f"Top original-question scores: {top_scores!r} | "
            f"Top source filenames: {filenames!r} | "
            f"Answerability decision: {answerable} | "
            f"Web fallback decision: {web_fallback}"
        )

    @staticmethod
    def _insufficient_information_response(question: str) -> str:
        return CANONICAL_INSUFFICIENT_INFO_RESPONSE

    @staticmethod
    def _sanitize_answer(text: str) -> str:
        """Strip hidden reasoning/label wrappers without touching anything else.

        Deliberately narrow: removes only <think>...</think> and
        <analysis>...</analysis> blocks, plus a single leading "Answer:" /
        "Final Answer:" label. It never rewrites, truncates, or otherwise
        changes the substance of the answer.
        """
        if not text:
            return text
        cleaned = _THINK_BLOCK_RE.sub("", text)
        cleaned = _ANALYSIS_BLOCK_RE.sub("", cleaned)
        cleaned = cleaned.strip()
        cleaned = _ANSWER_LABEL_RE.sub("", cleaned)
        return cleaned.strip()

    def _repair_answer(self, question: str, answer: str, critic_result: dict,
                       context: str) -> tuple[str | None, str]:
        """Apply the configured repair policy in one place.

        PASS         -> accept the answer as-is.
        UNCERTAIN    -> preserve the answer; the critic couldn't evaluate it.
        HALLUCINATED, severity "minor" (groundedness PASSed; only
            completeness/relevance were flagged) -> treat as INCOMPLETE and
            preserve the original rather than rewrite a claim that was
            already grounded.
        HALLUCINATED, severity "major" (groundedness FAILed) -> perform a
            substantive repair.
        A repair that fails to produce a usable answer -> preserve the
            original or abstain, per critic_abstain_on_failed_repair.
        """
        if not self.cfg.critic_enabled:
            return answer, "critic_disabled"
        verdict = critic_result.get("verdict")
        if verdict == "PASS":
            return answer, "none"
        if verdict == "UNCERTAIN":
            return answer, "uncertain"
        severity = critic_result.get("severity", "major")
        if verdict == "HALLUCINATED" and severity == "minor":
            return answer, "incomplete_preserved"

        critique = critic_result.get("critique", "")
        if self.cfg.destructive_critic_repair:
            repaired = self.critic.repair_destructive(question, context, answer, critique)
            if repaired:
                return repaired, "destructive"
        if self.cfg.constrained_critic_repair:
            repaired = self.critic.repair_constrained(question, context, answer, critique)
            if repaired:
                return repaired, "constrained"
        if self.cfg.critic_abstain_on_failed_repair:
            return None, "abstain"
        return answer, "unrepaired"

    @staticmethod
    def _page_quality(text: str, title: str, query: str, domain_score: float,
                      meaningful_paragraphs: int) -> tuple[float, float]:
        query_terms = {term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) > 2}
        page_text = f"{title} {text}".lower()
        matched = sum(term in page_text for term in query_terms)
        query_relevance = matched / len(query_terms) if query_terms else 0.0
        length_quality = min(len(text) / 1800.0, 1.0)
        paragraph_quality = min(meaningful_paragraphs / 8.0, 1.0)
        boilerplate_penalty = 0.25 if meaningful_paragraphs <= 1 else 0.0
        content_quality = max(0.0, min(1.0, 0.55 * length_quality + 0.45 * paragraph_quality - boilerplate_penalty))
        page_quality = (
            0.35 * (domain_score / 100.0)
            + 0.30 * content_quality
            + 0.25 * query_relevance
            + 0.10
        )
        return round(min(1.0, page_quality), 4), round(query_relevance, 4)

    def _broader_web_retrieval(self, question: str, queries: list[str],
                               chunks: list[dict], query_type: str,
                               limit: int, rerank_query: Optional[str] = None) -> list[dict]:
        """Retry weak retrieval with broader web evidence before generation."""
        broader_queries = list(dict.fromkeys([
            question,
            *queries,
            f"{question} authoritative sources evidence",
        ]))
        workers = max(1, min(len(broader_queries), int(self.cfg.web_fetch_workers)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(self._web_scrape_chunks, query) for query in broader_queries]
            web_chunks = [chunk for future in futures for chunk in future.result()]

        candidates = []
        seen = set()
        for chunk in [*chunks, *web_chunks]:
            key = f"{chunk.get('source_type', 'pdf')}:{chunk.get('chroma_id') or chunk.get('source_url', chunk.get('filename', ''))}:{chunk.get('text', '')[:100]}"
            if key not in seen:
                seen.add(key)
                candidates.append(chunk)
        reranked = self.reranker.rerank_against_original(
            rerank_query or question,
            candidates,
            max(limit * 2, limit + 2),
            self.cfg.min_rerank_score,
        )
        pdf_winners = [chunk for chunk in reranked if chunk.get("source_type") != "web"]
        web_winners = [chunk for chunk in reranked if chunk.get("source_type") == "web"]
        expanded_pdf = self.retriever.expand_to_context(pdf_winners)
        recovered = sorted(
            expanded_pdf + web_winners,
            key=lambda chunk: chunk.get("rerank_score", 0.0),
            reverse=True,
        )
        return self._diversify_sources(recovered, query_type, limit)

    def __init__(self, cfg: RAGConfig) -> None:
        self.cfg = cfg
        self.db = Database(cfg.db_path)
        self.memory = ConversationMemory(
            self.db,
            max_history_messages=cfg.max_history_messages,
        )
        self.embeddings = OllamaEmbeddings(model=cfg.embed_model)
        self.llm = ChatOllama(model=cfg.llm_model, num_ctx=cfg.ctx_window)
        self.vision_llm = (
            ChatOllama(model=cfg.vlm_model, num_ctx=cfg.ctx_window)
            if cfg.multimodal_enabled and cfg.vlm_generation_enabled else None
        )
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
        self._last_multimodal_usage = {
            "generation_mode": "not_run", "images_attached": 0, "tables_attached": 0,
        }

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

    def _contextualize_query(self, question: str,
                             chat_history: Optional[list[dict | str]] = None) -> tuple[int, str, list[str]]:
        """Resolve follow-ups into a standalone query before retrieval."""
        rewrite_id, queries = self.rewriter.rewrite_queries(
            question, chat_history=chat_history
        )
        standalone = (queries[0] if queries else question).strip() or question.strip()
        log.info("[Query] Contextualized query: %s", standalone)
        return rewrite_id, standalone, queries

    def query(self, question: str, metadata_filter: Optional[dict] = None,
              use_web_fallback: bool = True,
              chat_history: Optional[list[dict | str]] = None,
              conversation_id: Optional[str] = None) -> dict:
        assert self.retriever is not None, "Call await setup() before query()."
        original_question = question
        if conversation_id:
            chat_history = [
                *self.memory.history(conversation_id),
                *(chat_history or []),
            ]
            self.memory.add_message(conversation_id, "user", original_question)
        rewrite_id, contextualized_query, rewritten_queries = self._contextualize_query(
            original_question, chat_history
        )
        question = contextualized_query
        generation_request = (
            self._is_generation_request(original_question)
            or self._is_generation_request(question)
        )
        evidence_query = (
            self._generation_retrieval_query(question)
            if generation_request else question
        )
        trace = QueryTrace(query_text=question)
        log.info("\n[Query] '%s'", question[:80])
        query_emb = self._embed_query(question)
        query_type = self._classify_query(question)
        retrieval_plan = self._route_query(question, query_type, rewritten_queries)
        if generation_request:
            retrieval_plan["queries"] = list(dict.fromkeys([
                *retrieval_plan["queries"],
                evidence_query,
                f"{evidence_query} evidence examples documentation",
            ]))
            retrieval_plan["diversify"] = True
            retrieval_plan["candidate_k"] = max(
                retrieval_plan["candidate_k"],
                self.cfg.top_k_rerank * 2,
            )
        retrieval_queries = retrieval_plan["queries"]
        rerank_query = evidence_query if generation_request else question
        log.info("[Query] Route=%s | retrieval_queries=%d | top_k=%d | web=%s",
             query_type, len(retrieval_queries), retrieval_plan["top_k"],
             retrieval_plan["always_web"])
        rewritten = "\n".join(retrieval_queries)
        if generation_request:
            log.info("[Query] Generation request; evidence query: %s", evidence_query)
        trace.t_rewrite = time.time()
        trace.rewritten_query = rewritten
        answer_cache_key = (
            f"question={question}\nrewritten_query={rewritten}\n"
            f"retrieval_cache_schema_version={self.cfg.retrieval_cache_schema_version}\n"
            f"critic_config_version={self.cfg.critic_config_version}"
        )
        cached = self.cache.get_answer(
            question, query_emb, cache_key=answer_cache_key, include_metadata=True
        )
        cached_answer = None
        cached_sources = []
        if cached:
            cached_answer, cached_sources, _ = cached
            trace.answer_cache_hit = True
            log.info("[Cache] Using cached answer as supplemental conversational context")
        cached_chunks = self.cache.get_retrieval(rewritten)
        bm25_ids: set = set()
        dense_ids: set = set()
        web_scrape_used = False
        retrieval_calls: list[dict] = []
        unique_candidate_count = len(cached_chunks or [])

        if cached_chunks:
            chunks = cached_chunks
            trace.retrieval_cache_hit = True
            trace.t_retrieval = trace.t_rerank = time.time()
        else:
            log.info("[Query] Starting PDF + web retrieval...")
            outer_workers = max(2, int(self.cfg.web_fetch_workers) + 1)
            with ThreadPoolExecutor(max_workers=outer_workers) as pool:
                pdf_futures = {
                    pool.submit(self.retriever.retrieve_candidates, query, metadata_filter): query
                    for query in retrieval_queries
                }
                web_futures = ({
                    pool.submit(self._web_scrape_chunks, query): query
                    for query in retrieval_queries
                } if use_web_fallback and retrieval_plan["always_web"] else {})
                pdf_results = [future.result() for future in pdf_futures]
                web_results = [future.result() for future in web_futures]

                for index, query in enumerate(retrieval_queries):
                    pdf_count = len(pdf_results[index][0]) if index < len(pdf_results) else 0
                    web_count = len(web_results[index]) if index < len(web_results) else 0
                    retrieval_calls.append({
                        "query": query,
                        "pdf_candidates": pdf_count,
                        "web_candidates": web_count,
                        "candidates": pdf_count + web_count,
                    })

            pdf_candidates = []
            bm25_ids = set()
            dense_ids = set()
            for candidates, query_bm25_ids, query_dense_ids in pdf_results:
                pdf_candidates.extend(candidates)
                bm25_ids.update(query_bm25_ids)
                dense_ids.update(query_dense_ids)
            web_chunks = [chunk for results in web_results for chunk in results]
            web_scrape_used = bool(web_futures)
            trace.t_retrieval = time.time()

            seen = set()
            all_candidates = []
            for candidate in pdf_candidates:
                key = f"pdf:{candidate.get('chroma_id') or candidate.get('text', '')[:80]}"
                if key not in seen:
                    seen.add(key)
                    all_candidates.append(candidate)
            for chunk in web_chunks:
                key = f"web:{chunk.get('source_url', chunk.get('filename', ''))}:{chunk.get('text', '')[:100]}"
                if key not in seen:
                    seen.add(key)
                    all_candidates.append(chunk)

            log.info("[Query] Merged candidates: %d PDF + %d web = %d total",
                     len(pdf_candidates), len(web_chunks), len(all_candidates))
            unique_candidate_count = len(all_candidates)
            # One and only one cross-encoder pass after PDF/web fusion.
            chunks = self.reranker.rerank_against_original(
                rerank_query, all_candidates, retrieval_plan["candidate_k"], self.cfg.min_rerank_score
            )
            pdf_winners = [c for c in chunks if c.get("source_type") != "web"]
            web_winners = [c for c in chunks if c.get("source_type") == "web"]
            expanded_pdf = self.retriever.expand_to_context(pdf_winners)
            combined_chunks = sorted(expanded_pdf + web_winners,
                            key=lambda c: c.get("rerank_score", 0.0), reverse=True)
            chunks = self._diversify_sources(combined_chunks, query_type, retrieval_plan["top_k"])
            trace.t_rerank = time.time()

            pdf_evidence = [chunk for chunk in combined_chunks if chunk.get("source_type") != "web"]
            web_evidence = [chunk for chunk in combined_chunks if chunk.get("source_type") == "web"]
            pdf_answerable = self._combined_evidence_is_answerable(question, pdf_evidence)
            combined_answerable = self._combined_evidence_is_answerable(question, combined_chunks)
            if use_web_fallback and not retrieval_plan["always_web"] and not pdf_answerable and not combined_answerable:
                log.info("[Query] Local sources failed answerability; using web fallback")
                chunks = self._broader_web_retrieval(
                    question,
                    retrieval_queries,
                    chunks,
                    query_type,
                    retrieval_plan["top_k"],
                    rerank_query,
                )
                web_scrape_used = True
                trace.t_retrieval = time.time()
                trace.t_rerank = time.time()

            safe_chunks = [{k: v for k, v in c.items()
                            if isinstance(v, (str, int, float, bool, type(None), list))} for c in chunks]
            if self._combined_evidence_is_answerable(question, chunks):
                self.cache.set_retrieval(rewritten, safe_chunks)
            else:
                log.info("[Cache] Skipping weak retrieval result")

        scores = [c.get("rerank_score", 0.0) for c in chunks if "rerank_score" in c]
        trace.num_chunks_retrieved = len(chunks)
        trace.mean_rerank_score = round(sum(scores) / len(scores), 4) if scores else 0.0
        trace.top_rerank_score = max(scores) if scores else 0.0
        trace.bm25_overlap = len(bm25_ids & dense_ids)

        pdf_candidates = [chunk for chunk in chunks if chunk.get("source_type") != "web"]
        web_candidates = [chunk for chunk in chunks if chunk.get("source_type") == "web"]
        pdf_answerable = self._combined_evidence_is_answerable(question, pdf_candidates)
        answerable = self._combined_evidence_is_answerable(question, chunks)
        web_evidence_relevant = self._web_evidence_is_relevant(question, web_candidates)
        low_confidence = (
            trace.top_rerank_score < self.cfg.answerability_min_top_score
            or trace.mean_rerank_score < self.cfg.answerability_min_mean_score
        )
        require_web = low_confidence and self.cfg.low_confidence_requires_web
        web_fallback_requested = (
            not pdf_answerable
            and not answerable
            and use_web_fallback
            and not web_scrape_used
        )
        if web_fallback_requested:
            log.info("[Query] %s; broadening web retrieval",
                     "Low confidence requires web evidence" if require_web
                     else "Answerability gate failed")
            chunks = self._broader_web_retrieval(
                question,
                retrieval_queries,
                chunks,
                query_type,
                retrieval_plan["top_k"],
                rerank_query,
            )
            trace.t_retrieval = time.time()
            scores = [c.get("rerank_score", 0.0) for c in chunks if "rerank_score" in c]
            pdf_candidates = [chunk for chunk in chunks if chunk.get("source_type") != "web"]
            web_candidates = [chunk for chunk in chunks if chunk.get("source_type") == "web"]
            answerable = self._combined_evidence_is_answerable(question, chunks)
            web_evidence_relevant = self._web_evidence_is_relevant(question, web_candidates)
            trace.num_chunks_retrieved = len(chunks)
            trace.mean_rerank_score = round(sum(scores) / len(scores), 4) if scores else 0.0
            trace.top_rerank_score = max(scores) if scores else 0.0
            trace.t_rerank = time.time()
            web_scrape_used = True

        log.info(self._retrieval_debug_summary(
            question,
            retrieval_queries,
            retrieval_calls,
            unique_candidate_count,
            chunks,
            answerable,
            web_fallback_requested,
        ))

        context = self._format_context(chunks)
        if cached_answer:
            cached_context = self._format_cached_answer_context(
                cached_answer, cached_sources
            )
            context = f"{cached_context}\n\n[CURRENTLY RETRIEVED EVIDENCE]\n\n{context}"
        self._last_multimodal_usage = {
            "generation_mode": "not_run", "images_attached": 0, "tables_attached": 0,
        }
        if answerable:
            log.info("[Query] Generating answer over %d chunks...", len(chunks))
            raw_answer = self._generate_multimodal(question, context, chunks).strip()
            answer = self._sanitize_answer(raw_answer)
        else:
            log.info(
                "[Answerability] FAILED: %s",
                self._answerability_debug(chunks, question),
            )
            log.warning("[Query] Answerability gate failed; returning insufficient-information response")
            answer = self._insufficient_information_response(question)
        generated_answer = answer
        trace.t_generation = time.time()
        log.info("[Query] Web scrape used: %s", web_scrape_used)

        critic_result = {
            "verdict": "PASS", "severity": "none", "critique": "",
        }
        repair_mode = "critic_disabled"
        repair_attempts = 0
        if self.cfg.critic_enabled:
            critic_result = self.critic.evaluate(question, answer, context)
            repair_mode = "none"
            max_attempts = max(0, int(self.cfg.critic_max_repair_attempts))
            for attempt in range(max_attempts + 1):
                if critic_result["verdict"] == "PASS":
                    break
                if critic_result["verdict"] == "UNCERTAIN" or attempt >= max_attempts:
                    if critic_result["verdict"] == "HALLUCINATED" and self.cfg.critic_abstain_on_failed_repair:
                        answer = self._insufficient_information_response(question)
                        repair_mode = "abstain"
                    break
                repaired, repair_mode = self._repair_answer(question, answer, critic_result, context)
                repair_attempts += 1
                if repaired is None:
                    answer = self._insufficient_information_response(question)
                    repair_mode = "abstain"
                    break
                answer = self._sanitize_answer(repaired)
                if repair_mode == "incomplete_preserved":
                    # Nothing was rewritten (groundedness already passed), so
                    # re-running the critic on an unchanged answer would only
                    # repeat the same verdict for the cost of another LLM call.
                    critic_result = {**critic_result, "verdict": "PASS"}
                    break
                critic_result = (
                    self.critic.evaluate(question, answer, context)
                    if self.cfg.critic_require_context_grounding
                    else {"verdict": "PASS", "severity": "none", "critique": ""}
                )
            log.info("[Query] Critic verdict=%s severity=%s repair_mode=%s",
                     critic_result["verdict"], critic_result.get("severity"), repair_mode)

        critic_details = dict(self.critic.last_details)
        faith_score = 1.0 if critic_result["verdict"] == "PASS" else 0.0
        if self.cfg.critic_enabled and self.cfg.critic_polish_enabled:
            answer = self._sanitize_answer(self.critic.polish(answer))
        validated = self.cfg.critic_enabled and critic_result["verdict"] == "PASS"
        critic_metadata = {
            "initial_answer": generated_answer,
            "final_answer": answer,
            "verdict": critic_result["verdict"],
            "severity": critic_result.get("severity", "none"),
            "repair_mode": repair_mode,
            "repair_attempts": repair_attempts,
            "validated": validated,
        }
        trace.answer_faithfulness = faith_score

        sources = [self._source_provenance(c, i) for i, c in enumerate(chunks, 1)]
        multimodal_usage = self._usage_from_sources(
            sources,
            generation_mode=self._last_multimodal_usage["generation_mode"],
            images_attached=self._last_multimodal_usage["images_attached"],
            tables_attached=self._last_multimodal_usage["tables_attached"],
        )
        if validated:
            self.cache.set_answer(
                question,
                query_emb,
                answer,
                sources,
                cache_key=answer_cache_key,
                metadata=critic_metadata,
            )
        else:
            log.info("[Cache] Skipping answer cache; critic validation=%s", validated)
        trace.t_end = time.time()
        self.metrics.record(trace)
        if conversation_id:
            self.memory.add_message(conversation_id, "assistant", answer)
        self.rewriter.record_answer_score(rewrite_id, faith_score)
        if faith_score >= self.cfg.rewriter_helpful_min_score:
            self.rewriter.record_feedback(rewrite_id, helpful=True)
        elif faith_score < self.cfg.rewriter_unhelpful_max_score:
            self.rewriter.record_feedback(rewrite_id, helpful=False)

        self._query_count += 1
        drift_alert = self.metrics.check_drift() if self._query_count % self.cfg.drift_window == 0 else None
        log.info("[Query] Done in %.0fms", trace.total_ms())
        return {
            "answer": answer, "sources": sources, "critic": critic_metadata,
            "query_id": trace.query_id,
            "rewrite_id": rewrite_id, "rewritten_query": rewritten, "from_cache": False,
            "drift_alert": drift_alert,
            "conversation_id": conversation_id,
            "multimodal_usage": multimodal_usage,
            "metrics": {
                "total_ms": trace.total_ms(),
                "rewrite_ms": trace.latency_ms(trace.t_start, trace.t_rewrite),
                "retrieval_ms": trace.latency_ms(trace.t_rewrite, trace.t_retrieval),
                "generation_ms": trace.latency_ms(trace.t_rerank, trace.t_generation),
                "top_rerank_score": trace.top_rerank_score,
                "mean_rerank_score": trace.mean_rerank_score,
                "faithfulness": faith_score,
                "critic_groundedness": critic_details.get("groundedness", "N/A"),
                "critic_completeness": critic_details.get("completeness", "N/A"),
                "critic_relevance": critic_details.get("relevance", "N/A"),
                "chunks_used": len(chunks), "bm25_overlap": trace.bm25_overlap,
                "retrieval_cached": trace.retrieval_cache_hit,
                "query_type": query_type,
                "answerable": answerable,
                "low_confidence": low_confidence,
                "web_scrape_used": web_scrape_used,
                "pdf_answerable": pdf_answerable,
                "web_candidates": len(web_candidates),
                "web_evidence_relevant": web_evidence_relevant,
                "multimodal_usage": multimodal_usage,
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

    def _select_image_chunks(self, chunks: list[dict]) -> list[dict]:
        selected = []
        seen = set()
        for chunk in chunks:
            image_path = chunk.get("image_path")
            if not image_path or image_path in seen or not Path(image_path).is_file():
                continue
            seen.add(image_path)
            selected.append(chunk)
            if len(selected) >= self.cfg.image_top_k:
                break
        return selected

    def _generate_multimodal(self, question: str, context: str,
                             chunks: list[dict]) -> str:
        image_chunks = self._select_image_chunks(chunks)
        if not image_chunks or self.vision_llm is None:
            self._last_multimodal_usage = {
                "generation_mode": "text_only", "images_attached": 0, "tables_attached": 0,
            }
            return self._rag_chain.invoke({"context": context, "question": question})
        self._last_multimodal_usage = {
            "generation_mode": "vision", "images_attached": len(image_chunks),
            "tables_attached": sum(bool(chunk.get("has_table")) for chunk in image_chunks),
        }
        content = [{
            "type": "text",
            "text": (
                "Answer the question using the supplied text context and inspect the attached "
                "PDF page images when useful.\n\n"
                f"CONTEXT:\n{context}\n\nQUESTION: {question}"
            ),
        }]
        for chunk in image_chunks:
            encoded = base64.b64encode(Path(chunk["image_path"]).read_bytes()).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": f"data:image/png;base64,{encoded}",
            })
        try:
            response = self.vision_llm.invoke([HumanMessage(content=content)])
            return response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            log.warning("[Generation] Vision model failed; using text-only generation: %s", exc)
            self._last_multimodal_usage["generation_mode"] = "text_only_fallback"
            return self._rag_chain.invoke({"context": context, "question": question})

    @staticmethod
    def _usage_from_sources(sources: list[dict], generation_mode: str,
                            images_attached: int = 0, tables_attached: int = 0) -> dict:
        images_retrieved = sum(bool(source.get("has_image")) for source in sources)
        tables_retrieved = sum(bool(source.get("has_table")) for source in sources)
        return {
            "generation_mode": generation_mode,
            "images_retrieved": images_retrieved,
            "tables_retrieved": tables_retrieved,
            "images_attached": images_attached,
            "tables_attached": tables_attached,
            "used_image": images_attached > 0,
            "used_table": tables_attached > 0,
        }

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
            image_label = " | has image" if chunk.get("image_path") else ""
            parts.append(f"[Source {i} | type {source_type}{title_label} | {location} | score {score:.2f}{image_label}]\n{text}")
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _format_cached_answer_context(answer: str, sources: list[dict]) -> str:
        source_labels = []
        for source in sources:
            label = source.get("title") or source.get("filename") or source.get("url")
            if label:
                source_labels.append(str(label))
        source_note = (
            "\nAssociated prior sources: " + "; ".join(dict.fromkeys(source_labels))
            if source_labels else ""
        )
        return (
            "[PRIOR CACHED ANSWER - UNVERIFIED MODEL OUTPUT; USE ONLY AS "
            "SUPPORTING CONVERSATIONAL CONTEXT]\n\n"
            f"Previous generated answer:\n{answer}{source_note}"
        )

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
                "domain_score": chunk.get("domain_score"),
                "domain_score_reasons": chunk.get("domain_score_reasons", []),
                "page_quality_score": chunk.get("page_quality_score"),
                "retrieval_score": chunk.get("retrieval_score", chunk.get("rerank_score")),
                "query_relevance": chunk.get("query_relevance"),
                "http_status": chunk.get("http_status"),
                "redirect_count": chunk.get("redirect_count"),
                "final_url": chunk.get("final_url") or url,
                "meaningful_paragraphs": chunk.get("meaningful_paragraphs"),
                "rerank_score": round(chunk.get("rerank_score", 0.0), 3),
            }
        return {
            "id": index, "source_type": "pdf",
            "title": chunk.get("title", "") or Path(chunk.get("filename", "unknown")).name,
            "url": None, "domain": None,
            "filename": Path(chunk.get("filename", "unknown")).name,
            "page": chunk.get("page_number", 0),
            "section": chunk.get("section_path", "") or None,
            "has_image": bool(chunk.get("image_path")),
                "has_table": bool(chunk.get("has_table")),
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

        approved = _select_web_results(
            raw,
            self.cfg.max_scrape_urls,
            self.cfg.min_domain_score,
        )
        # new_chunks = 0
        # for result in approved:
        #     original_url = result["href"]
        #     if not _host_is_public(urlsplit(original_url).hostname or ""):
        #         log.warning("[WebScrape] Rejected non-public URL: %s", original_url)
        #         continue
        #     fetched = self._fetch_verified_url(original_url)
        #     if not fetched:
        #         continue
        #     canonical_url, title, text = fetched
        #     if self.web_store.is_fresh(canonical_url):
        #         continue
        #     new_chunks += self.web_store.upsert(canonical_url, title or result.get("title", "Web"), text)
        
        # log.info("[WebScrape] %d new chunks added to persistent store", new_chunks)
        # return self.web_store.search(query, k=self.cfg.web_top_k)

        # Evaluate and deduplicate URLs before making any HTTP requests.
        candidates = []
        seen_urls = set()
        rejected_urls = 0

        for result in approved:
            original_url = result.get("href", "")

            if self.cfg.url_evaluator_enabled:
                decision = evaluate_url(
                    original_url,
                    min_domain_score=self.cfg.min_domain_score,
                )

                if not decision["allowed"]:
                    rejected_urls += 1
                    log.info(
                        "[WebScrape] URL rejected before fetch: %s | reason=%s",
                        original_url,
                        decision["reason"],
                    )
                    continue

                normalized_url = decision["normalized_url"]
            else:
                normalized_url = _normalize_url(original_url)

            if not normalized_url or normalized_url in seen_urls:
                continue

            seen_urls.add(normalized_url)

            result_copy = dict(result)
            result_copy["href"] = normalized_url
            candidates.append(result_copy)

        log.info(
            "[WebScrape] URL evaluation: %d approved, %d rejected, %d unique",
            len(candidates),
            rejected_urls,
            len(seen_urls),
        )

        # Fetch approved URLs concurrently.
        fetch_workers = max(1, int(self.cfg.web_fetch_workers))
        fetched_results = []

        with ThreadPoolExecutor(max_workers=fetch_workers) as pool:
            futures = {
                pool.submit(self._fetch_verified_url, result["href"], query,
                            result.get("domain_score", 0)): result
                for result in candidates
            }

            for future in futures:
                result = futures[future]
                try:
                    fetched = future.result()
                except Exception as exc:
                    log.warning(
                        "[WebScrape] Fetch failed for %s: %s",
                        result["href"],
                        exc,
                    )
                    continue

                if fetched:
                    fetched_results.append((result, fetched))

        # Persist sequentially unless WebChunkStore is explicitly thread-safe.
        new_chunks = 0
        seen_content = set()

        for result, fetched in fetched_results:
            canonical_url, title, text, page_metadata = fetched

            content_hash = hashlib.sha256(re.sub(r"\s+", " ", text.lower()).encode()).hexdigest()
            if content_hash in seen_content:
                log.info("[WebScrape] Skipping duplicate page content: %s", canonical_url)
                continue
            seen_content.add(content_hash)

            if self.web_store.is_fresh(canonical_url):
                continue

            new_chunks += self.web_store.upsert(
                canonical_url,
                title or result.get("title", "Web"),
                text,
                metadata={
                    "domain_score": result.get("domain_score"),
                    "domain_score_reasons": result.get("domain_score_reasons", []),
                    **page_metadata,
                },
            )

        log.info(
            "[WebScrape] Fetched %d/%d approved URLs; %d new chunks added",
            len(fetched_results),
            len(candidates),
            new_chunks,
        )

        return self.web_store.search(query, k=self.cfg.web_top_k)

    def _fetch_verified_url(self, url: str, query: str = "", domain_score: float = 0.0,
                            char_limit: int = 2500):
        # normalized = _normalize_url(url)
        # if not normalized:
        #     return None
        if self.cfg.url_evaluator_enabled:
            decision = evaluate_url(
                url,
                min_domain_score=self.cfg.min_domain_score,
            )

            if not decision["allowed"]:
                log.info(
                    "[WebScrape] Fetch blocked by URL evaluator: %s | reason=%s",
                    url,
                    decision["reason"],
                )
                return None

            normalized = decision.get("normalized_url") or decision.get("url")
        else:
            normalized = _normalize_url(url)

        if not normalized:
            return None
        try:
            current_url = normalized
            with requests.Session() as session:
                # for _hop in range(6):
                for _hop in range(self.cfg.url_evaluator_max_redirects + 1):
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
                        # next_url = _normalize_url(urljoin(current_url, location))
                        # if not next_url or not _host_is_public(urlsplit(next_url).hostname or ""):
                        #     log.warning("[WebScrape] Rejected unsafe redirect: %s -> %s", current_url, location)
                        #     return None
                        next_url = urljoin(current_url, location)

                        if self.cfg.url_evaluator_enabled:
                            redirect_decision = evaluate_url(
                                next_url,
                                min_domain_score=self.cfg.min_domain_score,
                            )

                            if not redirect_decision["allowed"]:
                                log.warning(
                                    "[WebScrape] Rejected unsafe redirect: %s -> %s | reason=%s",
                                    current_url,
                                    next_url,
                                    redirect_decision["reason"],
                                )
                                return None

                            next_url = redirect_decision.get("normalized_url") or redirect_decision.get("url")
                        else:
                            next_url = _normalize_url(next_url)

                        if not next_url or not _host_is_public(
                            urlsplit(next_url).hostname or ""
                        ):
                            log.warning(
                                "[WebScrape] Rejected unsafe redirect: %s -> %s",
                                current_url,
                                next_url,
                            )
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
                    paragraphs = [paragraph for paragraph in text.splitlines() if len(paragraph.split()) >= 5]
                    page_quality, query_relevance = self._page_quality(
                        text, title, query, domain_score, len(paragraphs)
                    )
                    return final_url, title, text, {
                        "page_quality_score": page_quality,
                        "query_relevance": query_relevance,
                        "http_status": resp.status_code,
                        "redirect_count": _hop,
                        "final_url": final_url,
                        "meaningful_paragraphs": len(paragraphs),
                    }
                # log.warning("[WebScrape] Rejected redirect chain exceeding 5 hops: %s", normalized)
                log.warning(
                    "[WebScrape] Rejected redirect chain exceeding %d hops: %s",
                    self.cfg.url_evaluator_max_redirects,
                    normalized,
                )
                return None
        except (requests.RequestException, UnicodeError, ValueError, OSError) as exc:
            log.warning("[WebScrape] Fetch rejected %s: %s", normalized, exc)
            return None