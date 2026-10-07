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
from metrics import MetricsRecorder, QueryTrace, assess_rag_triad
from retrieval import BM25Index, CrossEncoderReranker, HybridRetriever, stable_chunk_sort_key
from rewriter import QueryRewriter
from web_store import WebChunkStore
from url_evaluator import evaluate_url

log = logging.getLogger(__name__)

_PRACTICE_PROMPT = ChatPromptTemplate.from_template("""
You are creating source-grounded practice material.

Use only concepts, formulas, relationships, and terminology supported by the
retrieved CONTEXT. Hypothetical numerical values are allowed, but every problem
must be internally consistent and solvable from the information given plus
relationships explicitly supported by the context.

Requirements:
- Create at most {max_problems} concise practice problems.
- Include all quantities or conditions required to solve each problem.
- Do not invent named problem sets, textbook labels, experimental facts, or
  physical laws that are absent from the context.
- Do not ask for a quantity that requires missing information.
- Avoid contradictions between the stated physical model and requested result.
- Prefer a mix of conceptual and quantitative questions when supported.
- Do not include solutions unless the user asks for them.
- Output the practice material only.

CONTEXT:
{context}

USER REQUEST:
{question}
""")

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


# ProductionRAGPipeline is the orchestrator for the real project.
# It coordinates ingestion, retrieval, query rewriting, answerability checks,
# optional web search, final generation, and critic-based repair. Think of this
# class as the main control loop that turns a user question into evidence,
# evidence into an answer, and that answer into a validated response.
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
        if any(marker in text for marker in (
            "compare ", "comparison", " versus ", " vs ",
            "differences between", "which is better", " whereas ",
        )) or (
            " but " in text
            and any(marker in text for marker in ("why ", "how ", "isn't", "isnt", "aren't", "arent"))
        ):
            return "comparison"
        if any(phrase in text for phrase in (
            "practice problem", "practice problems", "practice question",
            "practice questions", "quiz", "worksheet", "flashcard",
            "flashcards", "study guide", "review questions", "exercises",
        )):
            return "practice"
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
        if any(word in generation_verbs for word in words[:6]):
            return True

        # "Give/make me ..." is only generation when it targets an artifact.
        # This avoids treating ordinary requests such as "give me information"
        # as artifact-generation queries.
        text = question.lower()
        artifact_markers = (
            "practice problem", "practice problems", "practice question",
            "practice questions", "quiz", "worksheet", "flashcard",
            "flashcards", "study guide", "review questions", "exercises",
        )
        soft_generation_verbs = {"give", "make"}
        return (
            any(word in soft_generation_verbs for word in words[:4])
            and any(marker in text for marker in artifact_markers)
        )

    @staticmethod
    def _study_topic_query(query: str) -> str:
        """Strip course identifiers when a more specific study topic remains."""
        text = re.sub(r"\s+", " ", (query or "").strip())
        course_pattern = re.compile(
            r"\b(?:physics|phys|chemistry|chem|biology|bio|mathematics|math|"
            r"calculus|calc|statistics|stats|economics|econ|computer\s+science|cs)"
            r"\s+\d{1,3}[a-z]?\b",
            flags=re.IGNORECASE,
        )
        stripped = course_pattern.sub(" ", text)
        stripped = re.sub(r"\s+", " ", stripped).strip(" -:,.?")
        return stripped or text

    @staticmethod
    def _generation_retrieval_query(question: str) -> str:
        """Remove artifact instructions while preserving the evidence subject."""
        query = re.sub(r"^\s*(?:please\s+)?", "", question.strip(), flags=re.IGNORECASE)
        # Study/practice requests need retrieval over the subject, not over the
        # artifact words themselves ("give", "practice", "problems", etc.).
        query = re.sub(
            r"^(?:give|make|create|generate|write|prepare)\s+(?:me\s+)?"
            r"(?:(?:some|a|an|the)\s+)?"
            r"(?:practice\s+(?:problems?|questions?)|quiz(?:zes)?|worksheet|"
            r"flashcards?|exercises?|study\s+guide|review\s+questions?)\s+"
            r"(?:for|about|on|of)\s+",
            "",
            query,
            flags=re.IGNORECASE,
        )
        query = re.sub(
            r"^(?:write|draft|create|generate|implement|build|research|summarize|"
            r"translate|rewrite|refactor|design|develop|prepare|solve|make|give)\s+",
            "",
            query,
            flags=re.IGNORECASE,
        )
        query = re.sub(
            r"^(?:me\s+)?(?:a|an|the)?\s*(?:potential\s+)?"
            r"(?:introductory paragraph|introduction|essay|email|story|report|"
            r"implementation|study plan|study guide|practice problems?|"
            r"practice questions?|quiz(?:zes)?|worksheet|flashcards?|exercises?|"
            r"architecture|summary|translation|refactor|design|solution|piece of code)\s+"
            r"(?:for|about|on|of)\s+",
            "",
            query,
            flags=re.IGNORECASE,
        )
        query = re.sub(r"\s+", " ", query).strip(" .?!")
        query = ProductionRAGPipeline._study_topic_query(query)
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
        # Query routing decides how many retrieval variants to try and whether the
        # system should search the web as well. Different question shapes benefit
        # from different query expansions: how-to questions need guidance, while
        # comparisons and recommendations need multiple candidate angles.
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
            # Keep natural contrast questions cheap: the original query usually
            # contains all compared entities, so source diversification matters
            # more than generating many paraphrases.
            options = self._named_options(question)
            if options:
                plan["queries"].extend(
                    f"{option} {question}" for option in options[:2]
                )
            plan["queries"] = list(dict.fromkeys(plan["queries"]))[:3]
            plan["diversify"] = True
            plan["candidate_k"] = max(
                self.cfg.top_k_rerank + 4,
                min(self.cfg.top_k_rerank * 2, 12),
            )
        elif query_type == "practice":
            plan["queries"] = [question]
            plan["top_k"] = min(self.cfg.top_k_rerank, 6)
            plan["candidate_k"] = min(max(plan["top_k"] + 4, plan["top_k"]), 10)
            plan["diversify"] = True
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
        """Select relevant chunks while limiting redundant evidence.

        Even simple questions should not spend context budget on many near-duplicate
        chunks from the same URL. Complex queries still receive stronger domain
        diversification through the unseen-domain preference below.
        """
        remaining = sorted(chunks, key=stable_chunk_sort_key)
        selected = []
        source_counts: dict[str, int] = {}
        domain_counts: dict[str, int] = {}
        score_tolerance = 0.5

        while remaining and len(selected) < limit:
            eligible = []
            for chunk in remaining:
                source = chunk.get("source_url") or chunk.get("filename") or chunk.get("chroma_id") or "unknown"
                domain = urlsplit(source).hostname if "://" in source else source
                source_cap = (
                    max_per_source
                    if chunk.get("source_type") == "web"
                    else max(4, max_per_source)
                )
                if source_counts.get(source, 0) < source_cap and domain_counts.get(domain, 0) < max_per_domain:
                    eligible.append((chunk, source, domain))
            if not eligible:
                break

            best_score = eligible[0][0].get("rerank_score", 0.0)
            unseen_domain = None
            if query_type in {"comparison", "recommendation", "research", "multi_constraint"}:
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
    def _normalize_evidence_token(term: str) -> str:
        """Normalize light English morphology for lexical answerability checks."""
        token = (term or "").lower().strip()
        if len(token) <= 3:
            return token
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith(("sses", "xes", "zes", "ches", "shes")) and len(token) > 5:
            return token[:-2]
        if token.endswith("ses") and len(token) > 4:
            return token[:-2]
        if token.endswith("s") and not token.endswith(("ss", "us", "is")) and len(token) > 4:
            return token[:-1]
        return token

    @classmethod
    def _evidence_terms(cls, text: str) -> set[str]:
        """Extract content-bearing normalized terms, excluding query glue/contractions."""
        stop_words = {
            "a", "an", "and", "are", "be", "been", "being", "but", "can",
            "could", "did", "do", "does", "for", "from", "had", "has", "have",
            "how", "is", "isn", "it", "its", "me", "not", "of", "please",
            "should", "than", "that", "the", "their", "them", "then", "there",
            "these", "they", "this", "to", "was", "were", "what", "when",
            "where", "which", "who", "why", "will", "with", "would",
        }
        raw_terms = re.findall(r"[a-z0-9]+", (text or "").lower())
        return {
            cls._normalize_evidence_token(term)
            for term in raw_terms
            if len(term) > 2 and term not in stop_words
        }

    @classmethod
    def _query_evidence_terms(cls, question: str) -> set[str]:
        return cls._evidence_terms(question)

    def _query_evidence_coverage(self, question: str, chunks: list[dict]) -> tuple[float, set[str], set[str]]:
        terms = self._query_evidence_terms(question)
        if not terms:
            return (1.0 if chunks else 0.0), terms, set()
        evidence_terms: set[str] = set()
        for chunk in chunks:
            if chunk.get("context_only"):
                continue
            evidence_terms.update(
                self._evidence_terms(
                    str(chunk.get("text", chunk.get("text_preview", "")))
                )
            )
        matched = terms & evidence_terms
        return len(matched) / len(terms), terms, matched

    def _has_query_evidence(self, question: str, chunks: list[dict]) -> bool:
        coverage, _, _ = self._query_evidence_coverage(question, chunks)
        return coverage >= self.cfg.answerability_min_query_term_coverage

    def _rerank_confidence(self, score_chunks: list[dict]) -> tuple[bool, float | None, float | None]:
        """Evaluate rerank confidence without treating MS MARCO logits as probabilities."""
        scores = sorted(
            (float(chunk.get("rerank_score", 0.0)) for chunk in score_chunks),
            reverse=True,
        )
        if not scores:
            return False, None, None
        window = max(1, int(self.cfg.answerability_score_window))
        strongest = scores[:window]
        top_score = scores[0]
        mean_score = sum(strongest) / len(strongest)
        if getattr(self.cfg, "answerability_use_absolute_rerank_thresholds", False):
            passed = (
                top_score >= self.cfg.answerability_min_top_score
                and mean_score >= self.cfg.answerability_min_mean_score
            )
        else:
            floor = float(self.cfg.min_rerank_score)
            passed = top_score >= floor and mean_score >= floor
        return passed, top_score, mean_score

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
        score_gate_pass, top_score, mean_score_value = self._rerank_confidence(
            score_chunks
        )
        mean_score = mean_score_value if mean_score_value is not None else 0.0
        coverage, query_terms, matched_terms = self._query_evidence_coverage(
            question, score_chunks
        )
        lexical_pass = (
            coverage >= self.cfg.answerability_min_query_term_coverage
        )
        semantic_override_pass = bool(
            getattr(self.cfg, "answerability_semantic_override_enabled", True)
            and top_score is not None
            and top_score >= getattr(
                self.cfg, "answerability_semantic_override_top_score", 1.0
            )
            and mean_score >= getattr(
                self.cfg, "answerability_semantic_override_mean_score", 0.0
            )
            and coverage >= getattr(
                self.cfg, "answerability_semantic_override_min_coverage", 0.20
            )
        )
        return {
            "chunk_count": len(chunks),
            "scoreable_chunk_count": len(score_chunks),
            "top_score": top_score,
            "all_chunk_mean": sum(scores) / len(scores) if scores else None,
            "strongest_chunk_mean": mean_score,
            "query_terms": sorted(query_terms),
            "matched_query_terms": sorted(matched_terms),
            "query_term_coverage": round(coverage, 3),
            "rerank_score_mode": (
                "absolute"
                if getattr(self.cfg, "answerability_use_absolute_rerank_thresholds", False)
                else "model_floor"
            ),
            "rerank_confidence_pass": score_gate_pass,
            "top_score_pass": score_gate_pass if top_score is not None else False,
            "mean_score_pass": score_gate_pass if top_score is not None else False,
            "query_evidence_pass": lexical_pass,
            "semantic_override_pass": semantic_override_pass,
            "answerable_pass": bool(
                len(score_chunks) >= self.cfg.answerability_min_chunks
                and score_gate_pass
                and (lexical_pass or semantic_override_pass)
            ),
        }

    def _is_answerable(self, chunks: list[dict], question: str = "") -> bool:
        """Determine whether retrieved evidence is sufficient for generation."""
        # The answerability check prevents the model from answering with weak or
        # noisy evidence. We require enough relevant chunks and a strong enough
        # top-score signal before we allow generation to proceed.
        score_chunks = [
            chunk
            for chunk in chunks
            if "rerank_score" in chunk and not chunk.get("context_only")
        ]

        if len(score_chunks) < self.cfg.answerability_min_chunks:
            return False

        scoreable, top_score_value, mean_score_value = self._rerank_confidence(
            score_chunks
        )
        if not scoreable or top_score_value is None or mean_score_value is None:
            return False
        top_score = top_score_value
        mean_score = mean_score_value

        if not question:
            return True

        coverage, _, _ = self._query_evidence_coverage(question, score_chunks)
        if coverage >= self.cfg.answerability_min_query_term_coverage:
            return True

        # The cross-encoder is the semantic relevance model; lexical overlap is
        # only a guardrail. Allow strong semantic evidence through when there is
        # still some query-term overlap, rather than forcing exact wording.
        semantic_override = bool(
            getattr(self.cfg, "answerability_semantic_override_enabled", True)
            and top_score >= getattr(
                self.cfg, "answerability_semantic_override_top_score", 1.0
            )
            and mean_score >= getattr(
                self.cfg, "answerability_semantic_override_mean_score", 0.0
            )
            and coverage >= getattr(
                self.cfg, "answerability_semantic_override_min_coverage", 0.20
            )
        )
        return semantic_override

    def _web_evidence_is_relevant(self, question: str,
                                  web_candidates: list[dict]) -> bool:
        """Check web evidence independently before allowing web-backed generation."""
        usable = [chunk for chunk in web_candidates if not chunk.get("context_only")]
        if len(usable) < getattr(self.cfg, "web_min_candidates", 2):
            return False
        coverage, _, _ = self._query_evidence_coverage(question, usable)
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

    def _should_run_critic(self, *, low_confidence: bool, answerable: bool,
                           generation_request: bool = False) -> bool:
        """Respect the fast path while always validating generated artifacts."""
        if not self.cfg.critic_enabled:
            return False
        if (
            generation_request
            and getattr(self.cfg, "critic_on_generation_requests", True)
        ):
            return True
        if not self.cfg.critic_on_low_confidence_only:
            return True
        return bool(low_confidence or not answerable)

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
        HALLUCINATED, severity "minor" (groundedness PASSed) -> repair
            answer-relevance/completeness problems when possible; preserve
            context-relevance-only failures so retrieval can be triaged instead.
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
            details = getattr(self.critic, "last_details", {})
            repairable_focus_issue = (
                details.get("answer_relevance") == "FAIL"
                or details.get("completeness") == "FAIL"
            )
            if not repairable_focus_issue:
                # Low context relevance is a retrieval problem, not a wording problem.
                return answer, "context_issue_preserved"

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
        """Adaptive web fallback: fetch a small batch, re-check, then fetch more only if needed."""
        search_queries = list(dict.fromkeys([
            question,
            *queries,
            f"{question} authoritative source evidence",
        ]))
        current = list(chunks)
        seen_urls = {
            chunk.get("source_url") or chunk.get("filename")
            for chunk in current
            if chunk.get("source_type") == "web"
            and (chunk.get("source_url") or chunk.get("filename"))
        }

        adaptive = bool(getattr(self.cfg, "adaptive_web_enabled", True))
        max_rounds = (
            max(1, int(getattr(self.cfg, "adaptive_web_max_rounds", 2)))
            if adaptive else 1
        )
        initial_sources = max(
            1, int(getattr(self.cfg, "adaptive_web_initial_sources", 2))
        )
        per_round_sources = max(
            1, int(getattr(self.cfg, "adaptive_web_sources_per_round", 2))
        )
        hard_cap = max(1, int(self.cfg.max_scrape_urls))
        network_sources = 0
        rounds = 0
        stop_reason = "max_rounds"
        started = time.perf_counter()

        # Persistent web evidence is cheaper than another search/fetch cycle.
        # Reuse it first, rerank against the current question, and stop immediately
        # if the cached evidence already clears the same answerability gate.
        web_store = getattr(self, "web_store", None)
        cached_web = (
            web_store.search(
                question,
                k=min(self.cfg.web_top_k, max(limit, self.cfg.answerability_min_chunks)),
            )
            if web_store is not None else []
        )
        if cached_web:
            cached_candidates = []
            cached_seen = set()
            for chunk in [*current, *cached_web]:
                key = (
                    f"{chunk.get('source_type', 'pdf')}:"
                    f"{chunk.get('chroma_id') or chunk.get('source_url', chunk.get('filename', ''))}:"
                    f"{chunk.get('text', '')[:100]}"
                )
                if key not in cached_seen:
                    cached_seen.add(key)
                    cached_candidates.append(chunk)
            cached_reranked = self.reranker.rerank_against_original(
                rerank_query or question,
                cached_candidates,
                max(limit * 2, limit + 2),
                self.cfg.min_rerank_score,
            )
            cached_pdf = [
                chunk for chunk in cached_reranked
                if chunk.get("source_type") != "web"
            ]
            cached_web_winners = [
                chunk for chunk in cached_reranked
                if chunk.get("source_type") == "web"
            ]
            cached_expanded_pdf = self.retriever.expand_to_context(cached_pdf)
            cached_recovered = sorted(
                cached_expanded_pdf + cached_web_winners,
                key=stable_chunk_sort_key,
            )
            cached_current = self._diversify_sources(
                cached_recovered, query_type, limit
            )
            if self._combined_evidence_is_answerable(question, cached_current):
                self._last_adaptive_web_usage = {
                    "enabled": adaptive,
                    "rounds": 0,
                    "network_sources": 0,
                    "cache_chunks": len(cached_web),
                    "stop_reason": "web_cache_answerable",
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                }
                log.info("[AdaptiveWeb] %s", self._last_adaptive_web_usage)
                return cached_current
            current = cached_current
            seen_urls.update(
                chunk.get("source_url") or chunk.get("filename")
                for chunk in cached_web
                if chunk.get("source_url") or chunk.get("filename")
            )

        for round_index in range(max_rounds):
            if network_sources >= hard_cap:
                stop_reason = "source_cap"
                break

            requested = (
                initial_sources if round_index == 0 else per_round_sources
            )
            requested = min(requested, hard_cap - network_sources)
            if requested <= 0:
                stop_reason = "source_cap"
                break

            search_query = search_queries[min(round_index, len(search_queries) - 1)]
            web_chunks = self._web_scrape_chunks(
                search_query,
                max_urls=requested,
                exclude_urls=seen_urls,
            )
            rounds += 1
            scrape_stats = getattr(self, "_last_web_scrape_stats", {})
            fetched_urls = scrape_stats.get("fetched_urls", [])
            used_urls = scrape_stats.get("used_urls", fetched_urls)
            network_sources += len(fetched_urls)
            seen_urls.update(used_urls)

            candidates = []
            seen = set()
            for chunk in [*current, *web_chunks]:
                key = (
                    f"{chunk.get('source_type', 'pdf')}:"
                    f"{chunk.get('chroma_id') or chunk.get('source_url', chunk.get('filename', ''))}:"
                    f"{chunk.get('text', '')[:100]}"
                )
                if key not in seen:
                    seen.add(key)
                    candidates.append(chunk)

            if not candidates:
                stop_reason = "no_candidates"
                break

            reranked = self.reranker.rerank_against_original(
                rerank_query or question,
                candidates,
                max(limit * 2, limit + 2),
                self.cfg.min_rerank_score,
            )
            pdf_winners = [
                chunk for chunk in reranked if chunk.get("source_type") != "web"
            ]
            web_winners = [
                chunk for chunk in reranked if chunk.get("source_type") == "web"
            ]
            expanded_pdf = self.retriever.expand_to_context(pdf_winners)
            recovered = sorted(expanded_pdf + web_winners, key=stable_chunk_sort_key)
            current = self._diversify_sources(recovered, query_type, limit)

            if self._combined_evidence_is_answerable(question, current):
                stop_reason = "answerable"
                break
            if not used_urls:
                stop_reason = "no_new_sources"
                break

        self._last_adaptive_web_usage = {
            "enabled": adaptive,
            "rounds": rounds,
            "network_sources": network_sources,
            "cache_chunks": len(cached_web),
            "stop_reason": stop_reason,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        log.info("[AdaptiveWeb] %s", self._last_adaptive_web_usage)
        return current

    def __init__(self, cfg: RAGConfig) -> None:
        self.cfg = cfg
        self.db = Database(cfg.db_path)
        self.memory = ConversationMemory(
            self.db,
            max_history_messages=cfg.max_history_messages,
        )
        self.embeddings = OllamaEmbeddings(model=cfg.embed_model)
        self.llm = ChatOllama(
            model=cfg.llm_model, num_ctx=cfg.ctx_window, temperature=cfg.llm_temperature
        )
        self.rewriter_llm = ChatOllama(
            model=cfg.rewriter_model or cfg.llm_model,
            num_ctx=cfg.ctx_window,
            temperature=cfg.rewriter_temperature,
        )
        self.critic_llm = ChatOllama(
            model=cfg.critic_model or cfg.llm_model,
            num_ctx=cfg.ctx_window,
            temperature=cfg.critic_temperature,
        )
        self.vision_llm = (
            ChatOllama(
                model=cfg.vlm_model, num_ctx=cfg.ctx_window,
                temperature=cfg.vlm_temperature,
            )
            if cfg.multimodal_enabled and cfg.vlm_generation_enabled else None
        )
        self.vectorstore = Chroma(
            persist_directory=str(cfg.chroma_dir), embedding_function=self.embeddings
        )
        self.cache = CacheLayer(self.db, cfg)
        self.metrics = MetricsRecorder(self.db, cfg)
        self.rewriter = QueryRewriter(self.db, cfg, self.rewriter_llm)
        self.critic = CriticAndRepair(self.critic_llm, cfg=cfg)
        self.bm25: Optional[BM25Index] = None
        self.reranker: Optional[CrossEncoderReranker] = None
        self.retriever: Optional[HybridRetriever] = None
        self.web_store: Optional[WebChunkStore] = None
        self._rag_chain = _RAG_PROMPT | self.llm | StrOutputParser()
        self._practice_chain = _PRACTICE_PROMPT | self.llm | StrOutputParser()
        self._query_count = 0
        self._last_multimodal_usage = {
            "generation_mode": "not_run", "images_attached": 0, "tables_attached": 0,
        }
        self._last_adaptive_web_usage = {
            "enabled": bool(getattr(cfg, "adaptive_web_enabled", True)),
            "rounds": 0, "network_sources": 0, "cache_chunks": 0,
            "stop_reason": "not_run", "elapsed_ms": 0.0,
        }
        self._last_web_scrape_stats = {
            "query": "", "requested_urls": 0, "fetched_urls": [],
            "used_urls": [], "cache_hits": 0, "new_chunks": 0,
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
        trace = QueryTrace(query_text=original_question)
        self._last_adaptive_web_usage = {
            "enabled": bool(getattr(self.cfg, "adaptive_web_enabled", True)),
            "rounds": 0, "network_sources": 0, "cache_chunks": 0,
            "stop_reason": "not_run", "elapsed_ms": 0.0,
        }
        self._last_web_scrape_stats = {
            "query": "", "requested_urls": 0, "fetched_urls": [],
            "used_urls": [], "cache_hits": 0, "new_chunks": 0,
        }
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
        # All retrieval sufficiency checks must evaluate the evidence subject,
        # not the user's artifact instruction. Final generation still receives
        # the original question so the requested output format is preserved.
        answerability_query = evidence_query if generation_request else question
        trace.query_text = question
        trace.t_rewrite = time.time()
        log.info("\n[Query] '%s'", question[:80])
        query_emb = self._embed_query(question)
        query_type = self._classify_query(question)
        retrieval_plan = self._route_query(question, query_type, rewritten_queries)
        if generation_request:
            topic_query = self._study_topic_query(evidence_query)
            retrieval_plan["queries"] = list(dict.fromkeys([
                topic_query,
                f"{topic_query} concepts formulas examples",
                evidence_query,
            ]))
            retrieval_plan["diversify"] = True
            retrieval_plan["top_k"] = min(retrieval_plan["top_k"], 6)
            retrieval_plan["candidate_k"] = max(
                retrieval_plan["candidate_k"],
                min(self.cfg.top_k_rerank + 4, 10),
            )
        retrieval_queries = retrieval_plan["queries"]
        rerank_query = (
            self._study_topic_query(evidence_query)
            if generation_request else question
        )
        log.info("[Query] Route=%s | retrieval_queries=%d | top_k=%d | web=%s",
             query_type, len(retrieval_queries), retrieval_plan["top_k"],
             retrieval_plan["always_web"])
        rewritten = "\n".join(retrieval_queries)
        if generation_request:
            log.info("[Query] Generation request; evidence query: %s", evidence_query)
        trace.rewritten_query = rewritten
        answer_cache_key = (
            f"question={question}\nrewritten_query={rewritten}\n"
            f"answer_cache_schema_version={self.cfg.answer_cache_schema_version}\n"
            f"retrieval_cache_schema_version={self.cfg.retrieval_cache_schema_version}\n"
            f"critic_config_version={self.cfg.critic_config_version}\n"
            f"triad_config_version={self.cfg.triad_config_version}"
        )
        cached = self.cache.get_answer(
            question, query_emb, cache_key=answer_cache_key, include_metadata=True
        )
        cached_answer = None
        cached_sources = []
        cached_metadata = {}
        if cached:
            cached_answer, cached_sources, cached_metadata = cached
            trace.answer_cache_hit = True
            if (
                self.cfg.return_validated_answer_cache
                and cached_metadata.get("validated") is True
            ):
                now = time.time()
                trace.t_retrieval = trace.t_rerank = trace.t_generation = trace.t_end = now
                cached_triad = cached_metadata.get("rag_triad", {})
                trace.answer_relevance = cached_triad.get("answer_relevance")
                trace.context_relevance = cached_triad.get("context_relevance")
                trace.answer_faithfulness = cached_triad.get("faithfulness")
                trace.triad_pass = cached_triad.get("passed")
                trace.triage_action = ";".join(cached_triad.get("triage_actions", []))
                self.metrics.record(trace)
                if conversation_id:
                    self.memory.add_message(conversation_id, "assistant", cached_answer)
                multimodal_usage = self._usage_from_sources(
                    cached_sources, generation_mode="answer_cache"
                )
                log.info("[Cache] Returning validated answer cache HIT")
                return {
                    "answer": cached_answer,
                    "sources": cached_sources,
                    "critic": cached_metadata,
                    "query_id": trace.query_id,
                    "rewrite_id": rewrite_id,
                    "rewritten_query": rewritten,
                    "from_cache": True,
                    "drift_alert": None,
                    "conversation_id": conversation_id,
                    "multimodal_usage": multimodal_usage,
                    "metrics": {
                        "total_ms": trace.total_ms(),
                        "rewrite_ms": trace.latency_ms(trace.t_start, trace.t_rewrite),
                        "answer_cache_hit": True,
                        "adaptive_web": dict(self._last_adaptive_web_usage),
                        "answer_relevance": trace.answer_relevance,
                        "context_relevance": trace.context_relevance,
                        "faithfulness": trace.answer_faithfulness,
                        "triad_pass": trace.triad_pass,
                    },
                }
            log.info("[Cache] Using unvalidated/stale-policy cache entry only as supplemental context")
        retrieval_cache_key = (
            f"{rewritten}\n"
            f"retrieval_cache_schema_version={self.cfg.retrieval_cache_schema_version}"
        )
        cached_chunks = self.cache.get_retrieval(retrieval_cache_key)
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
            log.info("[Query] Starting local retrieval...")
            pdf_workers = max(1, min(len(retrieval_queries), 4))
            with ThreadPoolExecutor(max_workers=pdf_workers) as pool:
                pdf_futures = {
                    pool.submit(self.retriever.retrieve_candidates, query, metadata_filter): query
                    for query in retrieval_queries
                }
                pdf_results = [future.result() for future in pdf_futures]

                for index, query in enumerate(retrieval_queries):
                    pdf_count = len(pdf_results[index][0]) if index < len(pdf_results) else 0
                    retrieval_calls.append({
                        "query": query,
                        "pdf_candidates": pdf_count,
                        "web_candidates": 0,
                        "candidates": pdf_count,
                    })

            pdf_candidates = []
            bm25_ids = set()
            dense_ids = set()
            for candidates, query_bm25_ids, query_dense_ids in pdf_results:
                pdf_candidates.extend(candidates)
                bm25_ids.update(query_bm25_ids)
                dense_ids.update(query_dense_ids)
            web_chunks = []
            web_scrape_used = False
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
            combined_chunks = sorted(expanded_pdf + web_winners, key=stable_chunk_sort_key)
            chunks = self._diversify_sources(combined_chunks, query_type, retrieval_plan["top_k"])
            trace.t_rerank = time.time()

            pdf_evidence = [chunk for chunk in combined_chunks if chunk.get("source_type") != "web"]
            web_evidence = [chunk for chunk in combined_chunks if chunk.get("source_type") == "web"]
            pdf_answerable = self._combined_evidence_is_answerable(answerability_query, pdf_evidence)
            combined_answerable = self._combined_evidence_is_answerable(answerability_query, combined_chunks)
            need_adaptive_web = (
                use_web_fallback
                and (retrieval_plan["always_web"] or not combined_answerable)
            )
            if need_adaptive_web:
                log.info(
                    "[Query] %s; starting adaptive web retrieval",
                    "Route requires web evidence"
                    if retrieval_plan["always_web"]
                    else "Local sources failed answerability",
                )
                chunks = self._broader_web_retrieval(
                    answerability_query,
                    retrieval_queries,
                    chunks,
                    query_type,
                    retrieval_plan["top_k"],
                    rerank_query,
                )
                web_scrape_used = (
                    self._last_adaptive_web_usage.get("rounds", 0) > 0
                )
                trace.t_retrieval = time.time()
                trace.t_rerank = time.time()

            safe_chunks = [{k: v for k, v in c.items()
                            if isinstance(v, (str, int, float, bool, type(None), list))} for c in chunks]
            if self._combined_evidence_is_answerable(answerability_query, chunks):
                self.cache.set_retrieval(retrieval_cache_key, safe_chunks)
            else:
                log.info("[Cache] Skipping weak retrieval result")

        scores = [c.get("rerank_score", 0.0) for c in chunks if "rerank_score" in c]
        trace.num_chunks_retrieved = len(chunks)
        trace.mean_rerank_score = round(sum(scores) / len(scores), 4) if scores else 0.0
        trace.top_rerank_score = max(scores) if scores else 0.0
        trace.bm25_overlap = len(bm25_ids & dense_ids)

        pdf_candidates = [chunk for chunk in chunks if chunk.get("source_type") != "web"]
        web_candidates = [chunk for chunk in chunks if chunk.get("source_type") == "web"]
        pdf_answerable = self._combined_evidence_is_answerable(answerability_query, pdf_candidates)
        answerable = self._combined_evidence_is_answerable(answerability_query, chunks)
        web_evidence_relevant = self._web_evidence_is_relevant(answerability_query, web_candidates)
        confidence_chunks = [
            chunk for chunk in chunks
            if "rerank_score" in chunk and not chunk.get("context_only")
        ]
        rerank_confident, _, _ = self._rerank_confidence(confidence_chunks)
        low_confidence = not rerank_confident
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
                answerability_query,
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
            answerable = self._combined_evidence_is_answerable(answerability_query, chunks)
            web_evidence_relevant = self._web_evidence_is_relevant(answerability_query, web_candidates)
            trace.num_chunks_retrieved = len(chunks)
            trace.mean_rerank_score = round(sum(scores) / len(scores), 4) if scores else 0.0
            trace.top_rerank_score = max(scores) if scores else 0.0
            trace.t_rerank = time.time()
            web_scrape_used = (
                self._last_adaptive_web_usage.get("rounds", 0) > 0
            )

        log.info(self._retrieval_debug_summary(
            answerability_query,
            retrieval_queries,
            retrieval_calls,
            unique_candidate_count,
            chunks,
            answerable,
            web_fallback_requested,
        ))

        evidence_context = self._format_context(chunks)
        context = evidence_context
        if cached_answer:
            cached_context = self._format_cached_answer_context(
                cached_answer, cached_sources
            )
            context = f"{cached_context}\n\n[CURRENTLY RETRIEVED EVIDENCE]\n\n{evidence_context}"
        self._last_multimodal_usage = {
            "generation_mode": "not_run", "images_attached": 0, "tables_attached": 0,
        }
        if answerable:
            log.info("[Query] Generating answer over %d chunks...", len(chunks))
            raw_answer = self._generate_multimodal(
                question, context, chunks, query_type=query_type
            ).strip()
            answer = self._sanitize_answer(raw_answer)
        else:
            log.info(
                "[Answerability] FAILED: %s",
                self._answerability_debug(chunks, answerability_query),
            )
            log.warning("[Query] Answerability gate failed; returning insufficient-information response")
            answer = self._insufficient_information_response(question)
        generated_answer = answer
        trace.t_generation = time.time()
        log.info("[Query] Web scrape used: %s", web_scrape_used)

        critic_result = {
            "verdict": "PASS", "severity": "none", "critique": "",
        }
        critic_ran = False
        should_run_critic = self._should_run_critic(
            low_confidence=low_confidence,
            answerable=answerable,
            generation_request=generation_request,
        )
        repair_mode = (
            "critic_disabled"
            if not self.cfg.critic_enabled
            else "skipped_high_confidence"
        )
        repair_attempts = 0
        critic_started = time.perf_counter()
        if should_run_critic:
            critic_ran = True
            critic_result = self.critic.evaluate(question, answer, evidence_context)
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
                repaired, repair_mode = self._repair_answer(
                    question, answer, critic_result, evidence_context
                )
                repair_attempts += 1
                if repaired is None:
                    answer = self._insufficient_information_response(question)
                    repair_mode = "abstain"
                    break
                answer = self._sanitize_answer(repaired)
                if repair_mode in {"incomplete_preserved", "context_issue_preserved"}:
                    # Nothing was rewritten. Preserve the semantic check details
                    # for triad diagnostics while avoiding an identical LLM call.
                    critic_result = {**critic_result, "verdict": "PASS"}
                    break
                critic_result = (
                    self.critic.evaluate(question, answer, evidence_context)
                    if self.cfg.critic_require_context_grounding
                    else {"verdict": "PASS", "severity": "none", "critique": ""}
                )
            log.info("[Query] Critic verdict=%s severity=%s repair_mode=%s",
                     critic_result["verdict"], critic_result.get("severity"), repair_mode)

        critic_ms = (
            round((time.perf_counter() - critic_started) * 1000, 1)
            if critic_ran else 0.0
        )
        critic_details = (
            dict(self.critic.last_details) if critic_ran else {}
        )
        if self.cfg.critic_enabled and self.cfg.critic_polish_enabled:
            answer = self._sanitize_answer(self.critic.polish(answer))

        abstained = answer.strip() == CANONICAL_INSUFFICIENT_INFO_RESPONSE
        triad = assess_rag_triad(
            critic_details=critic_details,
            critic_verdict=critic_result["verdict"],
            answerable=answerable,
            abstained=abstained,
            cfg=self.cfg,
            critic_evaluated=critic_ran,
        )
        faith_score = triad.faithfulness
        trace.answer_relevance = triad.answer_relevance
        trace.context_relevance = triad.context_relevance
        trace.answer_faithfulness = faith_score
        trace.triad_pass = triad.passed
        trace.triage_action = ";".join(triad.triage_actions)
        if triad.passed is False:
            log.warning(
                "[RAG Triad] low=%s actions=%s",
                list(triad.low_metrics), list(triad.triage_actions),
            )

        validated = (
            critic_ran
            and critic_result["verdict"] == "PASS"
            and triad.passed is True
        )
        critic_metadata = {
            "initial_answer": generated_answer,
            "final_answer": answer,
            "verdict": critic_result["verdict"],
            "severity": critic_result.get("severity", "none"),
            "repair_mode": repair_mode,
            "repair_attempts": repair_attempts,
            "critic_ran": critic_ran,
            "validated": validated,
            "rag_triad": triad.as_dict(),
        }

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
        triad_values = [
            value for value in (
                triad.answer_relevance,
                triad.context_relevance,
                triad.faithfulness,
            )
            if value is not None
        ]
        triad_score = min(triad_values) if triad_values else None
        if triad_score is not None:
            self.rewriter.record_answer_score(rewrite_id, triad_score)
            if triad_score >= self.cfg.rewriter_helpful_min_score:
                self.rewriter.record_feedback(rewrite_id, helpful=True)
            elif triad_score < self.cfg.rewriter_unhelpful_max_score:
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
                "answer_relevance": triad.answer_relevance,
                "context_relevance": triad.context_relevance,
                "faithfulness": faith_score,
                "triad_pass": triad.passed,
                "triage_actions": list(triad.triage_actions),
                "critic_groundedness": critic_details.get("groundedness", "N/A"),
                "critic_answer_relevance": critic_details.get("answer_relevance", "N/A"),
                "critic_completeness": critic_details.get("completeness", "N/A"),
                "critic_relevance": critic_details.get("relevance", "N/A"),
                "chunks_used": len(chunks), "bm25_overlap": trace.bm25_overlap,
                "retrieval_cached": trace.retrieval_cache_hit,
                "query_type": query_type,
                "generation_request": generation_request,
                "evidence_query": evidence_query,
                "rerank_query": rerank_query,
                "answerable": answerable,
                "answerability_debug": self._answerability_debug(
                    chunks, answerability_query
                ),
                "low_confidence": low_confidence,
                "web_scrape_used": web_scrape_used,
                "adaptive_web": dict(self._last_adaptive_web_usage),
                "critic_ran": critic_ran,
                "critic_ms": critic_ms,
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
                             chunks: list[dict], query_type: str = "") -> str:
        image_chunks = self._select_image_chunks(chunks)
        if not image_chunks or self.vision_llm is None:
            self._last_multimodal_usage = {
                "generation_mode": "text_only", "images_attached": 0, "tables_attached": 0,
            }
            if query_type == "practice":
                return self._practice_chain.invoke({
                    "context": context,
                    "question": question,
                    "max_problems": self.cfg.practice_max_problems,
                })
            if query_type == "practice":
                return self._practice_chain.invoke({
                    "context": context,
                    "question": question,
                    "max_problems": self.cfg.practice_max_problems,
                })
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

    def _web_scrape_chunks(self, query: str, max_urls: Optional[int] = None,
                           exclude_urls: Optional[set[str]] = None) -> list[dict]:
        """Search/fetch a bounded number of sources and batch their embeddings."""
        assert self.web_store is not None, "Call await setup() before query()."
        requested_urls = max(
            1,
            min(
                int(max_urls or self.cfg.max_scrape_urls),
                int(self.cfg.max_scrape_urls),
            ),
        )
        excluded = {
            normalized
            for normalized in (
                _normalize_url(url) for url in (exclude_urls or set()) if url
            )
            if normalized
        }

        raw = []
        for attempt in range(self.cfg.ddg_retries):
            try:
                search_size = max(
                    self.cfg.max_scrape_urls * 2,
                    (requested_urls + len(excluded)) * 3,
                )
                with DDGS() as ddgs:
                    raw = list(ddgs.text(query, max_results=search_size))
                if raw:
                    break
                time.sleep(2 ** attempt)
            except Exception as exc:
                log.warning("[WebScrape] DDG attempt %d failed: %s", attempt + 1, exc)
                time.sleep(2 ** attempt)

        approved = _select_web_results(
            raw,
            max(self.cfg.max_scrape_urls, requested_urls + len(excluded)),
            self.cfg.min_domain_score,
        )

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
                    continue
                normalized_url = decision["normalized_url"]
            else:
                normalized_url = _normalize_url(original_url)

            if (
                not normalized_url
                or normalized_url in excluded
                or normalized_url in seen_urls
            ):
                continue
            seen_urls.add(normalized_url)
            result_copy = dict(result)
            result_copy["href"] = normalized_url
            candidates.append(result_copy)
            if len(candidates) >= requested_urls:
                break

        log.info(
            "[WebScrape] requested=%d approved=%d rejected=%d excluded=%d",
            requested_urls, len(candidates), rejected_urls, len(excluded),
        )

        cached_urls = []
        fetch_candidates = []
        for result in candidates:
            if self.web_store.is_fresh(result["href"]):
                cached_urls.append(result["href"])
            else:
                fetch_candidates.append(result)

        fetched_results = []
        fetch_workers = max(
            1, min(len(fetch_candidates) or 1, int(self.cfg.web_fetch_workers))
        )
        with ThreadPoolExecutor(max_workers=fetch_workers) as pool:
            futures = {
                pool.submit(
                    self._fetch_verified_url,
                    result["href"],
                    query,
                    result.get("domain_score", 0),
                ): result
                for result in fetch_candidates
            }
            for future, result in futures.items():
                try:
                    fetched = future.result()
                except Exception as exc:
                    log.warning("[WebScrape] Fetch failed for %s: %s", result["href"], exc)
                    continue
                if fetched:
                    fetched_results.append((result, fetched))

        pending_pages = []
        seen_content = set()
        fetched_urls = []
        for result, fetched in fetched_results:
            canonical_url, title, text, page_metadata = fetched
            content_hash = hashlib.sha256(
                re.sub(r"\s+", " ", text.lower()).encode()
            ).hexdigest()
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)
            fetched_urls.append(canonical_url)
            if self.web_store.is_fresh(canonical_url):
                continue
            pending_pages.append((
                canonical_url,
                title or result.get("title", "Web"),
                text,
                {
                    "domain_score": result.get("domain_score"),
                    "domain_score_reasons": result.get("domain_score_reasons", []),
                    **page_metadata,
                },
            ))

        # One batched embedding/upsert replaces one embedding call per source.
        new_chunks = self.web_store.upsert_batch(pending_pages) if pending_pages else 0
        self._last_web_scrape_stats = {
            "query": query,
            "requested_urls": requested_urls,
            "fetched_urls": fetched_urls,
            "used_urls": list(dict.fromkeys([*cached_urls, *fetched_urls])),
            "cache_hits": len(cached_urls),
            "new_chunks": new_chunks,
        }
        log.info(
            "[WebScrape] fetched=%d/%d new_chunks=%d",
            len(fetched_urls), len(candidates), new_chunks,
        )
        return self.web_store.search(
            query,
            k=min(self.cfg.web_top_k, max(2, requested_urls * 3)),
        )

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
                        timeout=self.cfg.web_request_timeout_seconds,
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