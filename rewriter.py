"""
rewriter.py — Trainable query rewrite pipeline (Rewrite → Retrieve → Read).
"""
import logging
import re
import time
from typing import Optional

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from checkpoints import checkpoint
from config import RAGConfig
from db import Database

log = logging.getLogger(__name__)
MAX_FEW_SHOT = 4

_STANDALONE_PROMPT = ChatPromptTemplate.from_template("""
Rewrite the latest user message as one standalone retrieval question.
Use CHAT HISTORY only to resolve pronouns, ellipsis, and implied subjects.
Preserve every explicit entity, comparison option, constraint, and requested
decision criterion from the latest message. Do not add a recommendation,
assumptions, examples, or a new scope. If it is already standalone, return it
with only harmless wording cleanup.

CHAT HISTORY:
{history}

LATEST USER MESSAGE:
{query}

Return ONLY the standalone question.""")

_EXPANSION_PROMPT = ChatPromptTemplate.from_template("""
You are generating parallel retrieval queries for a research system.
Return one retrieval query per line and nothing else.

Include:
- the standalone question's intent, entities, and constraints;
- distinct formulations that improve recall, not cosmetic rewrites;
- comparison dimensions and one focused query per option for comparisons.

Rules:
- Keep the original meaning and scope. Do not answer the question.
- Produce 3 to 5 distinct queries.
- Keep each query under 80 words.
- Do not use bullets, numbering, labels, or explanations.

Standalone question: {query}
Retrieval queries:""")


class QueryRewriter:
    """Trainable query rewriter backed by SQLite rewrite history."""

    def __init__(self, db: Database, cfg: RAGConfig, llm: ChatOllama) -> None:
        self.db = db
        self.cfg = cfg
        self._standalone_chain = _STANDALONE_PROMPT | llm | StrOutputParser()
        self._expansion_chain = _EXPANSION_PROMPT | llm | StrOutputParser()

    @staticmethod
    def _looks_ambiguous(query: str) -> bool:
        words = re.findall(r"[A-Za-z0-9_]+", query)
        if len(words) < 3:
            return True
        vague = {"this", "that", "it", "they", "thing", "stuff", "help", "better", "works"}
        return any(word.lower() in vague for word in words) or len(query.strip()) < 18

    @staticmethod
    def _needs_expansion(query: str) -> bool:
        words = re.findall(r"[A-Za-z0-9_]+", query.lower())
        intent_terms = {
            "best", "compare", "comparison", "versus", "vs", "recommend",
            "recommendation", "choose", "alternatives", "differences",
        }
        constraint_terms = {
            "with", "without", "for", "using", "including", "citations",
            "production", "local", "tool", "tools", "dimensions",
        }
        return (
            any(word in intent_terms for word in words)
            or sum(word in constraint_terms for word in words) >= 2
            or query.count(",") >= 2
            or len(words) >= 18
        )

    @staticmethod
    def _is_explicit_comparison(query: str) -> bool:
        text = query.lower().replace("’", "'")
        return any(marker in text for marker in (
            "what's better", "what is better", "which is better", " vs ",
            " versus ", "compare ", "comparison", "differences between",
        ))

    @staticmethod
    def _parse_queries(original: str, output: str, maximum: int) -> list[str]:
        queries = [original]
        for line in (output or "").splitlines():
            candidate = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            if not candidate:
                continue
            candidate = QueryRewriter._sanitize(original, candidate)
            if candidate and candidate.lower() not in {q.lower() for q in queries}:
                queries.append(candidate)
            if len(queries) >= maximum:
                break
        return queries

    def rewrite(self, query: str) -> tuple[int, str]:
        """Return the legacy single-query rewrite API."""
        rewrite_id, queries = self.rewrite_queries(query)
        return rewrite_id, queries[-1] if len(queries) == 1 else " ".join(queries)

    def rewrite_queries(self, query: str, chat_history: Optional[list[dict | str]] = None) -> tuple[int, list[str]]:
        """Resolve the question, then generate bounded parallel retrieval queries."""
        original = query.strip()
        if not self.cfg.rewrite_enabled or (
            self.cfg.rewrite_only_when_ambiguous
            and not self._looks_ambiguous(original)
            and not self._needs_expansion(original)
        ):
            checkpoint("rewrite.skipped", original, enabled=self.cfg.debug_checkpoints,
                       preview_chars=self.cfg.checkpoint_preview_chars,
                       sample_items=self.cfg.checkpoint_sample_items,
                       reason="disabled_or_clear_query")
            return 0, [original]

        checkpoint("rewrite.input", original, enabled=self.cfg.debug_checkpoints,
                   preview_chars=self.cfg.checkpoint_preview_chars,
                   sample_items=self.cfg.checkpoint_sample_items)
        history = "\n".join(
            item if isinstance(item, str) else f"{item.get('role', 'user')}: {item.get('content', '')}"
            for item in (chat_history or [])
        ) or "(No prior conversation.)"
        standalone = self._sanitize(original, self._standalone_chain.invoke({
            "query": original, "history": history,
        }).strip())
        generated = self._expansion_chain.invoke({"query": standalone}).strip()
        queries = self._parse_queries(standalone, generated,
                                     maximum=max(2, int(self.cfg.multi_query_max_queries)))
        rewritten = "\n".join(queries)
        rewrite_id = self._store(original, rewritten)
        checkpoint("rewrite.output", {"original": original, "rewritten": rewritten, "queries": queries},
                   enabled=self.cfg.debug_checkpoints, preview_chars=self.cfg.checkpoint_preview_chars,
                   sample_items=self.cfg.checkpoint_sample_items, rewrite_id=rewrite_id,
                   changed=(original != rewritten), query_count=len(queries))
        return rewrite_id, queries

    @staticmethod
    def _sanitize(original: str, rewritten: str) -> str:
        candidate = (rewritten or "").strip()
        if len(candidate) < 5:
            return original
        candidate = re.sub(r"^(?:rewritten query|query)\s*:\s*", "", candidate, flags=re.I).strip()
        candidate = candidate.strip("`\"'")
        pairs = {"(": ")", "[": "]", "{": "}"}
        stack = []
        for char in candidate:
            if char in pairs:
                stack.append(pairs[char])
            elif char in pairs.values() and stack and char == stack[-1]:
                stack.pop()
        if stack:
            candidate += "".join(reversed(stack))
        return original if candidate.endswith(("(", "[", "{", ":", "-")) else candidate

    def record_feedback(self, rewrite_id: int, helpful: bool) -> None:
        with self.db.connect() as conn:
            conn.execute("UPDATE query_rewrites SET was_helpful = ? WHERE id = ?", (1 if helpful else 0, rewrite_id))

    def record_answer_score(self, rewrite_id: int, score: float) -> None:
        with self.db.connect() as conn:
            conn.execute("UPDATE query_rewrites SET answer_score = ? WHERE id = ?", (round(score, 4), rewrite_id))

    def few_shot_pool_size(self) -> int:
        with self.db.connect() as conn:
            return conn.execute("SELECT COUNT(*) as n FROM query_rewrites WHERE was_helpful = 1").fetchone()["n"]

    def rewrite_stats(self) -> dict:
        with self.db.connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS total, SUM(CASE WHEN was_helpful=1 THEN 1 ELSE 0 END) AS positive, SUM(CASE WHEN was_helpful=0 THEN 1 ELSE 0 END) AS negative, AVG(answer_score) AS mean_score FROM query_rewrites").fetchone()
        return {"total_rewrites": row["total"], "positive": row["positive"], "negative": row["negative"], "few_shot_pool": row["positive"], "mean_answer_score": round(row["mean_score"], 3) if row["mean_score"] else None}

    def _fetch_positive_examples(self) -> list[dict]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT original_query, rewritten_query, answer_score FROM query_rewrites WHERE was_helpful = 1 ORDER BY COALESCE(answer_score, 0) DESC, created_at DESC LIMIT ?", (MAX_FEW_SHOT,)).fetchall()
        return [dict(r) for r in rows]

    def _store(self, original: str, rewritten: str) -> int:
        with self.db.connect() as conn:
            cursor = conn.execute("INSERT INTO query_rewrites (original_query, rewritten_query, created_at) VALUES (?, ?, ?)", (original, rewritten, time.time()))
            return cursor.lastrowid
