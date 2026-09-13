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

_COLD_PROMPT = ChatPromptTemplate.from_template("""
You are a search query optimizer for a RAG system.
Rewrite the user's query to maximise retrieval recall while preserving the
user's exact intent.

Rules:
- If the query is already clear, make only small retrieval-oriented expansions.
- Do not change the question being asked, its subject, or its requested scope.
- Expand acronyms and add closely related synonyms only when useful.
- Keep the rewritten query under 80 words.
- Preserve important terminology from the original query.
- Return ONLY the rewritten query — no preamble, no explanation.

Original query: {query}
Rewritten query:""")

_FEW_SHOT_PROMPT = ChatPromptTemplate.from_template("""
You are a search query optimizer for a RAG system.
Rewrite the user's query to maximise retrieval recall while preserving the
user's exact intent.

Here are examples of good rewrites that led to useful answers:
{examples}

Rules:
- Follow the style of the examples above without copying their subject matter.
- If the query is already clear, make only small retrieval-oriented expansions.
- Do not change the question being asked, its subject, or its requested scope.
- Expand acronyms and add closely related synonyms only when useful.
- Keep the rewritten query under 80 words.
- Preserve important terminology from the original query.
- Return ONLY the rewritten query — no preamble, no explanation.

Original query: {query}
Rewritten query:""")


class QueryRewriter:
    """Trainable query rewriter backed by SQLite rewrite history."""

    def __init__(self, db: Database, cfg: RAGConfig, llm: ChatOllama) -> None:
        self.db = db
        self.cfg = cfg
        self._cold_chain = _COLD_PROMPT | llm | StrOutputParser()
        self._few_chain = _FEW_SHOT_PROMPT | llm | StrOutputParser()

    @staticmethod
    def _looks_ambiguous(query: str) -> bool:
        words = re.findall(r"[A-Za-z0-9_]+", query)
        if len(words) < 3:
            return True
        vague = {"this", "that", "it", "they", "thing", "stuff", "help", "better", "works"}
        return any(word.lower() in vague for word in words) or len(query.strip()) < 18

    def rewrite(self, query: str) -> tuple[int, str]:
        """Rewrite only when enabled and the query is likely ambiguous."""
        original = query.strip()
        if not self.cfg.rewrite_enabled or (
            self.cfg.rewrite_only_when_ambiguous and not self._looks_ambiguous(original)
        ):
            checkpoint("rewrite.skipped", original, enabled=self.cfg.debug_checkpoints,
                       preview_chars=self.cfg.checkpoint_preview_chars,
                       sample_items=self.cfg.checkpoint_sample_items,
                       reason="disabled_or_clear_query")
            return 0, original

        checkpoint("rewrite.input", original, enabled=self.cfg.debug_checkpoints,
                   preview_chars=self.cfg.checkpoint_preview_chars,
                   sample_items=self.cfg.checkpoint_sample_items)
        examples = self._fetch_positive_examples()
        example_block = "\n".join(
            f"  Original:  {ex['original_query']}\n  Rewritten: {ex['rewritten_query']}"
            for ex in examples
        )
        if examples:
            rewritten = self._few_chain.invoke({"query": original, "examples": example_block}).strip()
        else:
            rewritten = self._cold_chain.invoke({"query": original}).strip()
        rewritten = self._sanitize(original, rewritten)
        rewrite_id = self._store(original, rewritten)
        checkpoint("rewrite.output", {"original": original, "rewritten": rewritten},
                   enabled=self.cfg.debug_checkpoints, preview_chars=self.cfg.checkpoint_preview_chars,
                   sample_items=self.cfg.checkpoint_sample_items, rewrite_id=rewrite_id,
                   changed=(original != rewritten))
        return rewrite_id, rewritten

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
