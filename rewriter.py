"""
rewriter.py — Trainable query rewrite pipeline (Rewrite → Retrieve → Read).

How it works:
  1. rewrite(query) — LLM expands the raw query into a richer retrieval query.
     Before calling the LLM it pulls up to MAX_FEW_SHOT positive examples from
     the rewrite history table (rows where was_helpful = 1). These examples
     are injected as few-shot demonstrations so the rewriter improves over time
     as feedback accumulates.

  2. record_feedback(rewrite_id, helpful) — called after the pipeline delivers
     an answer. Marks a rewrite as helpful (1) or not (0). Helpful rewrites
     become future few-shot examples; unhelpful ones are excluded.

  3. record_answer_score(rewrite_id, score) — stores the critic faithfulness
     score alongside the rewrite row, giving a continuous quality signal in
     addition to binary feedback.

"Trainable" in this context means the rewriter's behaviour shifts based on
accumulated feedback — no gradient updates, but the few-shot pool grows
and improves, which is the practical lightweight approach for local LLMs.
"""
import logging
import time
from typing import Optional

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import RAGConfig
from db import Database

log = logging.getLogger(__name__)

MAX_FEW_SHOT = 4   # max positive examples injected into the rewrite prompt


# ── Prompt templates ──────────────────────────────────────────────────────────

# Used when no few-shot examples are available yet (cold start)
_COLD_PROMPT = ChatPromptTemplate.from_template("""
You are a search query optimizer for a RAG system.
Rewrite the user's query to maximise retrieval recall.

Rules:
- Expand acronyms and add synonyms.
- Make implicit concepts explicit.
- Keep the rewritten query under 80 words.
- Return ONLY the rewritten query — no preamble, no explanation.

Original query: {query}
Rewritten query:""")

# Used once positive examples exist in the rewrite history
_FEW_SHOT_PROMPT = ChatPromptTemplate.from_template("""
You are a search query optimizer for a RAG system.
Rewrite the user's query to maximise retrieval recall.

Here are examples of good rewrites that led to useful answers:
{examples}

Rules:
- Follow the style of the examples above.
- Expand acronyms and add synonyms.
- Make implicit concepts explicit.
- Keep the rewritten query under 80 words.
- Return ONLY the rewritten query — no preamble, no explanation.

Original query: {query}
Rewritten query:""")


class QueryRewriter:
    """
    Trainable query rewriter backed by a SQLite rewrite history.

    Usage:
        rewriter = QueryRewriter(db, cfg, llm)

        rewrite_id, rewritten = rewriter.rewrite("what did perry do in japan")
        # → (42, "What were the political and economic consequences of
        #          Commodore Matthew Perry's 1853 expedition to Edo Bay, Japan,
        #          and how did it accelerate the Meiji-era westernization?")

        # After the answer is delivered and user rates it:
        rewriter.record_feedback(rewrite_id, helpful=True)
        rewriter.record_answer_score(rewrite_id, score=0.91)
    """

    def __init__(self, db: Database, cfg: RAGConfig, llm: ChatOllama) -> None:
        self.db  = db
        self.cfg = cfg
        self._cold_chain = _COLD_PROMPT    | llm | StrOutputParser()
        self._few_chain  = _FEW_SHOT_PROMPT | llm | StrOutputParser()

    # ── Public API ────────────────────────────────────────────────────────────

    def rewrite(self, query: str) -> tuple[int, str]:
        """
        Rewrite `query` for better retrieval.

        Returns:
            (rewrite_id, rewritten_query)
            rewrite_id is the SQLite row id — pass it back to record_feedback().
        """
        examples = self._fetch_positive_examples()

        if examples:
            example_block = "\n".join(
                f"  Original:  {ex['original_query']}\n"
                f"  Rewritten: {ex['rewritten_query']}"
                for ex in examples
            )
            rewritten = self._few_chain.invoke({
                "query":    query,
                "examples": example_block,
            }).strip()
            log.info(f"[Rewriter] Few-shot ({len(examples)} examples) rewrite done")
        else:
            rewritten = self._cold_chain.invoke({"query": query}).strip()
            log.info("[Rewriter] Cold-start rewrite done")

        # Sanitise: if the model returns something unusable, fall back to original
        if not rewritten or len(rewritten) < 5:
            log.warning("[Rewriter] LLM returned empty rewrite — using original")
            rewritten = query

        rewrite_id = self._store(query, rewritten)
        log.info(f"[Rewriter] '{query[:50]}' →\n           '{rewritten[:80]}'")
        return rewrite_id, rewritten

    def record_feedback(self, rewrite_id: int, helpful: bool) -> None:
        """
        Mark a rewrite as helpful (True) or not (False).
        Helpful rewrites join the few-shot pool for future queries.
        """
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE query_rewrites SET was_helpful = ? WHERE id = ?",
                (1 if helpful else 0, rewrite_id)
            )
        log.info(f"[Rewriter] Feedback recorded: id={rewrite_id} helpful={helpful}")

    def record_answer_score(self, rewrite_id: int, score: float) -> None:
        """Store the faithfulness score (0–1) alongside the rewrite row."""
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE query_rewrites SET answer_score = ? WHERE id = ?",
                (round(score, 4), rewrite_id)
            )

    def few_shot_pool_size(self) -> int:
        """How many positive examples are currently in the pool."""
        with self.db.connect() as conn:
            return conn.execute(
                "SELECT COUNT(*) as n FROM query_rewrites WHERE was_helpful = 1"
            ).fetchone()["n"]

    def rewrite_stats(self) -> dict:
        """Summary of rewrite history for the monitoring report."""
        with self.db.connect() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*)                          AS total,
                    SUM(CASE WHEN was_helpful=1 THEN 1 ELSE 0 END) AS positive,
                    SUM(CASE WHEN was_helpful=0 THEN 1 ELSE 0 END) AS negative,
                    AVG(answer_score)                 AS mean_score
                FROM query_rewrites
            """).fetchone()
        return {
            "total_rewrites":   row["total"],
            "positive":         row["positive"],
            "negative":         row["negative"],
            "few_shot_pool":    row["positive"],
            "mean_answer_score": round(row["mean_score"], 3) if row["mean_score"] else None,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _fetch_positive_examples(self) -> list[dict]:
        """
        Retrieve the top MAX_FEW_SHOT positive rewrites, ranked by answer_score
        descending (best-performing first). Falls back to recency if no scores set.
        """
        with self.db.connect() as conn:
            rows = conn.execute(
                """SELECT original_query, rewritten_query, answer_score
                   FROM query_rewrites
                   WHERE was_helpful = 1
                   ORDER BY COALESCE(answer_score, 0) DESC, created_at DESC
                   LIMIT ?""",
                (MAX_FEW_SHOT,)
            ).fetchall()
        return [dict(r) for r in rows]

    def _store(self, original: str, rewritten: str) -> int:
        """Persist a new rewrite row; returns its auto-increment id."""
        with self.db.connect() as conn:
            cursor = conn.execute(
                """INSERT INTO query_rewrites
                   (original_query, rewritten_query, created_at)
                   VALUES (?, ?, ?)""",
                (original, rewritten, time.time())
            )
            return cursor.lastrowid