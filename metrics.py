"""
metrics.py — Monitoring, quality measurement, and drift detection.

Tracks three measurement dimensions:
  1. Retrieval component  — rerank scores, BM25/dense overlap, chunks retrieved
  2. Answer quality       — faithfulness (critic score), user rating (1-5)
  3. End-to-end perf      — total/component latencies, cache hit rates

Drift detection:
  Every `drift_window` queries, compare the mean rerank score of the recent
  window against the historical baseline. A drop > drift_threshold triggers
  a re-embed recommendation logged to drift_log.
"""
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional

from db import Database
from config import RAGConfig

log = logging.getLogger(__name__)


@dataclass
class QueryTrace:
    """Populated incrementally as a query moves through the pipeline."""
    query_id:   str = field(default_factory=lambda: str(uuid.uuid4()))
    query_text: str = ""
    rewritten_query: Optional[str] = None

    # Latencies (ms)
    t_start:      float = field(default_factory=time.time)
    t_rewrite:    Optional[float] = None
    t_retrieval:  Optional[float] = None
    t_rerank:     Optional[float] = None
    t_generation: Optional[float] = None
    t_end:        Optional[float] = None

    # Cache
    answer_cache_hit:    bool = False
    retrieval_cache_hit: bool = False

    # Retrieval quality
    num_chunks_retrieved: int   = 0
    mean_rerank_score:    float = 0.0
    top_rerank_score:     float = 0.0
    bm25_overlap:         int   = 0   # chunks appearing in both BM25 + dense

    # Answer quality
    answer_faithfulness: Optional[float] = None
    user_rating:         Optional[int]   = None

    def latency_ms(self, t_from: Optional[float], t_to: Optional[float]) -> Optional[float]:
        if t_from is None or t_to is None:
            return None
        return round((t_to - t_from) * 1000, 1)

    def total_ms(self) -> Optional[float]:
        return self.latency_ms(self.t_start, self.t_end)


class MetricsRecorder:
    """Writes QueryTrace objects to SQLite and provides aggregate queries."""

    def __init__(self, db: Database, cfg: RAGConfig) -> None:
        self.db  = db
        self.cfg = cfg

    def record(self, trace: QueryTrace) -> None:
        """Persist a completed trace. Call at the end of every query."""
        now = time.time()
        with self.db.connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO query_metrics (
                    query_id, query_text, rewritten_query,
                    total_latency_ms, rewrite_latency_ms,
                    retrieval_latency_ms, rerank_latency_ms,
                    generation_latency_ms,
                    answer_cache_hit, retrieval_cache_hit,
                    num_chunks_retrieved, mean_rerank_score,
                    top_rerank_score, bm25_overlap,
                    user_rating, answer_faithfulness, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    trace.query_id, trace.query_text, trace.rewritten_query,
                    trace.total_ms(),
                    trace.latency_ms(trace.t_start,     trace.t_rewrite),
                    trace.latency_ms(trace.t_rewrite,   trace.t_retrieval),
                    trace.latency_ms(trace.t_retrieval, trace.t_rerank),
                    trace.latency_ms(trace.t_rerank,    trace.t_generation),
                    int(trace.answer_cache_hit),
                    int(trace.retrieval_cache_hit),
                    trace.num_chunks_retrieved,
                    trace.mean_rerank_score,
                    trace.top_rerank_score,
                    trace.bm25_overlap,
                    trace.user_rating,
                    trace.answer_faithfulness,
                    now,
                )
            )

        total = trace.total_ms()
        if total and total > self.cfg.latency_warn_ms:
            log.warning(f"[Metrics] Slow query ({total:.0f}ms): {trace.query_text[:60]}")
        if trace.mean_rerank_score < self.cfg.min_retrieval_score:
            log.warning(
                f"[Metrics] Poor retrieval score ({trace.mean_rerank_score:.3f}): "
                f"{trace.query_text[:60]}"
            )

    def record_user_rating(self, query_id: str, rating: int) -> None:
        """Call this when a user provides a 1-5 rating after the fact."""
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE query_metrics SET user_rating = ? WHERE query_id = ?",
                (rating, query_id)
            )

    # ── Drift detection ───────────────────────────────────────────────────────

    def check_drift(self) -> Optional[str]:
        """
        Compare the most recent `drift_window` queries against the full
        historical baseline. If mean rerank score drops by more than
        drift_threshold, log a re-embed recommendation.
        Returns an action string if drift detected, else None.
        """
        w = self.cfg.drift_window
        with self.db.connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) as n FROM query_metrics "
                "WHERE mean_rerank_score IS NOT NULL"
            ).fetchone()["n"]

            if total < w * 2:
                return None   # not enough data yet

            recent = conn.execute(
                "SELECT AVG(mean_rerank_score) as avg FROM ("
                "  SELECT mean_rerank_score FROM query_metrics "
                "  WHERE mean_rerank_score IS NOT NULL "
                "  ORDER BY created_at DESC LIMIT ?"
                ")", (w,)
            ).fetchone()["avg"]

            baseline = conn.execute(
                "SELECT AVG(mean_rerank_score) as avg FROM query_metrics "
                "WHERE mean_rerank_score IS NOT NULL"
            ).fetchone()["avg"]

        if baseline is None or recent is None:
            return None

        delta = (baseline - recent) / (abs(baseline) + 1e-9)
        action = None

        if delta >= self.cfg.drift_threshold:
            action = (
                f"DRIFT DETECTED: mean score dropped {delta*100:.1f}% "
                f"(recent={recent:.3f}, baseline={baseline:.3f}). "
                "Recommend re-embedding the document corpus."
            )
            log.warning(f"[Drift] {action}")
            with self.db.connect() as conn:
                conn.execute(
                    "INSERT INTO drift_log (window_queries, mean_score_recent, "
                    "mean_score_baseline, drift_delta, action_taken, logged_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (w, recent, baseline, delta, action, time.time())
                )

        return action

    # ── Monitoring report ─────────────────────────────────────────────────────

    def report(self, last_n: int = 50) -> dict:
        """Aggregate metrics over the last N queries. Used for dashboards."""
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM query_metrics ORDER BY created_at DESC LIMIT ?",
                (last_n,)
            ).fetchall()

        if not rows:
            return {"error": "no data"}

        vals = lambda key: [r[key] for r in rows if r[key] is not None]

        def avg(lst):   return round(sum(lst) / len(lst), 3) if lst else None
        def pct(lst):   return round(sum(lst) / len(lst) * 100, 1) if lst else None

        total_lats   = vals("total_latency_ms")
        retrieval_lats = vals("retrieval_latency_ms")
        rerank_lats  = vals("rerank_latency_ms")
        gen_lats     = vals("generation_latency_ms")
        rerank_scores = vals("mean_rerank_score")
        faithful     = vals("answer_faithfulness")
        ratings      = vals("user_rating")
        a_hits       = [r["answer_cache_hit"]    for r in rows]
        r_hits       = [r["retrieval_cache_hit"] for r in rows]

        return {
            "window":           last_n,
            "queries_recorded": len(rows),
            "latency_ms": {
                "total_avg":      avg(total_lats),
                "retrieval_avg":  avg(retrieval_lats),
                "rerank_avg":     avg(rerank_lats),
                "generation_avg": avg(gen_lats),
                "total_max":      max(total_lats)      if total_lats else None,
            },
            "cache_hit_rates": {
                "answer_pct":    pct(a_hits),
                "retrieval_pct": pct(r_hits),
            },
            "retrieval_quality": {
                "mean_rerank_score": avg(rerank_scores),
                "min_rerank_score":  min(rerank_scores) if rerank_scores else None,
            },
            "answer_quality": {
                "mean_faithfulness": avg(faithful),
                "mean_user_rating":  avg(ratings),
                "rated_queries":     len(ratings),
            },
        }