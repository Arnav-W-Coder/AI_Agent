"""
cache.py — Three-layer caching system.

Layer 1 — Answer cache (semantic):
  Stores (query_embedding, answer) pairs. On lookup, computes cosine similarity
  against all stored embeddings. If max sim > threshold → cache hit. This means
  "What is RAG?" and "Can you explain RAG?" resolve to the same cached answer.

Layer 2 — Retrieval cache (exact hash):
  Stores (query_hash → chunks) pairs. Exact match only — retrieval is cheap
  enough that near-miss queries should fetch fresh results.

Layer 3 — Index / metadata cache (in-memory):
  A hot dict of {doc_id → metadata} loaded at startup. Used for fast metadata
  filtering before any vector search. Invalidated when ingestion runs.
"""
import hashlib
import json
import time
import io
import logging
from typing import Any, Optional

import numpy as np

from db import Database
from config import RAGConfig

log = logging.getLogger(__name__)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def _emb_to_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr.astype(np.float32))
    return buf.getvalue()


def _bytes_to_emb(raw: bytes) -> np.ndarray:
    return np.load(io.BytesIO(raw))


def _query_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class CacheLayer:
    """Three-layer cache backed by SQLite + in-memory index."""

    def __init__(self, db: Database, cfg: RAGConfig) -> None:
        self.db  = db
        self.cfg = cfg
        self._index_cache: dict[str, dict] = {}   # Layer 3: in-memory metadata
        self._load_index_cache()

    # ── Layer 3: Index / metadata cache ──────────────────────────────────────

    def _load_index_cache(self) -> None:
        """Load all document metadata into memory at startup."""
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT id, filename, chunk_count, metadata_json FROM documents"
            ).fetchall()
        self._index_cache = {
            r["id"]: {
                "filename":    r["filename"],
                "chunk_count": r["chunk_count"],
                "metadata":    json.loads(r["metadata_json"] or "{}"),
            }
            for r in rows
        }
        log.info(f"[Cache] Index cache loaded: {len(self._index_cache)} documents")

    def invalidate_index_cache(self) -> None:
        """Call after ingestion completes."""
        self._load_index_cache()

    def get_document_metadata(self, doc_id: str) -> Optional[dict]:
        return self._index_cache.get(doc_id)

    def all_document_metadata(self) -> dict[str, dict]:
        return dict(self._index_cache)

    # ── Layer 2: Retrieval cache ──────────────────────────────────────────────

    def get_retrieval(self, query: str) -> Optional[list[dict]]:
        qhash = _query_hash(query)
        now   = time.time()
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT chunks_json, expires_at, id FROM retrieval_cache "
                "WHERE query_hash = ?",
                (qhash,)
            ).fetchone()
            if row is None or row["expires_at"] < now:
                return None
            conn.execute(
                "UPDATE retrieval_cache SET hit_count = hit_count + 1 WHERE id = ?",
                (row["id"],)
            )
        log.info("[Cache] Retrieval cache HIT")
        return json.loads(row["chunks_json"])

    def set_retrieval(self, query: str, chunks: list[dict]) -> None:
        qhash = _query_hash(query)
        now   = time.time()
        with self.db.connect() as conn:
            conn.execute(
                """INSERT INTO retrieval_cache
                   (query_hash, query_text, chunks_json, created_at, expires_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(query_hash) DO UPDATE SET
                       chunks_json = excluded.chunks_json,
                       expires_at  = excluded.expires_at,
                       hit_count   = 0""",
                (qhash, query, json.dumps(chunks), now,
                 now + self.cfg.retrieval_ttl)
            )

    # ── Layer 1: Answer cache (semantic similarity) ───────────────────────────

    def get_answer(
        self, query: str, query_embedding: np.ndarray
    ) -> Optional[tuple[str, list]]:
        """
        Semantic lookup: find the closest cached query by cosine similarity.
        Returns (answer, sources) if a hit is found, else None.
        """
        now = time.time()
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT id, query_embedding, answer, sources_json, expires_at "
                "FROM answer_cache WHERE expires_at > ?",
                (now,)
            ).fetchall()

        if not rows:
            return None

        best_sim, best_row = -1.0, None
        for row in rows:
            cached_emb = _bytes_to_emb(row["query_embedding"])
            sim = _cosine_sim(query_embedding, cached_emb)
            if sim > best_sim:
                best_sim, best_row = sim, row

        if best_sim >= self.cfg.answer_sim_threshold and best_row:
            log.info(f"[Cache] Answer cache HIT (sim={best_sim:.3f})")
            # Increment hit count
            with self.db.connect() as conn:
                conn.execute(
                    "UPDATE answer_cache SET hit_count = hit_count + 1 WHERE id = ?",
                    (best_row["id"],)
                )
            return best_row["answer"], json.loads(best_row["sources_json"] or "[]")

        log.info(f"[Cache] Answer cache MISS (best_sim={best_sim:.3f})")
        return None

    def set_answer(
        self,
        query: str,
        query_embedding: np.ndarray,
        answer: str,
        sources: list,
    ) -> None:
        qhash = _query_hash(query)
        now   = time.time()
        with self.db.connect() as conn:
            conn.execute(
                """INSERT INTO answer_cache
                   (query_hash, query_text, query_embedding, answer,
                    sources_json, created_at, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(query_hash) DO UPDATE SET
                       answer       = excluded.answer,
                       expires_at   = excluded.expires_at,
                       hit_count    = 0""",
                (qhash, query, _emb_to_bytes(query_embedding), answer,
                 json.dumps(sources), now, now + self.cfg.answer_ttl)
            )

    # ── Cache stats (for monitoring) ─────────────────────────────────────────

    def stats(self) -> dict:
        with self.db.connect() as conn:
            a = conn.execute(
                "SELECT COUNT(*) as n, SUM(hit_count) as hits FROM answer_cache "
                "WHERE expires_at > ?", (time.time(),)
            ).fetchone()
            r = conn.execute(
                "SELECT COUNT(*) as n, SUM(hit_count) as hits FROM retrieval_cache "
                "WHERE expires_at > ?", (time.time(),)
            ).fetchone()
        return {
            "answer_cache":    {"entries": a["n"], "total_hits": a["hits"] or 0},
            "retrieval_cache": {"entries": r["n"], "total_hits": r["hits"] or 0},
            "index_cache":     {"entries": len(self._index_cache)},
        }