"""cache.py — Three-layer caching system with observable data-flow checkpoints."""
import hashlib
import json
import time
import io
import logging
from typing import Any, Optional

import numpy as np

from checkpoints import checkpoint
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
        self.db = db
        self.cfg = cfg
        self._index_cache: dict[str, dict] = {}
        self._load_index_cache()

    def _load_index_cache(self) -> None:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT id, filename, chunk_count, metadata_json FROM documents").fetchall()
        self._index_cache = {
            r["id"]: {"filename": r["filename"], "chunk_count": r["chunk_count"], "metadata": json.loads(r["metadata_json"] or "{}")}
            for r in rows
        }
        log.info(f"[Cache] Index cache loaded: {len(self._index_cache)} documents")
        checkpoint("cache.index_loaded", self._index_cache, enabled=self.cfg.debug_checkpoints,
                   preview_chars=self.cfg.checkpoint_preview_chars, sample_items=self.cfg.checkpoint_sample_items,
                   documents=len(self._index_cache))

    def invalidate_index_cache(self) -> None:
        self._load_index_cache()

    def get_document_metadata(self, doc_id: str) -> Optional[dict]:
        return self._index_cache.get(doc_id)

    def all_document_metadata(self) -> dict[str, dict]:
        return dict(self._index_cache)

    def get_retrieval(self, query: str) -> Optional[list[dict]]:
        qhash = _query_hash(query)
        now = time.time()
        with self.db.connect() as conn:
            row = conn.execute("SELECT chunks_json, expires_at, id FROM retrieval_cache WHERE query_hash = ?", (qhash,)).fetchone()
            if row is None or row["expires_at"] < now:
                checkpoint("cache.retrieval_miss", {"query": query}, enabled=self.cfg.debug_checkpoints,
                           preview_chars=self.cfg.checkpoint_preview_chars, sample_items=self.cfg.checkpoint_sample_items,
                           reason="missing_or_expired")
                return None
            conn.execute("UPDATE retrieval_cache SET hit_count = hit_count + 1 WHERE id = ?", (row["id"],))
        chunks = json.loads(row["chunks_json"])
        log.info("[Cache] Retrieval cache HIT")
        checkpoint("cache.retrieval_hit", chunks, enabled=self.cfg.debug_checkpoints,
                   preview_chars=self.cfg.checkpoint_preview_chars, sample_items=self.cfg.checkpoint_sample_items,
                   query=query, returned=len(chunks))
        return chunks

    def set_retrieval(self, query: str, chunks: list[dict]) -> None:
        qhash = _query_hash(query)
        now = time.time()
        with self.db.connect() as conn:
            conn.execute("""INSERT INTO retrieval_cache
                   (query_hash, query_text, chunks_json, created_at, expires_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(query_hash) DO UPDATE SET
                       chunks_json = excluded.chunks_json,
                       expires_at  = excluded.expires_at,
                       hit_count   = 0""", (qhash, query, json.dumps(chunks), now, now + self.cfg.retrieval_ttl))
        checkpoint("cache.retrieval_store", chunks, enabled=self.cfg.debug_checkpoints,
                   preview_chars=self.cfg.checkpoint_preview_chars, sample_items=self.cfg.checkpoint_sample_items,
                   query=query, stored=len(chunks))

    def get_answer(self, query: str, query_embedding: np.ndarray) -> Optional[tuple[str, list]]:
        now = time.time()
        checkpoint("cache.answer_lookup_input", {"query": query, "embedding_dimension": len(query_embedding)},
                   enabled=self.cfg.debug_checkpoints, preview_chars=self.cfg.checkpoint_preview_chars,
                   sample_items=self.cfg.checkpoint_sample_items)
        with self.db.connect() as conn:
            rows = conn.execute("SELECT id, query_embedding, answer, sources_json, expires_at FROM answer_cache WHERE expires_at > ?", (now,)).fetchall()
        if not rows:
            checkpoint("cache.answer_miss", {"query": query}, enabled=self.cfg.debug_checkpoints,
                       preview_chars=self.cfg.checkpoint_preview_chars, sample_items=self.cfg.checkpoint_sample_items,
                       reason="empty_cache")
            return None

        best_sim, best_row = -1.0, None
        for row in rows:
            cached_emb = _bytes_to_emb(row["query_embedding"])
            sim = _cosine_sim(query_embedding, cached_emb)
            if sim > best_sim:
                best_sim, best_row = sim, row

        if best_sim >= self.cfg.answer_sim_threshold and best_row:
            log.info(f"[Cache] Answer cache HIT (sim={best_sim:.3f})")
            with self.db.connect() as conn:
                conn.execute("UPDATE answer_cache SET hit_count = hit_count + 1 WHERE id = ?", (best_row["id"],))
            answer, sources = best_row["answer"], json.loads(best_row["sources_json"] or "[]")
            checkpoint("cache.answer_hit", {"answer": answer, "sources": sources},
                       enabled=self.cfg.debug_checkpoints, preview_chars=self.cfg.checkpoint_preview_chars,
                       sample_items=self.cfg.checkpoint_sample_items, similarity=round(best_sim, 4))
            return answer, sources

        log.info(f"[Cache] Answer cache MISS (best_sim={best_sim:.3f})")
        checkpoint("cache.answer_miss", {"query": query}, enabled=self.cfg.debug_checkpoints,
                   preview_chars=self.cfg.checkpoint_preview_chars, sample_items=self.cfg.checkpoint_sample_items,
                   best_similarity=round(best_sim, 4), threshold=self.cfg.answer_sim_threshold)
        return None

    def set_answer(self, query: str, query_embedding: np.ndarray, answer: str, sources: list) -> None:
        qhash = _query_hash(query)
        now = time.time()
        with self.db.connect() as conn:
            conn.execute("""INSERT INTO answer_cache
                   (query_hash, query_text, query_embedding, answer,
                    sources_json, created_at, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(query_hash) DO UPDATE SET
                       answer = excluded.answer, expires_at = excluded.expires_at, hit_count = 0""",
                         (qhash, query, _emb_to_bytes(query_embedding), answer, json.dumps(sources), now, now + self.cfg.answer_ttl))
        checkpoint("cache.answer_store", {"answer": answer, "sources": sources},
                   enabled=self.cfg.debug_checkpoints, preview_chars=self.cfg.checkpoint_preview_chars,
                   sample_items=self.cfg.checkpoint_sample_items, query=query)

    def stats(self) -> dict:
        with self.db.connect() as conn:
            a = conn.execute("SELECT COUNT(*) as n, SUM(hit_count) as hits FROM answer_cache WHERE expires_at > ?", (time.time(),)).fetchone()
            r = conn.execute("SELECT COUNT(*) as n, SUM(hit_count) as hits FROM retrieval_cache WHERE expires_at > ?", (time.time(),)).fetchone()
        return {
            "answer_cache": {"entries": a["n"], "total_hits": a["hits"] or 0},
            "retrieval_cache": {"entries": r["n"], "total_hits": r["hits"] or 0},
            "index_cache": {"entries": len(self._index_cache)},
        }
