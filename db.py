"""
db.py — SQLite schema and connection management.
Stores everything except raw vectors (those live in ChromaDB):
  - document + chunk metadata
  - query rewrite history (trainable rewriter)
  - answer cache + retrieval cache
  - per-query metrics
  - drift log
"""
import sqlite3
import json
import time
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

SCHEMA = """
-- ── Documents ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    filepath      TEXT NOT NULL,
    file_hash     TEXT NOT NULL,       -- MD5; used to skip unchanged files
    page_count    INTEGER DEFAULT 0,
    chunk_count   INTEGER DEFAULT 0,
    ingested_at   REAL    NOT NULL,
    metadata_json TEXT    DEFAULT '{}'
);

-- ── Chunks (metadata alongside vectors) ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS chunks (
    id           TEXT PRIMARY KEY,
    doc_id       TEXT    NOT NULL REFERENCES documents(id),
    chroma_id    TEXT    NOT NULL,
    chunk_index  INTEGER NOT NULL,
    page_number  INTEGER DEFAULT 0,
    text_preview TEXT,                 -- first 200 chars for quick inspection
    UNIQUE(doc_id, chunk_index)
);

-- ── Query rewrite history (trainable rewriter) ───────────────────────────────
CREATE TABLE IF NOT EXISTS query_rewrites (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    original_query  TEXT NOT NULL,
    rewritten_query TEXT NOT NULL,
    was_helpful     INTEGER DEFAULT NULL,  -- 1 = yes, 0 = no, NULL = no feedback
    answer_score    REAL    DEFAULT NULL,  -- faithfulness score 0-1
    created_at      REAL    NOT NULL
);

-- ── Answer cache (semantic: stores embedding for similarity lookup) ───────────
CREATE TABLE IF NOT EXISTS answer_cache (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    query_hash      TEXT NOT NULL UNIQUE,
    query_text      TEXT NOT NULL,
    query_embedding BLOB NOT NULL,   -- numpy float32 array, raw bytes
    answer          TEXT NOT NULL,
    sources_json    TEXT DEFAULT '[]',
    created_at      REAL NOT NULL,
    expires_at      REAL NOT NULL,
    hit_count       INTEGER DEFAULT 0
);

-- ── Retrieval cache ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS retrieval_cache (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    query_hash   TEXT NOT NULL UNIQUE,
    query_text   TEXT NOT NULL,
    chunks_json  TEXT NOT NULL,      -- serialised list of chunk dicts
    created_at   REAL NOT NULL,
    expires_at   REAL NOT NULL,
    hit_count    INTEGER DEFAULT 0
);

-- ── Per-query metrics ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS query_metrics (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id              TEXT NOT NULL UNIQUE,
    query_text            TEXT NOT NULL,
    rewritten_query       TEXT,
    -- Latency breakdown (milliseconds)
    total_latency_ms      REAL,
    rewrite_latency_ms    REAL,
    retrieval_latency_ms  REAL,
    rerank_latency_ms     REAL,
    generation_latency_ms REAL,
    -- Cache
    answer_cache_hit      INTEGER DEFAULT 0,
    retrieval_cache_hit   INTEGER DEFAULT 0,
    -- Retrieval quality
    num_chunks_retrieved  INTEGER DEFAULT 0,
    mean_rerank_score     REAL,
    top_rerank_score      REAL,
    bm25_overlap          INTEGER DEFAULT 0,  -- chunks in both BM25 and dense
    -- Answer quality
    user_rating           INTEGER DEFAULT NULL,  -- 1-5, set after the fact
    answer_faithfulness   REAL    DEFAULT NULL,  -- critic score 0-1
    created_at            REAL    NOT NULL
);

-- ── Drift log ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS drift_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    window_queries      INTEGER,
    mean_score_recent   REAL,
    mean_score_baseline REAL,
    drift_delta         REAL,
    action_taken        TEXT,
    logged_at           REAL NOT NULL
);

-- ── Indices ───────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_answer_hash    ON answer_cache(query_hash);
CREATE INDEX IF NOT EXISTS idx_retrieval_hash ON retrieval_cache(query_hash);
CREATE INDEX IF NOT EXISTS idx_metrics_time   ON query_metrics(created_at);
CREATE INDEX IF NOT EXISTS idx_chunks_doc     ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_rewrites_help  ON query_rewrites(was_helpful);
"""


class Database:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._bootstrap()

    def _bootstrap(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")   # concurrent reads + writes
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def now(self) -> float:
        return time.time()