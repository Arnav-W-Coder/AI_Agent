"""
db.py — SQLite schema and connection management.
Stores metadata and full chunk text; raw vectors live in ChromaDB.
"""
import sqlite3
import json
import time
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    filepath TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    page_count INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0,
    ingested_at REAL NOT NULL,
    metadata_json TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(id),
    chroma_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER DEFAULT 0,
    text_preview TEXT,
    text TEXT,
    chunk_type TEXT DEFAULT 'child',
    parent_id TEXT,
    end_page INTEGER DEFAULT 0,
    section_path TEXT DEFAULT '',
    UNIQUE(doc_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS query_rewrites (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_query TEXT NOT NULL,
    rewritten_query TEXT NOT NULL,
    was_helpful INTEGER DEFAULT NULL,
    answer_score REAL DEFAULT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS answer_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_hash TEXT NOT NULL UNIQUE,
    query_text TEXT NOT NULL,
    query_embedding BLOB NOT NULL,
    answer TEXT NOT NULL,
    sources_json TEXT DEFAULT '[]',
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    hit_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS retrieval_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_hash TEXT NOT NULL UNIQUE,
    query_text TEXT NOT NULL,
    chunks_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    hit_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS query_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id TEXT NOT NULL UNIQUE,
    query_text TEXT NOT NULL,
    rewritten_query TEXT,
    total_latency_ms REAL,
    rewrite_latency_ms REAL,
    retrieval_latency_ms REAL,
    rerank_latency_ms REAL,
    generation_latency_ms REAL,
    answer_cache_hit INTEGER DEFAULT 0,
    retrieval_cache_hit INTEGER DEFAULT 0,
    num_chunks_retrieved INTEGER DEFAULT 0,
    mean_rerank_score REAL,
    top_rerank_score REAL,
    bm25_overlap INTEGER DEFAULT 0,
    user_rating INTEGER DEFAULT NULL,
    answer_faithfulness REAL DEFAULT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS web_scrape_cache (
    url TEXT PRIMARY KEY,
    title TEXT,
    scraped_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    chunk_count INTEGER DEFAULT 0,
    chunk_ids TEXT DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_web_cache_expires ON web_scrape_cache(expires_at);
CREATE INDEX IF NOT EXISTS idx_web_cache_scraped ON web_scrape_cache(scraped_at);

CREATE TABLE IF NOT EXISTS drift_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    window_queries INTEGER,
    mean_score_recent REAL,
    mean_score_baseline REAL,
    drift_delta REAL,
    action_taken TEXT,
    logged_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_answer_hash ON answer_cache(query_hash);
CREATE INDEX IF NOT EXISTS idx_retrieval_hash ON retrieval_cache(query_hash);
CREATE INDEX IF NOT EXISTS idx_metrics_time ON query_metrics(created_at);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_id);
CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(chunk_type);
CREATE INDEX IF NOT EXISTS idx_rewrites_help ON query_rewrites(was_helpful);
"""


class Database:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._bootstrap()

    def _bootstrap(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            # Non-destructive migration for existing rag.db files.
            columns = {r["name"] for r in conn.execute("PRAGMA table_info(chunks)").fetchall()}
            migrations = {
                "text": "ALTER TABLE chunks ADD COLUMN text TEXT",
                "chunk_type": "ALTER TABLE chunks ADD COLUMN chunk_type TEXT DEFAULT 'child'",
                "parent_id": "ALTER TABLE chunks ADD COLUMN parent_id TEXT",
                "end_page": "ALTER TABLE chunks ADD COLUMN end_page INTEGER DEFAULT 0",
                "section_path": "ALTER TABLE chunks ADD COLUMN section_path TEXT DEFAULT ''",
            }
            for name, statement in migrations.items():
                if name not in columns:
                    conn.execute(statement)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(chunk_type)")

    @contextmanager
    def connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
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
