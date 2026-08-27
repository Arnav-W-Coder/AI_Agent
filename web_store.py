"""
web_store.py — Persistent web chunk store backed by ChromaDB + SQLite.

Replaces the in-memory temp Chroma store that was created and destroyed on
every web query. At scale that approach fails for two reasons:
  1. RAM: N concurrent queries × M scraped pages × 768 floats × 4 bytes = GBs
  2. Wasted work: every query re-embeds pages that were already embedded before.

New architecture:
  ChromaDB collection "web_cache" persisted to ./chroma_web/
    - Same HNSW index, same similarity search, but survives restarts
    - Searched across ALL cached content, not just pages from the current query
    - Cross-query reuse: a page scraped for query A is immediately available for B

  SQLite table web_scrape_cache (added in db.py schema)
    - Tracks URL → (scraped_at, expires_at, chunk_ids[])
    - Enables targeted eviction: delete specific Chroma IDs by URL
    - Prevents re-scraping within TTL window without reading Chroma at all

Eviction strategy (two triggers):
  1. Startup eviction: remove all URLs past expires_at (TTL-based)
  2. Capacity eviction: when 90% of web_collection_max_chunks is reached,
     evict the oldest 20% of URLs by scrape timestamp

The result: the web store behaves like a rolling knowledge cache —
recent content is fast and free to query; old content quietly expires.
"""
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import RAGConfig
from db import Database

log = logging.getLogger(__name__)

_COLLECTION = "web_cache"


class WebChunkStore:
    """
    Persistent, TTL-aware web chunk store.

    Typical usage in pipeline._web_scrape_chunks():
        for url, title, text in scraped_pages:
            self.web_store.upsert(url, title, text)   # no-op if still fresh
        return self.web_store.search(query, k=cfg.web_top_k)
    """

    def __init__(
        self,
        db:         Database,
        cfg:        RAGConfig,
        embeddings: OllamaEmbeddings,
    ) -> None:
        self.db         = db
        self.cfg        = cfg
        self.embeddings = embeddings
        self._splitter  = RecursiveCharacterTextSplitter(
            chunk_size    = cfg.chunk_size,
            chunk_overlap = cfg.chunk_overlap,
        )

        # Dedicated persistent Chroma collection — never mixed with PDF store
        self._chroma = Chroma(
            collection_name   = _COLLECTION,
            persist_directory = str(cfg.web_chroma_dir),
            embedding_function= embeddings,
        )

        stale = self._evict_stale()
        n     = self._chroma._collection.count()
        log.info(
            f"[WebStore] Ready: {n} chunks cached | "
            f"{stale} stale chunks evicted on startup"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def is_fresh(self, url: str) -> bool:
        """True if this URL was scraped within the configured TTL window."""
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT expires_at FROM web_scrape_cache WHERE url = ?", (url,)
            ).fetchone()
        return row is not None and row["expires_at"] > time.time()

    def upsert(self, url: str, title: str, text: str) -> int:
        """
        Chunk, embed, and persist content from one URL.

        Skips silently if the URL is already fresh (TTL not expired).
        If the collection is near capacity, evicts the oldest URLs first.
        Returns the number of new chunks stored (0 = cache hit, no work done).
        """
        if self.is_fresh(url):
            log.info(f"[WebStore] Cache hit — skipping re-embed: {url[:70]}")
            return 0

        if not text or text.startswith("Error:"):
            return 0

        docs = self._splitter.create_documents([text])
        if not docs:
            return 0

        # Evict oldest URLs if near capacity before adding new content
        self._maybe_evict_oldest()

        now        = time.time()
        expires_at = now + self.cfg.web_chunk_ttl_hours * 3600
        chunk_ids  = [str(uuid.uuid4()) for _ in docs]
        texts      = [d.page_content for d in docs]
        metadatas  = [
            {
                "source":     url,
                "title":      title,
                "scraped_at": now,
                "expires_at": expires_at,
            }
            for _ in docs
        ]

        # Embed and upsert into the persistent Chroma collection
        try:
            embeddings_list = self.embeddings.embed_documents(texts)
            self._chroma._collection.upsert(
                ids        = chunk_ids,
                documents  = texts,
                embeddings = embeddings_list,
                metadatas  = metadatas,
            )
        except Exception as e:
            log.error(f"[WebStore] Chroma upsert failed for {url}: {e}")
            return 0

        # Record in SQLite so we can evict by URL later
        with self.db.connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO web_scrape_cache
                   (url, title, scraped_at, expires_at, chunk_count, chunk_ids)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (url, title, now, expires_at, len(chunk_ids), json.dumps(chunk_ids))
            )

        log.info(f"[WebStore] Stored {len(docs)} chunks from {url[:70]}")
        return len(docs)

    def search(self, query: str, k: int) -> list[dict]:
        """
        Similarity search across ALL cached web chunks (not just from this query).
        This is the key advantage over the old temp store: previous queries
        may have already cached highly relevant content.
        """
        n_total = self._chroma._collection.count()
        if n_total == 0:
            log.info("[WebStore] Collection empty — no web results.")
            return []

        k_actual = min(k, n_total)
        try:
            results = self._chroma.similarity_search_with_score(query, k=k_actual)
        except Exception as e:
            log.error(f"[WebStore] Search failed: {e}")
            return []

        return [
            {
                "text":         doc.page_content,
                "filename":     doc.metadata.get("source", "web"),
                "page_number":  0,
                "rerank_score": round(float(1 - score), 3),
                "source_type":  "web",
                "title":        doc.metadata.get("title", ""),
            }
            for doc, score in results
        ]

    def stats(self) -> dict:
        """Summary stats included in monitoring_report()."""
        n_chunks = self._chroma._collection.count()
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as n_urls, AVG(chunk_count) as avg FROM web_scrape_cache"
            ).fetchone()
        cap = self.cfg.web_collection_max_chunks
        return {
            "total_chunks":        n_chunks,
            "cached_urls":         row["n_urls"] or 0,
            "avg_chunks_per_url":  round(row["avg"] or 0.0, 1),
            "capacity_used_pct":   round(n_chunks / cap * 100, 1) if cap else 0.0,
            "ttl_hours":           self.cfg.web_chunk_ttl_hours,
        }

    # ── Eviction ──────────────────────────────────────────────────────────────

    def _evict_stale(self) -> int:
        """
        Delete all chunks whose TTL has expired.
        Called on startup and can be called manually for scheduled cleanup.
        Returns total number of Chroma chunks removed.
        """
        now = time.time()
        with self.db.connect() as conn:
            stale_rows = conn.execute(
                "SELECT url, chunk_ids FROM web_scrape_cache WHERE expires_at < ?",
                (now,)
            ).fetchall()

        if not stale_rows:
            return 0

        total_removed = 0
        for row in stale_rows:
            ids = json.loads(row["chunk_ids"] or "[]")
            if ids:
                try:
                    self._chroma._collection.delete(ids=ids)
                    total_removed += len(ids)
                except Exception as e:
                    log.warning(f"[WebStore] Eviction error ({row['url']}): {e}")

        stale_urls = [r["url"] for r in stale_rows]
        with self.db.connect() as conn:
            conn.execute(
                f"DELETE FROM web_scrape_cache WHERE url IN "
                f"({','.join('?'*len(stale_urls))})",
                stale_urls
            )

        log.info(
            f"[WebStore] Evicted {total_removed} stale chunks "
            f"from {len(stale_rows)} expired URLs"
        )
        return total_removed

    def _maybe_evict_oldest(self) -> None:
        """
        If the collection is >= 90% of capacity, evict the oldest 20% of URLs
        (by scrape timestamp) to make room for new content.
        """
        n   = self._chroma._collection.count()
        cap = self.cfg.web_collection_max_chunks

        if n < cap * 0.9:
            return

        log.info(
            f"[WebStore] Near capacity ({n}/{cap}) — evicting oldest 20% of URLs"
        )

        with self.db.connect() as conn:
            n_urls = conn.execute(
                "SELECT COUNT(*) FROM web_scrape_cache"
            ).fetchone()[0]

            target   = max(1, n_urls // 5)
            old_rows = conn.execute(
                "SELECT url, chunk_ids FROM web_scrape_cache "
                "ORDER BY scraped_at ASC LIMIT ?",
                (target,)
            ).fetchall()

        removed = 0
        for row in old_rows:
            ids = json.loads(row["chunk_ids"] or "[]")
            if ids:
                try:
                    self._chroma._collection.delete(ids=ids)
                    removed += len(ids)
                except Exception:
                    pass

        old_urls = [r["url"] for r in old_rows]
        if old_urls:
            with self.db.connect() as conn:
                conn.execute(
                    f"DELETE FROM web_scrape_cache WHERE url IN "
                    f"({','.join('?'*len(old_urls))})",
                    old_urls
                )

        log.info(
            f"[WebStore] Capacity eviction: removed {removed} chunks "
            f"from {len(old_rows)} oldest URLs"
        )