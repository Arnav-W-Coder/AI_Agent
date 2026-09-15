"""
web_store.py — Persistent web chunk store backed by ChromaDB + SQLite.
"""
import json
import logging
import time
import uuid
from urllib.parse import urlsplit

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from checkpoints import checkpoint
from config import RAGConfig
from db import Database

log = logging.getLogger(__name__)
_COLLECTION = "web_cache"


class WebChunkStore:
    """Persistent, TTL-aware web chunk store with explicit source provenance."""

    def __init__(self, db: Database, cfg: RAGConfig, embeddings: OllamaEmbeddings) -> None:
        self.db = db
        self.cfg = cfg
        self.embeddings = embeddings
        self._splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
        self._chroma = Chroma(collection_name=_COLLECTION, persist_directory=str(cfg.web_chroma_dir), embedding_function=embeddings)
        self._validate_embedding_contract()
        stale = self._evict_stale()
        n = self._chroma._collection.count()
        log.info("[WebStore] Ready: %d chunks cached | %d stale chunks evicted on startup", n, stale)
        checkpoint("web_store.ready", self.stats(), enabled=cfg.debug_checkpoints,
                   preview_chars=cfg.checkpoint_preview_chars, sample_items=cfg.checkpoint_sample_items)

    def _validate_embedding_contract(self) -> None:
        vector = self.embeddings.embed_documents(["__web_embedding_dimension_probe__"])[0]
        dim = len(vector)
        try:
            peek = self._chroma._collection.peek(limit=1)
            vectors = peek.get("embeddings")
            if vectors is not None and len(vectors) > 0:
                stored_dim = len(vectors[0])
                if stored_dim != dim:
                    raise RuntimeError(
                        f"Web Chroma collection dimension mismatch: collection={stored_dim}, model={dim}. "
                        "Reset/rebuild chroma_web before using the persistent web cache."
                    )
        except RuntimeError:
            raise
        except Exception as exc:
            log.warning("[WebStore] Could not inspect existing embedding dimension: %s", exc)
        log.info("[WebStore] Embedding dimension=%d", dim)
        checkpoint("web_store.embedding_contract", {"dimension": dim}, enabled=self.cfg.debug_checkpoints,
                   preview_chars=self.cfg.checkpoint_preview_chars, sample_items=self.cfg.checkpoint_sample_items)

    def is_fresh(self, url: str) -> bool:
        with self.db.connect() as conn:
            row = conn.execute("SELECT expires_at FROM web_scrape_cache WHERE url = ?", (url,)).fetchone()
        fresh = row is not None and row["expires_at"] > time.time()
        checkpoint("web_store.freshness_check", {"url": url, "fresh": fresh}, enabled=self.cfg.debug_checkpoints,
                   preview_chars=self.cfg.checkpoint_preview_chars, sample_items=self.cfg.checkpoint_sample_items)
        return fresh

    def upsert(self, url: str, title: str, text: str, metadata: dict | None = None) -> int:
        return self.upsert_batch([(url, title, text, metadata or {})])

    def upsert_batch(self, pages: list[tuple[str, str, str, dict]]) -> int:
        """Embed and persist multiple pages in batches instead of one request per page."""
        pending = []
        for url, title, text, metadata in pages:
            if self.is_fresh(url):
                log.info("[WebStore] Cache hit — skipping re-embed: %s", url[:70])
                continue
            if not text or text.startswith("Error:"):
                continue
            docs = self._splitter.create_documents([text])
            for doc in docs:
                pending.append((url, title, doc.page_content, metadata))

        if not pending:
            return 0

        self._maybe_evict_oldest()
        now = time.time()
        expires_at = now + self.cfg.web_chunk_ttl_hours * 3600
        chunk_ids = [str(uuid.uuid4()) for _ in pending]
        texts = [item[2] for item in pending]
        metadatas = [{
            "source": url, "source_url": url,
            "domain": urlsplit(url).hostname or "", "title": title,
            "source_type": "web", "scraped_at": now, "expires_at": expires_at,
            "domain_score": metadata.get("domain_score", 0),
            "domain_score_reasons": json.dumps(metadata.get("domain_score_reasons", [])),
            "page_quality_score": metadata.get("page_quality_score", 0.0),
            "query_relevance": metadata.get("query_relevance", 0.0),
            "http_status": metadata.get("http_status", 0),
            "redirect_count": metadata.get("redirect_count", 0),
            "final_url": metadata.get("final_url", url),
            "meaningful_paragraphs": metadata.get("meaningful_paragraphs", 0),
        } for url, title, _, metadata in pending]

        embeddings_list = []
        batch_size = max(1, int(self.cfg.web_embed_batch_size))
        try:
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                vectors = self.embeddings.embed_documents(batch)
                embeddings_list.extend(vectors)
                checkpoint("web_store.embedding_batch", {
                    "start": start, "batch_size": len(batch),
                    "vectors": len(vectors),
                    "dimension": len(vectors[0]) if vectors else 0,
                }, enabled=self.cfg.debug_checkpoints,
                           preview_chars=self.cfg.checkpoint_preview_chars,
                           sample_items=self.cfg.checkpoint_sample_items)
            if len(embeddings_list) != len(texts):
                raise RuntimeError("Embedding count did not match web chunk count")
            self._chroma._collection.upsert(ids=chunk_ids, documents=texts,
                                            embeddings=embeddings_list, metadatas=metadatas)
        except Exception as exc:
            log.error("[WebStore] Chroma batch upsert failed: %s", exc)
            return 0

        by_url = {}
        for chunk_id, (url, title, _, _) in zip(chunk_ids, pending):
            by_url.setdefault((url, title), []).append(chunk_id)
        with self.db.connect() as conn:
            for (url, title), ids in by_url.items():
                conn.execute("""INSERT OR REPLACE INTO web_scrape_cache
                       (url, title, scraped_at, expires_at, chunk_count, chunk_ids)
                       VALUES (?, ?, ?, ?, ?, ?)""", (url, title, now, expires_at, len(ids), json.dumps(ids)))
        checkpoint("web_store.upsert_complete", {"urls": len(by_url), "chunks": len(pending)},
                   enabled=self.cfg.debug_checkpoints, preview_chars=self.cfg.checkpoint_preview_chars,
                   sample_items=self.cfg.checkpoint_sample_items)
        return len(pending)

    def search(self, query: str, k: int) -> list[dict]:
        n_total = self._chroma._collection.count()
        if n_total == 0:
            return []
        k_actual = min(k, n_total)
        checkpoint("web_store.search_input", query, enabled=self.cfg.debug_checkpoints,
                   preview_chars=self.cfg.checkpoint_preview_chars, sample_items=self.cfg.checkpoint_sample_items,
                   requested_k=k, available_chunks=n_total)
        try:
            query_vector = self.embeddings.embed_documents([query])[0]
            results = self._chroma._collection.query(query_embeddings=[query_vector], n_results=k_actual)
        except Exception as exc:
            log.error("[WebStore] Search failed: %s", exc)
            return []

        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        output = [{
            "chroma_id": cid, "text": text or "", "filename": (meta or {}).get("source", "web"),
            "source_url": (meta or {}).get("source_url") or (meta or {}).get("source", ""),
            "domain": (meta or {}).get("domain", ""), "page_number": 0,
            "rerank_score": round(float(1 - distance), 3), "source_type": "web",
            "title": (meta or {}).get("title", ""), "scraped_at": (meta or {}).get("scraped_at"),
            "domain_score": (meta or {}).get("domain_score"),
            "domain_score_reasons": json.loads((meta or {}).get("domain_score_reasons", "[]")),
            "page_quality_score": (meta or {}).get("page_quality_score", 0.0),
            "query_relevance": (meta or {}).get("query_relevance", 0.0),
            "retrieval_score": round(float(1 - distance), 3),
            "http_status": (meta or {}).get("http_status"),
            "redirect_count": (meta or {}).get("redirect_count"),
            "final_url": (meta or {}).get("final_url", ""),
            "meaningful_paragraphs": (meta or {}).get("meaningful_paragraphs", 0),
        } for cid, text, meta, distance in zip(ids, docs, metas, distances)]
        checkpoint("web_store.search_output", output, enabled=self.cfg.debug_checkpoints,
                   preview_chars=self.cfg.checkpoint_preview_chars, sample_items=self.cfg.checkpoint_sample_items,
                   query=query, returned=len(output))
        return output

    def stats(self) -> dict:
        n_chunks = self._chroma._collection.count()
        with self.db.connect() as conn:
            row = conn.execute("SELECT COUNT(*) as n_urls, AVG(chunk_count) as avg FROM web_scrape_cache").fetchone()
        cap = self.cfg.web_collection_max_chunks
        return {
            "total_chunks": n_chunks, "cached_urls": row["n_urls"] or 0,
            "avg_chunks_per_url": round(row["avg"] or 0.0, 1),
            "capacity_used_pct": round(n_chunks / cap * 100, 1) if cap else 0.0,
            "ttl_hours": self.cfg.web_chunk_ttl_hours,
        }

    def _evict_stale(self) -> int:
        now = time.time()
        with self.db.connect() as conn:
            stale_rows = conn.execute("SELECT url, chunk_ids FROM web_scrape_cache WHERE expires_at < ?", (now,)).fetchall()
        if not stale_rows:
            return 0
        total_removed = 0
        for row in stale_rows:
            ids = json.loads(row["chunk_ids"] or "[]")
            if ids:
                try:
                    self._chroma._collection.delete(ids=ids)
                    total_removed += len(ids)
                except Exception as exc:
                    log.warning("[WebStore] Eviction error (%s): %s", row["url"], exc)
        stale_urls = [r["url"] for r in stale_rows]
        with self.db.connect() as conn:
            conn.execute(f"DELETE FROM web_scrape_cache WHERE url IN ({','.join('?' * len(stale_urls))})", stale_urls)
        checkpoint("web_store.stale_eviction", stale_urls, enabled=self.cfg.debug_checkpoints,
                   preview_chars=self.cfg.checkpoint_preview_chars, sample_items=self.cfg.checkpoint_sample_items,
                   urls_removed=len(stale_urls), chunks_removed=total_removed)
        return total_removed

    def _maybe_evict_oldest(self) -> None:
        n = self._chroma._collection.count()
        cap = self.cfg.web_collection_max_chunks
        if n < cap * 0.9:
            return
        with self.db.connect() as conn:
            n_urls = conn.execute("SELECT COUNT(*) FROM web_scrape_cache").fetchone()[0]
            target = max(1, n_urls // 5)
            old_rows = conn.execute("SELECT url, chunk_ids FROM web_scrape_cache ORDER BY scraped_at ASC LIMIT ?", (target,)).fetchall()
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
                conn.execute(f"DELETE FROM web_scrape_cache WHERE url IN ({','.join('?' * len(old_urls))})", old_urls)
        log.info("[WebStore] Capacity eviction: removed %d chunks", removed)
        checkpoint("web_store.capacity_eviction", old_urls, enabled=self.cfg.debug_checkpoints,
                   preview_chars=self.cfg.checkpoint_preview_chars, sample_items=self.cfg.checkpoint_sample_items,
                   urls_removed=len(old_urls), chunks_removed=removed)
