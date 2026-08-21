"""
retrieval.py — Hybrid retrieval: BM25 (sparse) + ChromaDB (dense) + reranking.

Pipeline per query:
  1. Metadata filter  — narrows ChromaDB search to matching documents
  2. Dense ANN search — ChromaDB HNSW, top_k_dense candidates
  3. Sparse BM25      — rank_bm25 on all indexed chunks, top_k_sparse candidates
  4. RRF fusion       — merge both lists using Reciprocal Rank Fusion
  5. Cross-encoder    — sentence-transformers reranks merged list, keep top_k_rerank

BM25 index:
  Built in-memory from all chunk texts stored in SQLite. Rebuilt after ingestion.
  Lightweight for tens of thousands of chunks; for millions, switch to Elasticsearch.
"""
import logging
import json
import re
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_chroma import Chroma

from config import RAGConfig
from db import Database

log = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """
    In-memory BM25 index over all chunk texts.
    Rebuilt from SQLite on startup and after ingestion.
    """

    def __init__(self, db: Database) -> None:
        self.db = db
        self._chunks: list[dict] = []   # [{chroma_id, text, doc_id, page}, ...]
        self._bm25:   Optional[BM25Okapi] = None
        self.rebuild()

    def rebuild(self) -> None:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT c.chroma_id, c.text_preview, c.page_number, c.doc_id, "
                "       d.filename "
                "FROM chunks c JOIN documents d ON c.doc_id = d.id"
            ).fetchall()

        self._chunks = [dict(r) for r in rows]

        if not self._chunks:
            self._bm25 = None
            log.info("[BM25] No chunks — index empty")
            return

        corpus   = [_tokenize(c["text_preview"] or "") for c in self._chunks]
        self._bm25 = BM25Okapi(corpus)
        log.info(f"[BM25] Index rebuilt: {len(self._chunks)} chunks")

    def search(self, query: str, top_k: int) -> list[dict]:
        """Returns list of {chroma_id, score, doc_id, filename, page_number}."""
        if self._bm25 is None or not self._chunks:
            return []
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_i  = np.argsort(scores)[::-1][:top_k]
        return [
            {
                **self._chunks[i],
                "bm25_score": float(scores[i]),
            }
            for i in top_i
            if scores[i] > 0
        ]


class CrossEncoderReranker:
    """
    Wraps sentence-transformers CrossEncoder.
    Downloads ~80MB model on first use (requires internet on first run only).
    """

    def __init__(self, model_name: str) -> None:
        log.info(f"[Reranker] Loading cross-encoder: {model_name}")
        self._model = CrossEncoder(model_name)
        log.info("[Reranker] Ready")

    def rerank(
        self, query: str, chunks: list[dict], top_k: int, min_score: float
    ) -> list[dict]:
        """
        Score (query, passage) pairs and return top_k above min_score.
        Adds 'rerank_score' key to each chunk dict.
        """
        if not chunks:
            return []

        pairs  = [(query, c.get("text", c.get("text_preview", ""))) for c in chunks]
        scores = self._model.predict(pairs).tolist()

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = round(float(score), 4)

        ranked   = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
        filtered = [c for c in ranked if c["rerank_score"] >= min_score]
        return filtered[:top_k]


class HybridRetriever:
    """
    Combines dense (ChromaDB) + sparse (BM25) search via RRF,
    then reranks with a cross-encoder.

    retrieve() returns:
      - chunks:      final reranked list with 'text', 'rerank_score', metadata
      - bm25_ids:    set of chroma_ids from BM25 (for overlap metric)
      - dense_ids:   set of chroma_ids from dense search
    """

    def __init__(
        self,
        vectorstore: Chroma,
        bm25_index:  BM25Index,
        reranker:    CrossEncoderReranker,
        cfg:         RAGConfig,
    ) -> None:
        self.vs      = vectorstore
        self.bm25    = bm25_index
        self.reranker = reranker
        self.cfg     = cfg

    def _rrf_score(self, rank: int) -> float:
        return 1.0 / (self.cfg.rrf_k + rank + 1)

    def _dense_search(
        self, query: str, top_k: int, metadata_filter: Optional[dict]
    ) -> list[dict]:
        """Query ChromaDB HNSW index with optional metadata filter."""
        kwargs: dict = {"query_texts": [query], "n_results": top_k}
        if metadata_filter:
            kwargs["where"] = metadata_filter

        try:
            results = self.vs._collection.query(**kwargs)
        except Exception as e:
            log.error(f"[Dense] ChromaDB error: {e}")
            return []

        chunks = []
        ids_list  = results.get("ids",       [[]])[0]
        docs_list = results.get("documents", [[]])[0]
        metas     = results.get("metadatas", [[]])[0]
        dists     = results.get("distances", [[]])[0]

        for chroma_id, text, meta, dist in zip(ids_list, docs_list, metas, dists):
            chunks.append({
                "chroma_id":    chroma_id,
                "text":         text,
                "doc_id":       (meta or {}).get("doc_id",   ""),
                "filename":     (meta or {}).get("source",   ""),
                "page_number":  (meta or {}).get("page",     0),
                "dense_score":  round(1 - float(dist), 4),   # convert L2 → similarity
            })
        return chunks

    def retrieve(
        self, query: str, metadata_filter: Optional[dict] = None
    ) -> tuple[list[dict], set, set]:
        """
        Full hybrid retrieval pipeline.
        Returns (reranked_chunks, bm25_chroma_ids, dense_chroma_ids).
        """
        # ── Dense ─────────────────────────────────────────────────────────────
        dense_chunks = self._dense_search(query, self.cfg.top_k_dense, metadata_filter)
        dense_ids    = {c["chroma_id"] for c in dense_chunks}

        # ── Sparse (BM25) ─────────────────────────────────────────────────────
        bm25_chunks = self.bm25.search(query, self.cfg.top_k_sparse)
        bm25_ids    = {c["chroma_id"] for c in bm25_chunks}

        log.info(
            f"[Retrieval] Dense: {len(dense_chunks)} | "
            f"BM25: {len(bm25_chunks)} | "
            f"Overlap: {len(dense_ids & bm25_ids)}"
        )

        # ── RRF fusion ────────────────────────────────────────────────────────
        # Assign each chunk an RRF score from whichever lists it appears in.
        rrf: dict[str, dict] = {}

        for rank, chunk in enumerate(dense_chunks):
            cid = chunk["chroma_id"]
            if cid not in rrf:
                rrf[cid] = dict(chunk)
                rrf[cid]["rrf_score"] = 0.0
            rrf[cid]["rrf_score"] += self._rrf_score(rank)

        for rank, chunk in enumerate(bm25_chunks):
            cid = chunk["chroma_id"]
            if cid not in rrf:
                # BM25 chunks may not have full text — fetch from Chroma
                try:
                    fetched = self.vs._collection.get(
                        ids=[cid], include=["documents", "metadatas"]
                    )
                    text = fetched["documents"][0] if fetched["documents"] else \
                           chunk.get("text_preview", "")
                    meta = fetched["metadatas"][0] if fetched["metadatas"] else {}
                except Exception:
                    text = chunk.get("text_preview", "")
                    meta = {}
                rrf[cid] = {
                    **chunk,
                    "text":        text,
                    "filename":    meta.get("source",   chunk.get("filename", "")),
                    "page_number": meta.get("page",     chunk.get("page_number", 0)),
                    "rrf_score":   0.0,
                }
            rrf[cid]["rrf_score"] += self._rrf_score(rank)

        # Sort by RRF score, take top candidates for reranking
        candidates = sorted(rrf.values(), key=lambda c: c["rrf_score"], reverse=True)
        candidates = candidates[: self.cfg.top_k_dense]  # cap before rerank

        # ── Cross-encoder rerank ───────────────────────────────────────────────
        reranked = self.reranker.rerank(
            query, candidates,
            self.cfg.top_k_rerank,
            self.cfg.min_rerank_score,
        )

        log.info(
            f"[Rerank] {len(candidates)} → {len(reranked)} chunks | "
            f"top score: {reranked[0]['rerank_score'] if reranked else 'n/a'}"
        )
        return reranked, bm25_ids, dense_ids