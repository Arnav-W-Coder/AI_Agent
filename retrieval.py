"""
retrieval.py — Hybrid child retrieval: BM25 + Chroma dense + RRF + reranking.

Retrieval operates on small child chunks. After reranking, winners expand to
parent context units and nearby parent sections under a strict context budget.
"""
import logging
import re
from typing import Optional

import numpy as np
import tiktoken
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_chroma import Chroma

from config import RAGConfig
from db import Database

log = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _token_count(text: str) -> int:
    try:
        return len(tiktoken.get_encoding("cl100k_base").encode(text or ""))
    except Exception:
        return max(1, len((text or "").split()))


class BM25Index:
    """In-memory BM25 index over the FULL text of retrieval children."""
    def __init__(self, db: Database) -> None:
        self.db = db
        self._chunks: list[dict] = []
        self._bm25: Optional[BM25Okapi] = None
        self.rebuild()

    def rebuild(self) -> None:
        with self.db.connect() as conn:
            rows = conn.execute(
                """SELECT c.chroma_id, c.text, c.text_preview, c.page_number,
                          c.end_page, c.doc_id, c.parent_id, c.section_path,
                          d.filename
                   FROM chunks c JOIN documents d ON c.doc_id = d.id
                   WHERE c.chunk_type = 'child' AND c.chroma_id <> ''"""
            ).fetchall()
        self._chunks = [dict(r) for r in rows]
        if not self._chunks:
            self._bm25 = None
            log.info("[BM25] No child chunks — index empty")
            return
        self._bm25 = BM25Okapi([_tokenize(c.get("text") or c.get("text_preview") or "") for c in self._chunks])
        log.info("[BM25] Index rebuilt: %d child chunks", len(self._chunks))

    def search(self, query: str, top_k: int) -> list[dict]:
        if self._bm25 is None or not self._chunks:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        top_i = np.argsort(scores)[::-1][:top_k]
        return [{**self._chunks[i], "bm25_score": float(scores[i])} for i in top_i if scores[i] > 0]


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        log.info("[Reranker] Loading cross-encoder: %s", model_name)
        self._model = CrossEncoder(model_name)
        log.info("[Reranker] Ready")

    def rerank(self, query: str, chunks: list[dict], top_k: int, min_score: float) -> list[dict]:
        if not chunks:
            return []
        scores = self._model.predict([(query, c.get("text", c.get("text_preview", ""))) for c in chunks]).tolist()
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = round(float(score), 4)
        ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
        return [c for c in ranked if c["rerank_score"] >= min_score][:top_k]


class HybridRetriever:
    def __init__(self, vectorstore: Chroma, bm25_index: BM25Index,
                 reranker: CrossEncoderReranker, cfg: RAGConfig) -> None:
        self.vs = vectorstore
        self.bm25 = bm25_index
        self.reranker = reranker
        self.cfg = cfg
        # BM25 already owns the canonical SQLite connection target.
        self.db: Database = bm25_index.db

    def _rrf_score(self, rank: int) -> float:
        return 1.0 / (self.cfg.rrf_k + rank + 1)

    def _dense_search(self, query: str, top_k: int, metadata_filter: Optional[dict]) -> list[dict]:
        kwargs = {"query_texts": [query], "n_results": top_k}
        if metadata_filter:
            kwargs["where"] = metadata_filter
        try:
            results = self.vs._collection.query(**kwargs)
        except Exception as exc:
            log.error("[Dense] ChromaDB error: %s", exc)
            return []
        chunks = []
        for cid, text, meta, dist in zip(
            results.get("ids", [[]])[0], results.get("documents", [[]])[0],
            results.get("metadatas", [[]])[0], results.get("distances", [[]])[0]
        ):
            meta = meta or {}
            chunks.append({
                "chroma_id": cid, "text": text or "", "doc_id": meta.get("doc_id", ""),
                "parent_id": meta.get("parent_id", ""), "filename": meta.get("source", ""),
                "page_number": meta.get("page", 0), "end_page": meta.get("end_page", meta.get("page", 0)),
                "section_path": meta.get("section_path", ""), "chunk_type": "child",
                "dense_score": round(1 - float(dist), 4),
            })
        return chunks

    def _fuse(self, dense_chunks: list[dict], bm25_chunks: list[dict]) -> tuple[list[dict], set, set]:
        dense_ids = {c["chroma_id"] for c in dense_chunks}
        bm25_ids = {c["chroma_id"] for c in bm25_chunks}
        rrf: dict[str, dict] = {}
        for rank, chunk in enumerate(dense_chunks):
            cid = chunk["chroma_id"]
            rrf.setdefault(cid, {**chunk, "rrf_score": 0.0})
            rrf[cid]["rrf_score"] += self._rrf_score(rank)
        for rank, chunk in enumerate(bm25_chunks):
            cid = chunk["chroma_id"]
            if cid not in rrf:
                try:
                    fetched = self.vs._collection.get(ids=[cid], include=["documents", "metadatas"])
                    text = fetched.get("documents", [""])[0] if fetched.get("documents") else ""
                    meta = fetched.get("metadatas", [{}])[0] if fetched.get("metadatas") else {}
                except Exception:
                    text, meta = chunk.get("text", chunk.get("text_preview", "")), {}
                rrf[cid] = {
                    **chunk, "text": text,
                    "filename": meta.get("source", chunk.get("filename", "")),
                    "page_number": meta.get("page", chunk.get("page_number", 0)),
                    "parent_id": meta.get("parent_id", chunk.get("parent_id", "")),
                    "section_path": meta.get("section_path", chunk.get("section_path", "")),
                    "rrf_score": 0.0,
                }
            rrf[cid]["rrf_score"] += self._rrf_score(rank)
        candidates = sorted(rrf.values(), key=lambda c: c["rrf_score"], reverse=True)
        return candidates[:max(self.cfg.top_k_dense, self.cfg.top_k_sparse)], bm25_ids, dense_ids

    def retrieve(self, query: str, metadata_filter: Optional[dict] = None) -> tuple[list[dict], set, set]:
        dense = self._dense_search(query, self.cfg.top_k_dense, metadata_filter)
        bm25 = self.bm25.search(query, self.cfg.top_k_sparse)
        candidates, bm25_ids, dense_ids = self._fuse(dense, bm25)
        reranked = self.reranker.rerank(query, candidates, self.cfg.top_k_rerank, self.cfg.min_rerank_score)
        return reranked, bm25_ids, dense_ids

    def retrieve_candidates(self, query: str, metadata_filter: Optional[dict] = None) -> tuple[list[dict], set, set]:
        children, bm25_ids, dense_ids = self.retrieve(query, metadata_filter)
        return self.expand_to_context(children), bm25_ids, dense_ids

    def expand_to_context(self, children: list[dict]) -> list[dict]:
        """Expand children to parents + adjacent parent sections within budget."""
        if not children:
            return []
        parent_ids = list(dict.fromkeys(c.get("parent_id") for c in children if c.get("parent_id")))
        if not parent_ids:
            return children
        placeholders = ",".join("?" for _ in parent_ids)
        with self.db.connect() as conn:
            rows = conn.execute(
                f"""SELECT id, doc_id, chunk_index, text, page_number, end_page, section_path
                    FROM chunks WHERE chunk_type='parent' AND id IN ({placeholders})""", parent_ids
            ).fetchall()
            parent_map = {r["id"]: dict(r) for r in rows}

            # Preserve document-local ordering so neighbors are meaningful.
            neighbors: list[dict] = []
            for parent in rows:
                for delta in range(1, self.cfg.context_neighbor_count + 1):
                    for idx in (parent["chunk_index"] - delta, parent["chunk_index"] + delta):
                        row = conn.execute(
                            """SELECT id, doc_id, chunk_index, text, page_number, end_page, section_path
                               FROM chunks WHERE chunk_type='parent' AND doc_id=? AND chunk_index=?""",
                            (parent["doc_id"], idx),
                        ).fetchone()
                        if row:
                            neighbors.append(dict(row))

        selected: list[dict] = []
        seen: set[str] = set()
        budget = self.cfg.context_budget_tokens
        for child in children:
            pid = child.get("parent_id")
            parent = parent_map.get(pid)
            if not parent or pid in seen:
                continue
            tokens = _token_count(parent["text"])
            if selected and sum(_token_count(x["text"]) for x in selected) + tokens > budget:
                continue
            if not selected and tokens > budget:
                parent["text"] = " ".join(parent["text"].split()[:budget])
            selected.append({
                **child, "text": parent["text"], "chunk_type": "parent_context",
                "parent_id": pid, "page_number": parent["page_number"],
                "end_page": parent["end_page"], "section_path": parent["section_path"],
            })
            seen.add(pid)

        # Neighbors are lower-priority context and never displace retrieved parents.
        for neighbor in neighbors:
            nid = neighbor["id"]
            if nid in seen:
                continue
            used = sum(_token_count(x["text"]) for x in selected)
            if used + _token_count(neighbor["text"]) > budget:
                continue
            selected.append({
                "text": neighbor["text"], "filename": "", "page_number": neighbor["page_number"],
                "end_page": neighbor["end_page"], "section_path": neighbor["section_path"],
                "doc_id": neighbor["doc_id"], "parent_id": nid, "chunk_type": "neighbor_context",
                "rerank_score": 0.0, "rrf_score": 0.0,
            })
            seen.add(nid)
        return selected or children
