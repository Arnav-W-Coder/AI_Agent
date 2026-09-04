"""
ingestion.py — Async, batched PDF ingestion with hierarchical chunking.

PDFs are split into structural sections, semantic blocks, bounded parents, and
small retrieval children. Only children are embedded/indexed; parents remain in
SQLite for context expansion after retrieval.
"""
import asyncio
import hashlib
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import RAGConfig
from db import Database
from chunking import HierarchicalChunker, ChunkRecord

log = logging.getLogger(__name__)


def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _load_pdf(path: Path) -> tuple[list[Document], int]:
    loader = PyPDFLoader(str(path))
    pages = loader.load()
    return pages, len(pages)


class AsyncIngestionPipeline:
    def __init__(self, db: Database, cfg: RAGConfig, vectorstore: Chroma, embeddings: OllamaEmbeddings) -> None:
        self.db = db
        self.cfg = cfg
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.chunker = HierarchicalChunker(cfg, self.embeddings.embed_documents)

    def _already_ingested(self, filepath: Path, file_hash: str) -> bool:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT file_hash FROM documents WHERE filepath = ?", (str(filepath),)
            ).fetchone()
        return row is not None and row["file_hash"] == file_hash

    async def _load_pdf_async(self, path: Path, executor: ThreadPoolExecutor) -> tuple[list[Document], int]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, _load_pdf, path)

    def _remove_old_document(self, filepath: Path) -> None:
        """Remove stale vectors/rows before replacing a changed document."""
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT id, chroma_id FROM chunks WHERE doc_id IN "
                "(SELECT id FROM documents WHERE filepath = ?)", (str(filepath),)
            ).fetchall()
            doc_ids = [r["id"] for r in conn.execute(
                "SELECT id FROM documents WHERE filepath = ?", (str(filepath),)
            ).fetchall()]
        if rows:
            try:
                self.vectorstore._collection.delete(ids=[r["chroma_id"] for r in rows])
            except Exception as exc:
                log.warning("[Ingestion] Could not remove old Chroma vectors: %s", exc)
        if doc_ids:
            with self.db.connect() as conn:
                conn.executemany("DELETE FROM chunks WHERE doc_id = ?", [(d,) for d in doc_ids])
                conn.executemany("DELETE FROM documents WHERE id = ?", [(d,) for d in doc_ids])

    async def _embed_batch(self, chunks: list[ChunkRecord], doc_id: str, executor: ThreadPoolExecutor) -> None:
        loop = asyncio.get_event_loop()
        texts = [c.text for c in chunks]
        vectors = await loop.run_in_executor(executor, self.embeddings.embed_documents, texts)
        chroma_ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = []
        for chunk in chunks:
            metadatas.append({
                "doc_id": doc_id,
                "parent_id": chunk.parent_id or "",
                "chunk_type": "child",
                "source": chunk.metadata.get("source", ""),
                "page": chunk.start_page,
                "end_page": chunk.end_page,
                "section_path": chunk.section_path,
            })
        self.vectorstore._collection.upsert(
            ids=chroma_ids, documents=texts, embeddings=vectors, metadatas=metadatas
        )
        with self.db.connect() as conn:
            for i, (chroma_id, chunk) in enumerate(zip(chroma_ids, chunks)):
                conn.execute(
                    """INSERT INTO chunks
                       (id, doc_id, chroma_id, chunk_index, page_number, text_preview,
                        text, chunk_type, parent_id, end_page, section_path)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (chunk.id, doc_id, chroma_id, chunk.chunk_index, chunk.start_page,
                     chunk.text[:200], chunk.text, "child", chunk.parent_id,
                     chunk.end_page, chunk.section_path),
                )

    def _store_parents(self, parents: list[ChunkRecord], doc_id: str, source: str) -> None:
        with self.db.connect() as conn:
            for parent in parents:
                conn.execute(
                    """INSERT INTO chunks
                       (id, doc_id, chroma_id, chunk_index, page_number, text_preview,
                        text, chunk_type, parent_id, end_page, section_path)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (parent.id, doc_id, "", parent.chunk_index, parent.start_page,
                     parent.text[:200], parent.text, "parent", None, parent.end_page, parent.section_path),
                )

    async def _ingest_file(self, path: Path, executor: ThreadPoolExecutor) -> Optional[dict]:
        file_hash = _file_hash(path)
        if self._already_ingested(path, file_hash):
            log.info("[Ingestion] Skipping (unchanged): %s", path.name)
            return None

        log.info("[Ingestion] Processing: %s", path.name)
        t0 = time.time()
        pages, page_count = await self._load_pdf_async(path, executor)
        doc_id = str(uuid.uuid4())
        parents, children = self.chunker.chunk(pages, doc_id)
        if not parents or not children:
            log.warning("[Ingestion] No hierarchical chunks produced from %s", path.name)
            return None

        self._remove_old_document(path)
        with self.db.connect() as conn:
            conn.execute(
                """INSERT INTO documents
                   (id, filename, filepath, file_hash, page_count, chunk_count, ingested_at, metadata_json)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (doc_id, path.name, str(path), file_hash, page_count, len(children), time.time(),
                 json.dumps({"source": str(path), "chunking": "hierarchical-semantic"})),
            )

        for parent in parents:
            parent.metadata["source"] = str(path)
        self._store_parents(parents, doc_id, str(path))
        for child in children:
            child.metadata["source"] = str(path)

        batch_size = self.cfg.embed_batch_size
        for i in range(0, len(children), batch_size):
            batch = children[i:i + batch_size]
            await self._embed_batch(batch, doc_id, executor)

        elapsed = time.time() - t0
        log.info("[Ingestion] ✓ %s: %d pages, %d parents, %d children in %.1fs",
                 path.name, page_count, len(parents), len(children), elapsed)
        return {
            "filename": path.name, "pages": page_count, "parents": len(parents),
            "children": len(children), "chunks": len(children), "elapsed_s": round(elapsed, 1),
        }

    async def run(self) -> list[dict]:
        docs_dir = Path(self.cfg.docs_dir)
        pdfs = list(docs_dir.glob("**/*.pdf"))
        if not pdfs:
            log.warning("[Ingestion] No PDFs found in %s", docs_dir)
            return []

        summaries: list[dict] = []
        with ThreadPoolExecutor(max_workers=self.cfg.ingest_workers) as executor:
            tasks = [self._ingest_file(p, executor) for p in pdfs]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                log.error("[Ingestion] Error: %s", result)
            elif result is not None:
                summaries.append(result)
        log.info("[Ingestion] Complete: %d new files ingested, %d skipped (unchanged)",
                 len(summaries), len(pdfs) - len(summaries))
        return summaries
