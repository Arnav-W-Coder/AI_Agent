"""
ingestion.py — Async, batched PDF ingestion pipeline.

Design:
  - Offline and async: scans ./docs, skips already-ingested files (via MD5 hash)
  - Loads PDFs in parallel using ThreadPoolExecutor (IO + CPU bound)
  - Embeds in batches of embed_batch_size to avoid OOM
  - Stores metadata in SQLite alongside vectors in ChromaDB
  - Builds/rebuilds the BM25 index after every ingestion run
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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import RAGConfig
from db import Database

log = logging.getLogger(__name__)


def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_pdf(path: Path) -> tuple[list[Document], int]:
    """Load a single PDF; returns (pages, page_count)."""
    loader = PyPDFLoader(str(path))
    pages  = loader.load()
    return pages, len(pages)


class AsyncIngestionPipeline:
    """
    Async ingestion pipeline. Call `run()` to ingest all PDFs in docs_dir.
    Skips files whose MD5 hash hasn't changed since last ingestion.
    """

    def __init__(
        self,
        db: Database,
        cfg: RAGConfig,
        vectorstore: Chroma,
        embeddings: OllamaEmbeddings,
    ) -> None:
        self.db          = db
        self.cfg         = cfg
        self.vectorstore = vectorstore
        self.embeddings  = embeddings
        self.splitter    = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )

    def _already_ingested(self, filepath: Path, file_hash: str) -> bool:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT file_hash FROM documents WHERE filepath = ?",
                (str(filepath),)
            ).fetchone()
        return row is not None and row["file_hash"] == file_hash

    async def _load_pdf_async(
        self, path: Path, executor: ThreadPoolExecutor
    ) -> tuple[list[Document], int]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, _load_pdf, path)

    async def _embed_batch(
        self, chunks: list[Document], doc_id: str, executor: ThreadPoolExecutor
    ) -> None:
        """Embed one batch of chunks and upsert into ChromaDB + SQLite."""
        loop   = asyncio.get_event_loop()
        texts  = [c.page_content for c in chunks]

        # Embed via Ollama (blocking HTTP call → offload to thread)
        embeddings_list = await loop.run_in_executor(
            executor, self.embeddings.embed_documents, texts
        )

        # Build Chroma documents with metadata
        chroma_ids  = [str(uuid.uuid4()) for _ in chunks]
        metadatas   = []
        for i, chunk in enumerate(chunks):
            meta = dict(chunk.metadata)
            meta["doc_id"] = doc_id
            metadatas.append(meta)

        # Upsert into ChromaDB
        self.vectorstore._collection.upsert(
            ids=chroma_ids,
            documents=texts,
            embeddings=embeddings_list,
            metadatas=metadatas,
        )

        # Store chunk metadata in SQLite
        with self.db.connect() as conn:
            for i, (chroma_id, chunk) in enumerate(zip(chroma_ids, chunks)):
                chunk_id = str(uuid.uuid4())
                conn.execute(
                    """INSERT OR IGNORE INTO chunks
                       (id, doc_id, chroma_id, chunk_index, page_number, text_preview)
                       VALUES (?,?,?,?,?,?)""",
                    (
                        chunk_id, doc_id, chroma_id, i,
                        chunk.metadata.get("page", 0),
                        chunk.page_content[:200],
                    )
                )

    async def _ingest_file(
        self, path: Path, executor: ThreadPoolExecutor
    ) -> Optional[dict]:
        """Ingest a single PDF file. Returns summary dict or None if skipped."""
        file_hash = _file_hash(path)

        if self._already_ingested(path, file_hash):
            log.info(f"[Ingestion] Skipping (unchanged): {path.name}")
            return None

        log.info(f"[Ingestion] Processing: {path.name}")
        t0 = time.time()

        pages, page_count = await self._load_pdf_async(path, executor)
        chunks = self.splitter.split_documents(pages)

        if not chunks:
            log.warning(f"[Ingestion] No chunks produced from {path.name}")
            return None

        doc_id = str(uuid.uuid4())

        # Register document in SQLite before embedding (so FK refs work)
        with self.db.connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO documents
                   (id, filename, filepath, file_hash, page_count,
                    chunk_count, ingested_at, metadata_json)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    doc_id, path.name, str(path), file_hash,
                    page_count, len(chunks), time.time(),
                    json.dumps({"source": str(path)}),
                )
            )

        # Embed in batches
        batch_size = self.cfg.embed_batch_size
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            log.info(
                f"[Ingestion] {path.name} — embedding batch "
                f"{i//batch_size + 1}/{-(-len(chunks)//batch_size)}"
            )
            await self._embed_batch(batch, doc_id, executor)

        elapsed = time.time() - t0
        log.info(
            f"[Ingestion] ✓ {path.name}: {page_count} pages, "
            f"{len(chunks)} chunks in {elapsed:.1f}s"
        )
        return {
            "filename":    path.name,
            "pages":       page_count,
            "chunks":      len(chunks),
            "elapsed_s":   round(elapsed, 1),
        }

    async def run(self) -> list[dict]:
        """
        Scan docs_dir, ingest all new/changed PDFs concurrently.
        Returns list of ingestion summaries.
        """
        docs_dir = Path(self.cfg.docs_dir)
        pdfs     = list(docs_dir.glob("**/*.pdf"))

        if not pdfs:
            log.warning(f"[Ingestion] No PDFs found in {docs_dir}")
            return []

        log.info(f"[Ingestion] Found {len(pdfs)} PDFs in {docs_dir}")
        summaries = []

        with ThreadPoolExecutor(max_workers=self.cfg.ingest_workers) as executor:
            tasks = [self._ingest_file(p, executor) for p in pdfs]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                log.error(f"[Ingestion] Error: {r}")
            elif r is not None:
                summaries.append(r)

        log.info(
            f"[Ingestion] Complete: {len(summaries)} new files ingested, "
            f"{len(pdfs) - len(summaries)} skipped (unchanged)"
        )
        return summaries