"""Embedding and vector-store contract tests.

All default tests are deterministic and do not require Ollama. The optional
live_ollama test can be enabled with RUN_OLLAMA_TESTS=1.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chunking import ChunkRecord
from config import RAGConfig
from db import Database
from ingestion import AsyncIngestionPipeline
from web_store import WebChunkStore


class FakeCollection:
    def __init__(self, stored_vector=None):
        self.stored_vector = stored_vector
        self.upserts = []
        self._count = 0

    def peek(self, limit=1):
        if self.stored_vector is None:
            return {"embeddings": []}
        return {"embeddings": [self.stored_vector]}

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)
        self._count += len(kwargs.get("ids", []))

    def count(self):
        return self._count

    def delete(self, ids):
        return None


class FakeVectorStore:
    def __init__(self, collection):
        self._collection = collection


class FakeEmbeddings:
    def __init__(self, dim=3):
        self.dim = dim
        self.calls = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return [
            [float((i + 1) * (j + 1)) for j in range(self.dim)]
            for i, _ in enumerate(texts)
        ]


def cfg_for_tests(tmp_path):
    cfg = RAGConfig()
    cfg.db_path = tmp_path / "test.db"
    cfg.chroma_dir = tmp_path / "chroma"
    cfg.web_chroma_dir = tmp_path / "web_chroma"
    cfg.debug_checkpoints = False
    cfg.web_embed_batch_size = 1
    return cfg


def insert_doc(db, doc_id="doc-1"):
    with db.connect() as conn:
        conn.execute(
            """INSERT INTO documents
               (id, filename, filepath, file_hash, page_count, chunk_count, ingested_at, metadata_json)
               VALUES (?,?,?,?,?,?,?,?)""",
            (doc_id, "sample.pdf", "/tmp/sample.pdf", "hash", 1, 1, time.time(), "{}"),
        )


@pytest.mark.layer1
@pytest.mark.embedding
def test_ingestion_embedding_contract_accepts_matching_dimensions(tmp_path):
    cfg = cfg_for_tests(tmp_path)
    pipeline = AsyncIngestionPipeline.__new__(AsyncIngestionPipeline)
    pipeline.cfg = cfg
    pipeline.embeddings = FakeEmbeddings(dim=3)
    pipeline.vectorstore = FakeVectorStore(FakeCollection(stored_vector=[0.0, 0.0, 0.0]))

    pipeline._validate_embedding_contract()


@pytest.mark.layer1
@pytest.mark.embedding
def test_ingestion_embedding_contract_rejects_document_query_mismatch(tmp_path):
    cfg = cfg_for_tests(tmp_path)

    class MismatchedEmbeddings:
        def embed_documents(self, texts):
            if texts == ["__rag_query_dimension_probe__"]:
                return [[1.0, 2.0, 3.0, 4.0]]
            return [[1.0, 2.0, 3.0]]

    pipeline = AsyncIngestionPipeline.__new__(AsyncIngestionPipeline)
    pipeline.cfg = cfg
    pipeline.embeddings = MismatchedEmbeddings()
    pipeline.vectorstore = FakeVectorStore(FakeCollection())

    with pytest.raises(RuntimeError, match="Embedding dimension mismatch"):
        pipeline._validate_embedding_contract()


@pytest.mark.layer1
@pytest.mark.embedding
def test_ingestion_embedding_contract_rejects_existing_chroma_dimension(tmp_path):
    cfg = cfg_for_tests(tmp_path)
    pipeline = AsyncIngestionPipeline.__new__(AsyncIngestionPipeline)
    pipeline.cfg = cfg
    pipeline.embeddings = FakeEmbeddings(dim=3)
    pipeline.vectorstore = FakeVectorStore(
        FakeCollection(stored_vector=[0.0, 0.0, 0.0, 0.0])
    )

    with pytest.raises(RuntimeError, match="Chroma collection dimension mismatch"):
        pipeline._validate_embedding_contract()


@pytest.mark.layer1
@pytest.mark.embedding
def test_ingestion_embedding_contract_rejects_empty_vector(tmp_path):
    cfg = cfg_for_tests(tmp_path)

    class EmptyEmbeddings:
        def embed_documents(self, texts):
            return [[]]

    pipeline = AsyncIngestionPipeline.__new__(AsyncIngestionPipeline)
    pipeline.cfg = cfg
    pipeline.embeddings = EmptyEmbeddings()
    pipeline.vectorstore = FakeVectorStore(FakeCollection())

    with pytest.raises(RuntimeError, match="empty vector"):
        pipeline._validate_embedding_contract()


@pytest.mark.asyncio
@pytest.mark.layer1
@pytest.mark.embedding
async def test_embed_batch_persists_matching_vectors_and_metadata(tmp_path):
    cfg = cfg_for_tests(tmp_path)
    db = Database(cfg.db_path)
    insert_doc(db)
    collection = FakeCollection()

    pipeline = AsyncIngestionPipeline.__new__(AsyncIngestionPipeline)
    pipeline.cfg = cfg
    pipeline.db = db
    pipeline.embeddings = FakeEmbeddings(dim=3)
    pipeline.vectorstore = FakeVectorStore(collection)

    chunks = [
        ChunkRecord(
            id="child-1",
            text="fluid pressure increases with depth",
            chunk_type="child",
            chunk_index=0,
            parent_id="parent-1",
            start_page=1,
            end_page=1,
            section_path="Fluids > Pressure",
            metadata={"source": "/tmp/sample.pdf"},
        )
    ]

    with ThreadPoolExecutor(max_workers=1) as executor:
        await pipeline._embed_batch(chunks, "doc-1", executor)

    assert len(collection.upserts) == 1
    upsert = collection.upserts[0]
    assert len(upsert["embeddings"]) == 1
    assert len(upsert["embeddings"][0]) == 3
    assert upsert["metadatas"][0]["parent_id"] == "parent-1"

    with db.connect() as conn:
        row = conn.execute(
            "SELECT text, chunk_type, parent_id FROM chunks WHERE id='child-1'"
        ).fetchone()
    assert row["text"] == "fluid pressure increases with depth"
    assert row["chunk_type"] == "child"
    assert row["parent_id"] == "parent-1"


@pytest.mark.asyncio
@pytest.mark.layer1
@pytest.mark.embedding
async def test_embed_batch_rejects_vector_count_mismatch(tmp_path):
    cfg = cfg_for_tests(tmp_path)

    class CountMismatch:
        def embed_documents(self, texts):
            return [[1.0, 2.0, 3.0]]

    pipeline = AsyncIngestionPipeline.__new__(AsyncIngestionPipeline)
    pipeline.cfg = cfg
    pipeline.db = Database(cfg.db_path)
    pipeline.embeddings = CountMismatch()
    pipeline.vectorstore = FakeVectorStore(FakeCollection())

    chunks = [
        ChunkRecord("a", "one", "child", 0, None, 1, 1, "", {}),
        ChunkRecord("b", "two", "child", 1, None, 1, 1, "", {}),
    ]
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(RuntimeError, match="returned 1 vectors for 2 chunks"):
            await pipeline._embed_batch(chunks, "doc-1", executor)


@pytest.mark.asyncio
@pytest.mark.layer1
@pytest.mark.embedding
async def test_embed_batch_rejects_inconsistent_dimensions(tmp_path):
    cfg = cfg_for_tests(tmp_path)

    class Inconsistent:
        def embed_documents(self, texts):
            return [[1.0, 2.0], [1.0, 2.0, 3.0]]

    pipeline = AsyncIngestionPipeline.__new__(AsyncIngestionPipeline)
    pipeline.cfg = cfg
    pipeline.db = Database(cfg.db_path)
    pipeline.embeddings = Inconsistent()
    pipeline.vectorstore = FakeVectorStore(FakeCollection())

    chunks = [
        ChunkRecord("a", "one", "child", 0, None, 1, 1, "", {}),
        ChunkRecord("b", "two", "child", 1, None, 1, 1, "", {}),
    ]
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(RuntimeError, match="inconsistent vector dimensions"):
            await pipeline._embed_batch(chunks, "doc-1", executor)


@pytest.mark.layer1
@pytest.mark.embedding
def test_web_store_embedding_contract_detects_dimension_mismatch(tmp_path):
    cfg = cfg_for_tests(tmp_path)
    store = WebChunkStore.__new__(WebChunkStore)
    store.cfg = cfg
    store.embeddings = FakeEmbeddings(dim=3)
    store._chroma = FakeVectorStore(
        FakeCollection(stored_vector=[0.0, 0.0, 0.0, 0.0])
    )

    with pytest.raises(RuntimeError, match="Web Chroma collection dimension mismatch"):
        store._validate_embedding_contract()


@pytest.mark.live_ollama
@pytest.mark.embedding
def test_live_ollama_embedding_dimensions_are_stable():
    if os.getenv("RUN_OLLAMA_TESTS") != "1":
        pytest.skip("set RUN_OLLAMA_TESTS=1 to enable live Ollama embedding test")

    from langchain_ollama import OllamaEmbeddings

    embeddings = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
    vectors = embeddings.embed_documents([
        "fluid pressure and buoyancy",
        "database transactions and indexes",
    ])
    query = embeddings.embed_documents(["fluid pressure"])[0]

    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1]) == len(query)
    assert len(query) > 0
    assert np.isfinite(np.asarray(vectors)).all()
