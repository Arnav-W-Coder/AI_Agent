"""
config.py — Single source of truth for all tunable parameters.
Change values here; nothing else needs editing for basic tuning.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RAGConfig:

    # ── Models ────────────────────────────────────────────────────────────────
    embed_model:  str = "nomic-embed-text"          # Ollama embedding model
    llm_model:    str = "llama3.2"                  # Ollama generation model
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # HF cross-encoder

    # ── Paths ─────────────────────────────────────────────────────────────────
    docs_dir:   Path = field(default_factory=lambda: Path("./docs"))
    chroma_dir: Path = field(default_factory=lambda: Path("./chroma_db"))
    db_path:    Path = field(default_factory=lambda: Path("./rag.db"))

    # ── LLM ───────────────────────────────────────────────────────────────────
    ctx_window: int = 16384

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size:    int = 800
    chunk_overlap: int = 150

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k_dense:      int   = 20    # dense ANN candidates before rerank
    top_k_sparse:     int   = 20    # BM25 candidates before rerank
    top_k_rerank:     int   = 5     # final chunks after cross-encoder rerank
    rrf_k:            int   = 60    # RRF constant (60 is the standard)
    min_rerank_score: float = -8.0  # drop chunks scoring below this

    # ── Caching ───────────────────────────────────────────────────────────────
    answer_ttl:              int   = 3600   # seconds before answer cache expires
    retrieval_ttl:           int   = 1800   # seconds before retrieval cache expires
    answer_sim_threshold:    float = 0.92   # cosine sim for semantic answer cache hit
    retrieval_sim_threshold: float = 0.97   # cosine sim for retrieval cache hit

    # ── Ingestion ─────────────────────────────────────────────────────────────
    embed_batch_size: int = 16   # chunks per embedding API call
    ingest_workers:   int = 4    # parallel PDF loaders

    # ── Monitoring / drift ────────────────────────────────────────────────────
    drift_window:        int   = 50    # queries per drift-check window
    drift_threshold:     float = 0.12  # fractional drop in mean score → re-embed
    min_retrieval_score: float = 0.20  # below this flags poor retrieval in logs
    latency_warn_ms:     int   = 6000  # log warning if total latency > this

    # ── Web scraping ──────────────────────────────────────────────────────────
    max_scrape_urls:  int = 5
    ddg_retries:      int = 3
    min_domain_score: int = 30