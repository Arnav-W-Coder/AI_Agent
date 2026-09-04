"""
config.py — Single source of truth for all tunable parameters.
Change values here; nothing else needs editing for basic tuning.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RAGConfig:
    # ── Models ────────────────────────────────────────────────────────────────
    embed_model:  str = "nomic-embed-text"
    llm_model:    str = "llama3.2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Paths ─────────────────────────────────────────────────────────────────
    docs_dir:   Path = field(default_factory=lambda: Path("./docs"))
    chroma_dir: Path = field(default_factory=lambda: Path("./chroma_db"))
    db_path:    Path = field(default_factory=lambda: Path("./rag.db"))

    # ── LLM ───────────────────────────────────────────────────────────────────
    ctx_window: int = 16384

    # ── Hierarchical chunking ─────────────────────────────────────────────────
    # Structure-aware sectioning is followed by semantic grouping, then a hard
    # recursive cap. Parents are context units; children are retrieval units.
    semantic_chunking_enabled: bool = True
    semantic_breakpoint_percentile: float = 75.0
    semantic_min_distance: float = 0.10
    semantic_min_paragraph_tokens: int = 20
    semantic_min_block_tokens: int = 80
    parent_target_tokens: int = 900
    parent_max_tokens: int = 1200
    child_max_tokens: int = 220
    child_overlap_tokens: int = 30
    context_budget_tokens: int = 7000
    context_neighbor_count: int = 1

    # Legacy aliases retained for compatibility with older callers.
    chunk_size: int = 800
    chunk_overlap: int = 150

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k_dense:      int   = 20
    top_k_sparse:     int   = 20
    top_k_rerank:     int   = 5
    rrf_k:            int   = 60
    min_rerank_score: float = -8.0

    # ── Caching ───────────────────────────────────────────────────────────────
    answer_ttl:              int   = 3600
    retrieval_ttl:           int   = 1800
    answer_sim_threshold:    float = 0.92
    retrieval_sim_threshold: float = 0.97

    # ── Ingestion ─────────────────────────────────────────────────────────────
    embed_batch_size: int = 16
    ingest_workers:   int = 4

    # ── Monitoring / drift ────────────────────────────────────────────────────
    drift_window:        int   = 50
    drift_threshold:     float = 0.12
    min_retrieval_score: float = 0.20
    latency_warn_ms:     int   = 6000

    # ── Web scraping ──────────────────────────────────────────────────────────
    max_scrape_urls:   int = 5
    ddg_retries:       int = 3
    min_domain_score:  int = 30
    web_top_k:         int = 6
    always_scrape_web: bool = True
    web_chroma_dir: Path = field(default_factory=lambda: Path("./chroma_web"))
    web_chunk_ttl_hours: int = 24
    web_collection_max_chunks: int = 8000

    # ── Critic thresholds ─────────────────────────────────────────────────────
    critic_uncertainty_threshold: float = 0.50
    critic_claim_penalty: float = 0.20

    # ── Rewriter auto-labeling ────────────────────────────────────────────────
    rewriter_helpful_min_score: float = 0.80
    rewriter_unhelpful_max_score: float = 0.40
