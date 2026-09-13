"""
config.py — Single source of truth for all tunable parameters.
Change values here; nothing else needs editing for basic tuning.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RAGConfig:
    # Models
    embed_model: str = "nomic-embed-text"
    llm_model: str = "llama3.2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    critic_model: str | None = None
    rewriter_model: str | None = None

    # Paths
    docs_dir: Path = field(default_factory=lambda: Path("./docs"))
    chroma_dir: Path = field(default_factory=lambda: Path("./chroma_db"))
    db_path: Path = field(default_factory=lambda: Path("./rag.db"))

    # LLM
    ctx_window: int = 16384
    max_answer_chars: int = 5000

    # Hierarchical chunking
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
    chunk_size: int = 800
    chunk_overlap: int = 150

    # Retrieval
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_rerank: int = 5
    rrf_k: int = 60
    min_rerank_score: float = -8.0
    min_mean_rerank_score: float = 0.25
    min_top_rerank_score: float = 0.75
    retrieval_quality_margin: float = 0.20
    low_confidence_pdf_limit: int = 5
    require_retrieval_evidence: bool = True

    # Caching
    answer_ttl: int = 3600
    retrieval_ttl: int = 1800
    answer_sim_threshold: float = 0.92
    retrieval_sim_threshold: float = 0.97
    answer_cache_schema_version: int = 2
    retrieval_cache_schema_version: int = 2

    # Ingestion
    embed_batch_size: int = 16
    ingest_workers: int = 4
    web_embed_batch_size: int = 32

    # Monitoring / drift
    drift_window: int = 50
    drift_threshold: float = 0.12
    min_retrieval_score: float = 0.20
    latency_warn_ms: int = 6000

    # Debug / checkpoints
    debug_checkpoints: bool = True
    checkpoint_preview_chars: int = 160
    checkpoint_sample_items: int = 3

    # Web scraping
    max_scrape_urls: int = 5
    ddg_retries: int = 3
    min_domain_score: int = 30
    web_top_k: int = 6
    always_scrape_web: bool = True
    web_chroma_dir: Path = field(default_factory=lambda: Path("./chroma_web"))
    web_chunk_ttl_hours: int = 24
    web_collection_max_chunks: int = 8000
    web_fetch_workers: int = 5
    web_request_timeout_seconds: int = 12
    web_min_text_chars: int = 400
    web_authoritative_domains: tuple[str, ...] = (
        "cppreference.com", "cplusplus.com", "learn.microsoft.com",
        "docs.python.org", "developer.mozilla.org", "docs.oracle.com",
        "kernel.org", "llvm.org", "gnu.org", "iso.org", "arxiv.org",
    )

    # Critic
    critic_enabled: bool = True
    critic_on_low_confidence_only: bool = True
    critic_uncertainty_threshold: float = 0.50
    critic_claim_penalty: float = 0.20

    # Rewriter
    rewrite_enabled: bool = True
    rewrite_only_when_ambiguous: bool = True
    rewriter_helpful_min_score: float = 0.80
    rewriter_unhelpful_max_score: float = 0.40
