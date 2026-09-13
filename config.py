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

    # Retrieval
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_rerank: int = 5
    rrf_k: int = 60
    min_rerank_score: float = -8.0
    min_mean_rerank_score: float = 0.0
    min_top_rerank_score: float = 0.0
    retrieval_quality_margin: float = 0.0
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
    min_domain_score: int = 65
    web_top_k: int = 6
    always_scrape_web: bool = False
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
    url_evaluator_enabled: bool = True
    url_evaluator_min_domain_score: int = 65
    url_evaluator_max_redirects: int = 5

    # Critic
    critic_enabled: bool = True
    critic_on_low_confidence_only: bool = True
    critic_uncertainty_threshold: float = 0.50
    critic_claim_penalty: float = 0.20
    critic_polish_enabled: bool = False

    # Rewriter
    rewrite_enabled: bool = True
    rewrite_only_when_ambiguous: bool = True
    rewriter_helpful_min_score: float = 0.80
    rewriter_unhelpful_max_score: float = 0.40
