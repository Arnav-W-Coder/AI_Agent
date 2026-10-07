"""
config.py — Single source of truth for all tunable parameters.
Change values here; nothing else needs editing for basic tuning.
"""
from dataclasses import dataclass, field
from pathlib import Path


# This dataclass is the "control panel" for the entire RAG system.
# Most runtime behavior is derived from these defaults: which model to use,
# where to persist data, how many chunks to retrieve, and when to trigger
# web fallback or critic-based repair.
@dataclass
class RAGConfig:
    # Models
    embed_model: str = "nomic-embed-text"
    llm_model: str = "llama3.2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    critic_model: str | None = None
    rewriter_model: str | None = None

    # Deterministic generation/evaluation defaults. Keep these at 0 for
    # reproducible RAG outputs; tune explicitly only when variability is desired.
    llm_temperature: float = 0.0
    rewriter_temperature: float = 0.0
    critic_temperature: float = 0.0
    vlm_temperature: float = 0.0

    # Paths
    docs_dir: Path = field(default_factory=lambda: Path("./docs"))
    chroma_dir: Path = field(default_factory=lambda: Path("./chroma_db"))
    db_path: Path = field(default_factory=lambda: Path("./rag.db"))
    image_store_dir: Path = field(default_factory=lambda: Path("./image_store"))

    # LLM
    ctx_window: int = 16384
    max_answer_chars: int = 7000
    max_history_messages: int = 12

    # Multimodal ingestion and generation
    multimodal_enabled: bool = True
    vlm_model: str = "llama3.2-vision"
    image_render_dpi: int = 144
    image_min_drawing_count: int = 12
    image_max_per_doc: int = 24
    image_top_k: int = 4
    vlm_ingest_workers: int = 2
    vlm_generation_enabled: bool = True

    # Retrieval
    # Keep the initial candidate pool broad enough for recall, but avoid
    # expensive 40+40 candidate fan-out on every simple question.
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_rerank: int = 8
    rrf_k: int = 60
    min_rerank_score: float = -7.5
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
    answer_cache_schema_version: int = 3
    retrieval_cache_schema_version: int = 5
    return_validated_answer_cache: bool = True

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
    # max_scrape_urls is a hard per-query ceiling. Adaptive web retrieval starts
    # smaller and only spends more network requests when evidence is still weak.
    max_scrape_urls: int = 6
    ddg_retries: int = 2
    min_domain_score: int = 55
    web_top_k: int = 6
    always_scrape_web: bool = False
    adaptive_web_enabled: bool = True
    adaptive_web_initial_sources: int = 2
    adaptive_web_sources_per_round: int = 2
    adaptive_web_max_rounds: int = 2
    web_chroma_dir: Path = field(default_factory=lambda: Path("./chroma_web"))
    web_chunk_ttl_hours: int = 24
    web_collection_max_chunks: int = 8000
    web_fetch_workers: int = 4
    web_request_timeout_seconds: int = 8
    web_min_text_chars: int = 400
    web_min_page_quality: float = 0.45
    web_min_query_relevance: float = 0.20
    web_min_candidates: int = 2
    web_authoritative_domains: tuple[str, ...] = (
        "cppreference.com", "cplusplus.com", "learn.microsoft.com",
        "docs.python.org", "developer.mozilla.org", "docs.oracle.com",
        "kernel.org", "llvm.org", "gnu.org", "iso.org", "arxiv.org",
        "github.com", "huggingface.co", "langchain.com", "ollama.com",
    )
    url_evaluator_enabled: bool = True
    url_evaluator_min_domain_score: int = 55
    url_evaluator_max_redirects: int = 5

    # Critic
    critic_enabled: bool = True
    critic_on_low_confidence_only: bool = True
    critic_uncertainty_threshold: float = 0.50
    critic_claim_penalty: float = 0.20
    critic_polish_enabled: bool = False
    constrained_critic_repair: bool = True
    critic_max_repair_attempts: int = 1
    critic_abstain_on_failed_repair: bool = False
    critic_require_context_grounding: bool = True
    critic_config_version: int = 4

    # RAG triad diagnostics (Answer Relevance / Context Relevance / Faithfulness)
    triad_enabled: bool = True
    triad_answer_relevance_min: float = 1.0
    triad_context_relevance_min: float = 1.0
    triad_faithfulness_min: float = 1.0
    triad_config_version: int = 1

    # Rewriter
    rewrite_enabled: bool = True
    multi_query_max_queries: int = 4
    # Clear standalone explanation/definition queries skip the rewrite LLM.
    # Ambiguous, comparison and multi-constraint queries still expand.
    rewrite_only_when_ambiguous: bool = True
    rewriter_helpful_min_score: float = 0.80
    rewriter_unhelpful_max_score: float = 0.40

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 75
    semantic_chunking_enabled: bool = True
    semantic_breakpoint_percentile: float = 90.0
    semantic_min_distance: float = 0.0
    semantic_min_block_tokens: int = 80
    parent_target_tokens: int = 600
    parent_max_tokens: int = 900
    child_max_tokens: int = 220
    child_overlap_tokens: int = 40
    context_neighbor_count: int = 1
    context_budget_tokens: int = 5000

    # Query routing / confidence
    query_routing_enabled: bool = True
    recommendation_query_expansion_enabled: bool = True
    # MS MARCO cross-encoder outputs are ranking logits, not calibrated
    # probabilities. By default, answerability uses the reranker's own floor
    # rather than assuming relevant scores must be >= 0.
    answerability_use_absolute_rerank_thresholds: bool = False
    answerability_min_top_score: float = 0.0
    answerability_min_mean_score: float = -0.5
    answerability_min_chunks: int = 2
    answerability_score_window: int = 5
    answerability_min_query_term_coverage: float = 0.50
    # A strong cross-encoder signal may override incomplete literal term coverage.
    # This prevents semantically relevant evidence (e.g. "gas" vs "air") from
    # being rejected solely because wording differs.
    answerability_semantic_override_enabled: bool = True
    answerability_semantic_override_top_score: float = 1.0
    answerability_semantic_override_mean_score: float = 0.0
    answerability_semantic_override_min_coverage: float = 0.20
    comparison_require_all_options: bool = True
    low_confidence_requires_web: bool = True
    destructive_critic_repair: bool = False
