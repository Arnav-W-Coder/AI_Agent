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
    max_scrape_urls:   int  = 5
    ddg_retries:       int  = 3
    min_domain_score:  int  = 30
    web_top_k:         int  = 4     # web chunks kept after temp-store similarity search
    always_scrape_web: bool = True  # True → parallel PDF+web always; False → fallback only

    # ── Critic thresholds ─────────────────────────────────────────────────────
    # All formerly hardcoded — change here without touching critic.py or pipeline.py
    critic_uncertainty_threshold: float = 0.50
    # Ratio of hedge sentences in an answer before it is auto-accepted as
    # "I don't know". Lower = stricter (more answers sent to LLM critic).
    # 0.50 means half the sentences must be hedges; 0.33 means one in three.

    critic_claim_penalty: float = 0.20
    # Faithfulness score deducted per unsupported claim found by the critic.
    # score = max(0, 1.0 - penalty × num_claims). Lower = more lenient.
    # 0.20 → 1 claim = 0.80, 2 claims = 0.60, 5+ claims = 0.0.

    # ── Rewriter auto-labeling ────────────────────────────────────────────────
    rewriter_helpful_min_score:    float = 0.80
    # Faithfulness score >= this → rewrite is auto-labeled helpful (enters few-shot pool).
    rewriter_unhelpful_max_score:  float = 0.40
    # Faithfulness score <  this → rewrite is auto-labeled unhelpful (excluded from pool).
    # Scores between the two thresholds receive no auto-label (neutral; awaits user rating).