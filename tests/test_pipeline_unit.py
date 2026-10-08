"""Fast deterministic unit tests for routing, answerability and quality gates."""
import pytest

from config import RAGConfig
from pipeline import ProductionRAGPipeline, _normalize_url


@pytest.fixture
def cfg():
    config = RAGConfig()
    config.debug_checkpoints = False
    return config


def bare_pipeline(cfg):
    pipeline = ProductionRAGPipeline.__new__(ProductionRAGPipeline)
    pipeline.cfg = cfg
    return pipeline


@pytest.mark.layer1
@pytest.mark.parametrize("query", [
    "give me practice problems for fluids physics 2c",
    "give me some practice test problems for my physics fluids final",
    "make me a mock exam for fluid mechanics",
    "create final review questions for electrostatics",
    "prepare a worksheet on Bernoulli's equation",
    "make flashcards for hydrostatic pressure",
])
def test_practice_variants_route_consistently(query):
    assert ProductionRAGPipeline._is_practice_request(query)
    assert ProductionRAGPipeline._classify_query(query) == "practice"
    assert ProductionRAGPipeline._is_generation_request(query)


@pytest.mark.layer1
@pytest.mark.parametrize("query", [
    "give me information about fluids",
    "why are liquids incompressible?",
    "what is Bernoulli's equation?",
])
def test_non_artifact_queries_are_not_false_generation_requests(query):
    assert not ProductionRAGPipeline._is_generation_request(query)


@pytest.mark.layer1
def test_study_subject_cleanup_removes_instruction_and_exam_noise():
    assert ProductionRAGPipeline._generation_retrieval_query(
        "give me some practice test problems for my physics fluids final"
    ) == "physics fluids"
    assert ProductionRAGPipeline._generation_retrieval_query(
        "give me a quiz on electrostatics"
    ) == "electrostatics"


@pytest.mark.layer1
def test_evidence_terms_remove_query_glue_and_singularize():
    terms = ProductionRAGPipeline._query_evidence_terms(
        "how are liquids incompressible but air isn't"
    )
    assert {"liquid", "incompressible", "air"}.issubset(terms)
    assert "liquids" not in terms
    assert "but" not in terms
    assert "isn" not in terms


@pytest.mark.layer1
def test_model_floor_accepts_negative_ms_marco_logits(cfg):
    cfg.answerability_use_absolute_rerank_thresholds = False
    cfg.min_rerank_score = -7.5
    pipeline = bare_pipeline(cfg)
    passed, top, mean = pipeline._rerank_confidence([
        {"rerank_score": -2.0},
        {"rerank_score": -5.0},
        {"rerank_score": -6.0},
    ])
    assert passed
    assert top == -2.0
    assert mean < 0


@pytest.mark.layer1
def test_absolute_threshold_mode_can_reject_same_negative_logits(cfg):
    cfg.answerability_use_absolute_rerank_thresholds = True
    cfg.answerability_min_top_score = 0.0
    cfg.answerability_min_mean_score = -0.5
    pipeline = bare_pipeline(cfg)
    passed, _, _ = pipeline._rerank_confidence([
        {"rerank_score": -2.0},
        {"rerank_score": -5.0},
    ])
    assert not passed


@pytest.mark.layer1
def test_context_only_chunk_cannot_make_weak_evidence_answerable(cfg):
    pipeline = bare_pipeline(cfg)
    chunks = [
        {"text": "unrelated text", "rerank_score": 3.0},
        {
            "text": "liquid incompressible pressure volume",
            "rerank_score": 10.0,
            "context_only": True,
        },
    ]
    assert pipeline._is_answerable(chunks, "why are liquids incompressible") is False


@pytest.mark.layer1
def test_semantic_override_requires_some_lexical_overlap(cfg):
    pipeline = bare_pipeline(cfg)
    chunks = [
        {"text": "completely unrelated cooking recipe", "rerank_score": 8.0},
        {"text": "kitchen ingredients and oven temperature", "rerank_score": 7.0},
    ]
    assert not pipeline._is_answerable(chunks, "why are liquids incompressible")


@pytest.mark.layer1
def test_source_diversification_caps_duplicate_web_source():
    chunks = [
        {
            "source_type": "web",
            "source_url": "https://same.example/page",
            "text": f"duplicate {i}",
            "rerank_score": 10.0 - i,
        }
        for i in range(5)
    ] + [
        {
            "source_type": "web",
            "source_url": "https://other.example/page",
            "text": "independent evidence",
            "rerank_score": 4.5,
        }
    ]
    selected = ProductionRAGPipeline._diversify_sources(
        chunks, "explanation", limit=6, max_per_source=2
    )
    assert sum(c.get("source_url") == "https://same.example/page" for c in selected) == 2
    assert any(c.get("source_url") == "https://other.example/page" for c in selected)


@pytest.mark.layer1
def test_generation_requests_always_run_critic(cfg):
    cfg.critic_enabled = True
    cfg.critic_on_low_confidence_only = True
    cfg.critic_on_generation_requests = True
    pipeline = bare_pipeline(cfg)
    assert not pipeline._should_run_critic(low_confidence=False, answerable=True)
    assert pipeline._should_run_critic(
        low_confidence=False, answerable=True, generation_request=True
    )


@pytest.mark.layer1
def test_sanitize_answer_removes_hidden_reasoning_only():
    text = "<think>private notes</think>\nFinal Answer: Supported answer."
    assert ProductionRAGPipeline._sanitize_answer(text) == "Supported answer."


@pytest.mark.layer1
def test_page_quality_rewards_relevant_meaningful_page():
    good, good_rel = ProductionRAGPipeline._page_quality(
        "fluid pressure buoyancy bernoulli " * 120,
        "Fluid mechanics",
        "fluid pressure buoyancy",
        80,
        8,
    )
    weak, weak_rel = ProductionRAGPipeline._page_quality(
        "cookie banner login subscribe",
        "Home",
        "fluid pressure buoyancy",
        80,
        1,
    )
    assert good > weak
    assert good_rel > weak_rel


@pytest.mark.layer1
@pytest.mark.parametrize("url", [
    "ftp://example.com/file",
    "http://user:pass@example.com/private",
    "http://localhost/admin",
    "http://metadata.google.internal/latest",
])
def test_normalize_url_rejects_unsafe_or_unsupported_urls(url):
    assert _normalize_url(url) is None
