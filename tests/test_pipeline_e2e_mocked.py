"""End-to-end orchestration tests with model/network boundaries mocked.

These tests execute ProductionRAGPipeline.query() through routing, cache,
answerability, generation, critic, triad, provenance, and metrics without
requiring Ollama or live web access.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from config import RAGConfig
from critic import CANONICAL_INSUFFICIENT_INFO_RESPONSE
from pipeline import ProductionRAGPipeline


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[1.0, 0.5, 0.25] for _ in texts]


class FakeCache:
    def __init__(self, chunks=None, answer_cache=None):
        self.chunks = chunks or []
        self.answer_cache = answer_cache
        self.saved_answers = []

    def get_answer(self, *args, **kwargs):
        return self.answer_cache

    def get_retrieval(self, key):
        return list(self.chunks)

    def set_retrieval(self, key, chunks):
        self.chunks = list(chunks)

    def set_answer(self, *args, **kwargs):
        self.saved_answers.append((args, kwargs))


class FakeMetrics:
    def __init__(self):
        self.traces = []

    def record(self, trace):
        self.traces.append(trace)

    def check_drift(self):
        return None


class FakeRewriter:
    def __init__(self):
        self.scores = []
        self.feedback = []

    def rewrite_queries(self, question, chat_history=None):
        return 1, [question]

    def record_answer_score(self, rewrite_id, score):
        self.scores.append((rewrite_id, score))

    def record_feedback(self, rewrite_id, helpful):
        self.feedback.append((rewrite_id, helpful))


class FakeMemory:
    def history(self, conversation_id):
        return []

    def add_message(self, *args, **kwargs):
        return None


class FakeChain:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        return self.answer


class FakeCritic:
    def __init__(self, fail_once=False):
        self.last_details = {}
        self.fail_once = fail_once
        self.calls = 0

    def evaluate(self, question, answer, context):
        self.calls += 1
        if self.fail_once and self.calls == 1:
            self.last_details = {
                "groundedness": "PASS",
                "answer_relevance": "PASS",
                "completeness": "FAIL",
                "relevance": "PASS",
            }
            return {
                "verdict": "HALLUCINATED",
                "severity": "minor",
                "critique": "one practice problem is incomplete",
                "unsupported_claims": [],
            }
        self.last_details = {
            "groundedness": "PASS",
            "answer_relevance": "PASS",
            "completeness": "PASS",
            "relevance": "PASS",
        }
        return {
            "verdict": "PASS",
            "severity": "none",
            "critique": "",
            "unsupported_claims": [],
        }

    def repair_constrained(self, question, context, answer, critique):
        return "1. A complete, supported practice problem about fluid pressure."

    def repair_destructive(self, question, context, answer, critique):
        return None

    def polish(self, answer):
        return answer


class FakeRetriever:
    pass


def strong_fluid_chunks(image_path=None):
    first = {
        "chroma_id": "chunk-1",
        "source_type": "pdf",
        "filename": "/docs/fluids.pdf",
        "page_number": 1,
        "section_path": "Fluid pressure",
        "text": "Fluid pressure and buoyancy are core fluid mechanics concepts.",
        "rerank_score": 3.0,
    }
    if image_path is not None:
        first["image_path"] = str(image_path)
        first["has_table"] = False
    return [
        first,
        {
            "chroma_id": "chunk-2",
            "source_type": "pdf",
            "filename": "/docs/fluids.pdf",
            "page_number": 2,
            "section_path": "Flow",
            "text": "Fluids can be analyzed with pressure, continuity, and flow relationships.",
            "rerank_score": 2.0,
        },
    ]


def build_pipeline(chunks, *, rag_answer="text answer", practice_answer="practice answer",
                   answer_cache=None, critic=None):
    cfg = RAGConfig()
    cfg.debug_checkpoints = False
    cfg.critic_enabled = True
    cfg.critic_on_low_confidence_only = True
    cfg.critic_on_generation_requests = True
    cfg.critic_polish_enabled = False
    cfg.always_scrape_web = False

    pipeline = ProductionRAGPipeline.__new__(ProductionRAGPipeline)
    pipeline.cfg = cfg
    pipeline.embeddings = FakeEmbeddings()
    pipeline.cache = FakeCache(chunks, answer_cache)
    pipeline.metrics = FakeMetrics()
    pipeline.rewriter = FakeRewriter()
    pipeline.memory = FakeMemory()
    pipeline.critic = critic or FakeCritic()
    pipeline.retriever = FakeRetriever()
    pipeline.reranker = object()
    pipeline.web_store = None
    pipeline._rag_chain = FakeChain(rag_answer)
    pipeline._practice_chain = FakeChain(practice_answer)
    pipeline.vision_llm = None
    pipeline._query_count = 0
    pipeline._last_multimodal_usage = {
        "generation_mode": "not_run",
        "images_attached": 0,
        "tables_attached": 0,
    }
    pipeline._last_adaptive_web_usage = {
        "enabled": True,
        "rounds": 0,
        "network_sources": 0,
        "cache_chunks": 0,
        "stop_reason": "not_run",
        "elapsed_ms": 0.0,
    }
    pipeline._last_web_scrape_stats = {
        "query": "",
        "requested_urls": 0,
        "fetched_urls": [],
        "used_urls": [],
        "cache_hits": 0,
        "new_chunks": 0,
    }
    return pipeline


@pytest.mark.e2e
def test_explanation_query_runs_full_orchestration_without_unneeded_critic():
    pipeline = build_pipeline(
        strong_fluid_chunks(),
        rag_answer="Liquids change volume very little under ordinary pressure.",
    )

    result = pipeline.query("Explain fluid pressure")

    assert result["answer"].startswith("Liquids")
    assert result["metrics"]["query_type"] == "explanation"
    assert result["metrics"]["answerable"] is True
    assert result["metrics"]["critic_ran"] is False
    assert result["metrics"]["faithfulness"] is None
    assert result["metrics"]["triad_pass"] is None
    assert result["multimodal_usage"]["generation_mode"] == "text_only"


@pytest.mark.e2e
def test_practice_query_uses_subject_retrieval_practice_chain_and_critic():
    pipeline = build_pipeline(
        strong_fluid_chunks(),
        practice_answer="1. A tank contains water. Determine the pressure at a stated depth.",
    )

    result = pipeline.query("give me practice problems for fluids physics 2c")

    assert result["metrics"]["query_type"] == "practice"
    assert result["metrics"]["generation_request"] is True
    assert result["metrics"]["evidence_query"] == "fluids"
    assert result["metrics"]["rerank_query"] == "fluids"
    assert result["metrics"]["critic_ran"] is True
    assert result["metrics"]["triad_pass"] is True
    assert pipeline._practice_chain.calls
    assert not pipeline._rag_chain.calls
    assert pipeline.cache.saved_answers, "validated generation should be answer-cached"


@pytest.mark.e2e
def test_practice_critic_can_repair_incomplete_generated_artifact():
    critic = FakeCritic(fail_once=True)
    pipeline = build_pipeline(
        strong_fluid_chunks(),
        practice_answer="1. Find pressure without enough givens.",
        critic=critic,
    )

    result = pipeline.query("make me practice problems for fluid pressure")

    assert result["answer"].startswith("1. A complete")
    assert result["critic"]["repair_attempts"] == 1
    assert result["critic"]["repair_mode"] == "constrained"
    assert result["critic"]["validated"] is True
    assert critic.calls == 2


@pytest.mark.e2e
def test_weak_retrieval_abstains_without_calling_generation():
    weak = [
        {
            "chroma_id": "weak-1",
            "source_type": "pdf",
            "filename": "/docs/random.pdf",
            "page_number": 1,
            "text": "unrelated cooking recipe",
            "rerank_score": -9.0,
        },
        {
            "chroma_id": "weak-2",
            "source_type": "pdf",
            "filename": "/docs/random.pdf",
            "page_number": 2,
            "text": "more unrelated material",
            "rerank_score": -9.5,
        },
    ]
    pipeline = build_pipeline(weak)

    result = pipeline.query(
        "Why are liquids incompressible?",
        use_web_fallback=False,
    )

    assert result["answer"] == CANONICAL_INSUFFICIENT_INFO_RESPONSE
    assert result["metrics"]["answerable"] is False
    assert result["metrics"]["low_confidence"] is True
    assert not pipeline._rag_chain.calls


@pytest.mark.e2e
def test_validated_answer_cache_short_circuits_retrieval_and_generation():
    cached_metadata = {
        "validated": True,
        "rag_triad": {
            "answer_relevance": 1.0,
            "context_relevance": 1.0,
            "faithfulness": 1.0,
            "passed": True,
            "triage_actions": [],
        },
    }
    cached_sources = [{
        "source_type": "web",
        "url": "https://example.com/fluids",
        "title": "Fluids",
        "has_image": False,
        "has_table": False,
    }]
    pipeline = build_pipeline(
        [],
        answer_cache=("cached validated answer", cached_sources, cached_metadata),
    )

    result = pipeline.query("Explain fluid pressure")

    assert result["from_cache"] is True
    assert result["answer"] == "cached validated answer"
    assert result["metrics"]["answer_cache_hit"] is True
    assert result["multimodal_usage"]["generation_mode"] == "answer_cache"
    assert not pipeline._rag_chain.calls


@pytest.mark.e2e
@pytest.mark.multimodal
def test_query_uses_visual_evidence_when_available(tmp_path):
    image = tmp_path / "pressure.png"
    image.write_bytes(b"image bytes")
    pipeline = build_pipeline(strong_fluid_chunks(image))
    pipeline.vision_llm = SimpleNamespace(
        invoke=lambda messages: SimpleNamespace(content="The diagram shows fluid pressure.")
    )

    result = pipeline.query("Explain fluid pressure diagram")

    assert result["answer"] == "The diagram shows fluid pressure."
    assert result["multimodal_usage"]["generation_mode"] == "vision"
    assert result["multimodal_usage"]["images_attached"] == 1
    assert result["multimodal_usage"]["used_image"] is True
