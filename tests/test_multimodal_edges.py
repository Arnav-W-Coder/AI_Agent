"""Additional multimodal edge cases beyond the core attachment tests."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import multimodal
from multimodal import ImageRecord, VisionCaptioner, extract_and_caption, markdown_has_table
from pipeline import ProductionRAGPipeline


def pipeline_stub(image_top_k=2):
    pipeline = ProductionRAGPipeline.__new__(ProductionRAGPipeline)
    pipeline.cfg = SimpleNamespace(
        image_top_k=image_top_k,
        practice_max_problems=6,
    )
    pipeline._last_multimodal_usage = {}
    pipeline._rag_chain = MagicMock()
    pipeline._practice_chain = MagicMock()
    pipeline.vision_llm = None
    return pipeline


@pytest.mark.layer1
@pytest.mark.multimodal
def test_practice_text_only_generation_uses_practice_chain():
    pipeline = pipeline_stub()
    pipeline._practice_chain.invoke.return_value = "1. Practice problem"
    pipeline._rag_chain.invoke.return_value = "generic answer"

    answer = pipeline._generate_multimodal(
        "give me practice problems for fluids",
        "fluid mechanics context",
        [],
        query_type="practice",
    )

    assert answer == "1. Practice problem"
    pipeline._practice_chain.invoke.assert_called_once()
    pipeline._rag_chain.invoke.assert_not_called()
    assert pipeline._last_multimodal_usage["generation_mode"] == "text_only"


@pytest.mark.layer1
@pytest.mark.multimodal
def test_practice_vlm_failure_falls_back_to_practice_chain(tmp_path):
    image = tmp_path / "page.png"
    image.write_bytes(b"fake png bytes")

    pipeline = pipeline_stub()
    pipeline.vision_llm = MagicMock()
    pipeline.vision_llm.invoke.side_effect = RuntimeError("vision unavailable")
    pipeline._practice_chain.invoke.return_value = "practice fallback"
    pipeline._rag_chain.invoke.return_value = "generic fallback"

    answer = pipeline._generate_multimodal(
        "make a practice quiz",
        "visual fluids context",
        [{
            "image_path": str(image),
            "has_table": True,
            "text": "pressure diagram",
            "rerank_score": 2.0,
        }],
        query_type="practice",
    )

    assert answer == "practice fallback"
    pipeline._practice_chain.invoke.assert_called_once()
    pipeline._rag_chain.invoke.assert_not_called()
    assert pipeline._last_multimodal_usage["generation_mode"] == "text_only_fallback"
    assert pipeline._last_multimodal_usage["tables_attached"] == 1


@pytest.mark.layer1
@pytest.mark.multimodal
def test_image_selection_respects_top_k_deduplicates_and_skips_missing(tmp_path):
    first = tmp_path / "one.png"
    second = tmp_path / "two.png"
    third = tmp_path / "three.png"
    for path in (first, second, third):
        path.write_bytes(b"x")

    pipeline = pipeline_stub(image_top_k=2)
    selected = pipeline._select_image_chunks([
        {"image_path": str(tmp_path / "missing.png"), "rerank_score": 10.0},
        {"image_path": str(first), "rerank_score": 9.0},
        {"image_path": str(first), "rerank_score": 8.0},
        {"image_path": str(second), "rerank_score": 7.0},
        {"image_path": str(third), "rerank_score": 6.0},
    ])

    assert [Path(c["image_path"]).name for c in selected] == ["one.png", "two.png"]


@pytest.mark.layer1
@pytest.mark.multimodal
def test_captioner_gracefully_handles_unlabeled_model_output(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"image")

    captioner = VisionCaptioner(enabled=True)
    captioner._llm = MagicMock()
    captioner._llm.invoke.return_value = SimpleNamespace(
        content="A diagram showing pressure increasing with depth."
    )

    caption, markdown = captioner.caption(image)

    assert caption == "A diagram showing pressure increasing with depth."
    assert markdown == ""


@pytest.mark.layer1
@pytest.mark.multimodal
def test_extract_and_caption_contains_caption_failure_to_single_record(tmp_path, monkeypatch):
    image = tmp_path / "page.png"
    image.write_bytes(b"image")
    records = [
        ImageRecord("doc", 1, str(image), ""),
    ]

    monkeypatch.setattr(
        multimodal.PageImageExtractor,
        "extract",
        lambda self, pdf_path, doc_id: records,
    )
    monkeypatch.setattr(
        multimodal.VisionCaptioner,
        "caption",
        lambda self, image_path: (_ for _ in ()).throw(RuntimeError("bad VLM")),
    )

    output = extract_and_caption(
        tmp_path / "fake.pdf",
        "doc",
        tmp_path / "images",
        enabled=True,
        workers=1,
    )

    assert output is records
    assert output[0].caption == ""
    assert output[0].markdown == ""


@pytest.mark.layer1
@pytest.mark.multimodal
@pytest.mark.parametrize("text", [
    "plain text with | a pipe but no table separator",
    "| one | two |\n| value | value |",
    "a || b",
])
def test_table_detector_rejects_pipe_text_without_separator_row(text):
    assert markdown_has_table(text) is False
