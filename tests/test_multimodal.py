"""Deterministic tests for visual retrieval and VLM attachment.

Run with: pytest -q tests/test_multimodal.py
Live Ollama evaluation belongs in the existing ``e2e`` suite; these tests
verify the local contracts without requiring a model server.
"""

import base64
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from multimodal import ImageRecord, PageImageExtractor, VisionCaptioner, markdown_has_table
from pipeline import ProductionRAGPipeline


EVAL_CASES = Path(__file__).parent / "data" / "multimodal_eval.jsonl"
PAGE_FIXTURES = [EVAL_CASES.parent / f"page_{page}.pdf" for page in range(1, 6)]


def _pipeline(image_top_k: int = 4) -> ProductionRAGPipeline:
    instance = ProductionRAGPipeline.__new__(ProductionRAGPipeline)
    instance.cfg = SimpleNamespace(image_top_k=image_top_k)
    instance._last_multimodal_usage = {}
    return instance


def _image_chunk(path: Path, **overrides) -> dict:
    chunk = {
        "image_path": str(path),
        "has_table": False,
        "rerank_score": 0.87,
        "text": "A labeled diagram of component A connected to component B.",
        "filename": "doc_001.pdf",
        "page_number": 1,
    }
    chunk.update(overrides)
    return chunk


@pytest.mark.layer1
@pytest.mark.multimodal
def test_image_record_preserves_document_page_and_existing_raw_path(tmp_path):
    image_path = tmp_path / "image_store" / "doc_001" / "page_1.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"png bytes")
    record = ImageRecord(
        doc_id="doc_001", page_number=1, file_path=str(image_path),
        caption="Diagram with components A and B.",
    )

    assert record.doc_id == "doc_001"
    assert record.page_number == 1
    assert Path(record.file_path).is_file()
    assert record.caption.startswith("Diagram")


@pytest.mark.layer1
@pytest.mark.multimodal
def test_attached_page_fixtures_are_real_one_page_pdfs():
    pytest.importorskip("fitz")
    import fitz

    assert all(path.is_file() for path in PAGE_FIXTURES)
    for path in PAGE_FIXTURES:
        with fitz.open(path) as document:
            assert len(document) == 1, path.name
            assert document[0].get_images(full=True) or document[0].get_text("text").strip() == ""


@pytest.mark.layer1
@pytest.mark.multimodal
def test_real_pdf_page_is_rendered_and_attached_to_vlm(tmp_path):
    pytest.importorskip("fitz")
    records = PageImageExtractor(
        tmp_path / "image_store", dpi=72, min_drawing_count=0, max_per_doc=1
    ).extract(PAGE_FIXTURES[0], "attached_doc")

    assert len(records) == 1
    assert records[0].doc_id == "attached_doc"
    assert records[0].page_number == 1
    rendered_path = Path(records[0].file_path)
    assert rendered_path.is_file()
    assert records[0].width > 0 and records[0].height > 0

    pipeline = _pipeline()
    pipeline._rag_chain = MagicMock()
    pipeline.vision_llm = MagicMock()
    pipeline.vision_llm.invoke.return_value = SimpleNamespace(content="Visual answer")
    pipeline._generate_multimodal(
        "What stages are shown?", "Attached page 1", [_image_chunk(rendered_path)]
    )

    message = pipeline.vision_llm.invoke.call_args.args[0][0]
    encoded = message.content[1]["image_url"].split(",", 1)[1]
    assert base64.b64decode(encoded) == rendered_path.read_bytes()
    assert pipeline._last_multimodal_usage["generation_mode"] == "vision"


@pytest.mark.layer1
@pytest.mark.multimodal
def test_table_markdown_is_detected_for_table_only_cases():
    markdown = "| Model | Latency (ms) | Accuracy (%) | Memory (GB) |\n| --- | ---: | ---: | ---: |\n| Model B | 80 | 88 | 6 |"

    assert markdown_has_table(markdown)
    assert not markdown_has_table("Model B has a latency of 80 ms.")


@pytest.mark.layer1
@pytest.mark.multimodal
def test_retrieval_selects_existing_unique_images_in_rank_order(tmp_path):
    first = tmp_path / "page_1.png"
    second = tmp_path / "page_2.png"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    pipeline = _pipeline(image_top_k=2)

    selected = pipeline._select_image_chunks([
        _image_chunk(first, rerank_score=0.9),
        _image_chunk(first, rerank_score=0.8),
        _image_chunk(second, rerank_score=0.7),
        _image_chunk(tmp_path / "missing.png", rerank_score=1.0),
    ])

    assert [chunk["image_path"] for chunk in selected] == [str(first), str(second)]


@pytest.mark.layer1
@pytest.mark.multimodal
def test_generation_attaches_raw_image_to_vision_request(tmp_path):
    image_path = tmp_path / "page_1.png"
    image_bytes = b"deterministic image bytes"
    image_path.write_bytes(image_bytes)
    pipeline = _pipeline()
    pipeline._rag_chain = MagicMock()
    pipeline.vision_llm = MagicMock()
    pipeline.vision_llm.invoke.return_value = SimpleNamespace(content="The arrow points to B.")

    answer = pipeline._generate_multimodal(
        "What does the arrow point toward?",
        "Caption: a diagram with an arrow.",
        [_image_chunk(image_path)],
    )

    message = pipeline.vision_llm.invoke.call_args.args[0][0]
    image_part = message.content[1]
    encoded = image_part["image_url"].split(",", 1)[1]
    assert answer == "The arrow points to B."
    assert image_part["type"] == "image_url"
    assert base64.b64decode(encoded) == image_bytes
    assert pipeline._last_multimodal_usage == {
        "generation_mode": "vision", "images_attached": 1, "tables_attached": 0,
    }


@pytest.mark.layer1
@pytest.mark.multimodal
def test_table_image_is_counted_as_attached_table(tmp_path):
    image_path = tmp_path / "table.png"
    image_path.write_bytes(b"table")
    pipeline = _pipeline()
    pipeline._rag_chain = MagicMock()
    pipeline.vision_llm = MagicMock()
    pipeline.vision_llm.invoke.return_value = SimpleNamespace(content="Model B is 80 ms.")

    pipeline._generate_multimodal(
        "What is the latency of Model B?", "table caption", [_image_chunk(image_path, has_table=True)]
    )

    assert pipeline._last_multimodal_usage["tables_attached"] == 1


@pytest.mark.layer1
@pytest.mark.multimodal
def test_missing_visual_evidence_uses_text_only_generation():
    pipeline = _pipeline()
    pipeline._rag_chain = MagicMock()
    pipeline._rag_chain.invoke.return_value = "I cannot verify that from the supplied evidence."
    pipeline.vision_llm = MagicMock()

    answer = pipeline._generate_multimodal(
        "Which label is absent?", "No visual evidence.", []
    )

    assert answer == "I cannot verify that from the supplied evidence."
    pipeline.vision_llm.invoke.assert_not_called()
    assert pipeline._last_multimodal_usage["generation_mode"] == "text_only"
    assert pipeline._last_multimodal_usage["images_attached"] == 0


@pytest.mark.layer1
@pytest.mark.multimodal
def test_vision_failure_is_recorded_as_text_only_fallback(tmp_path):
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"image")
    pipeline = _pipeline()
    pipeline._rag_chain = MagicMock()
    pipeline._rag_chain.invoke.return_value = "Fallback answer"
    pipeline.vision_llm = MagicMock()
    pipeline.vision_llm.invoke.side_effect = RuntimeError("VLM unavailable")

    assert pipeline._generate_multimodal("What is shown?", "caption", [_image_chunk(image_path)]) == "Fallback answer"
    assert pipeline._last_multimodal_usage["generation_mode"] == "text_only_fallback"
    assert pipeline._last_multimodal_usage["images_attached"] == 1


@pytest.mark.layer1
@pytest.mark.multimodal
def test_usage_and_provenance_report_retrieved_visual_source(tmp_path):
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"image")
    pipeline = _pipeline()
    source = pipeline._source_provenance(_image_chunk(image_path, has_table=True), 1)
    usage = pipeline._usage_from_sources([source], "vision", 1, 1)

    assert source["page"] == 1
    assert source["has_image"] is True
    assert source["has_table"] is True
    assert usage == {
        "generation_mode": "vision", "images_retrieved": 1,
        "tables_retrieved": 1, "images_attached": 1, "tables_attached": 1,
        "used_image": True, "used_table": True,
    }


@pytest.mark.layer1
@pytest.mark.multimodal
def test_vision_captioner_sends_image_bytes_to_caption_model(tmp_path):
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"caption image")
    captioner = VisionCaptioner(enabled=True)
    captioner._llm = MagicMock()
    captioner._llm.invoke.return_value = SimpleNamespace(
        content="CAPTION: labeled diagram\nMARKDOWN: A -> B"
    )

    caption, markdown = captioner.caption(image_path)
    message = captioner._llm.invoke.call_args.args[0][0]

    assert caption == "labeled diagram"
    assert markdown == "A -> B"
    assert base64.b64decode(message["images"][0]) == b"caption image"


@pytest.mark.layer1
@pytest.mark.multimodal
def test_evaluation_cases_are_independent_jsonl_answer_keys():
    cases = [json.loads(line) for line in EVAL_CASES.read_text(encoding="utf-8").splitlines()]

    assert len(cases) >= 10
    assert {case["type"] for case in cases} >= {"image", "table", "negative"}
    assert all(case["query"] and case["expected_page"] == 1 for case in cases)
    assert all("expected_answer" in case for case in cases)
    assert all((EVAL_CASES.parent / case["source_fixture"]).is_file() for case in cases)
    assert all(
        case["expected_image_id"] == f"img_page_{int(case['source_fixture'][5]) :03d}"
        for case in cases
    )