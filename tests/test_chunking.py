"""
Unit tests for the hierarchical/semantic RAG chunking layer.

Save this file as:
    tests/test_chunking.py

Run:
    pytest -q tests/test_chunking.py
"""

from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from chunking import HierarchicalChunker


@pytest.fixture
def cfg():
    return SimpleNamespace(
        semantic_chunking_enabled=True,
        semantic_breakpoint_percentile=75.0,
        semantic_min_distance=0.10,
        semantic_min_block_tokens=1,
        parent_target_tokens=80,
        parent_max_tokens=100,
        child_max_tokens=30,
        child_overlap_tokens=5,
    )


@pytest.fixture
def chunker(cfg):
    # Deterministic fake embeddings: no Ollama is needed for these tests.
    def fake_embeddings(texts):
        vectors = []
        for text in texts:
            t = text.lower()
            if "tcp" in t:
                vectors.append([1.0, 0.0, 0.0])
            elif "cpu" in t:
                vectors.append([0.0, 1.0, 0.0])
            elif "routing" in t:
                vectors.append([0.0, 0.0, 1.0])
            else:
                vectors.append([1.0, 1.0, 0.0])
        return vectors

    return HierarchicalChunker(cfg, fake_embeddings)


def test_structure_aware_sections(chunker):
    pages = [Document(
        page_content=(
            "# Networking\n\n"
            "TCP provides reliable delivery.\n\n"
            "TCP retransmits missing data.\n\n"
            "# Routing\n\n"
            "Routers forward packets between networks."
        ),
        metadata={"page": 0},
    )]

    sections = chunker._structure_sections(pages)

    assert len(sections) == 2
    assert sections[0].title == "Networking"
    assert sections[0].path == "Networking"
    assert "TCP provides reliable delivery." in sections[0].text
    assert sections[1].title == "Routing"
    assert sections[1].path == "Routing"


def test_nested_heading_paths(chunker):
    pages = [Document(
        page_content=(
            "# Networking\n\n"
            "Introduction.\n\n"
            "## TCP\n\n"
            "TCP provides reliable delivery.\n\n"
            "### Retransmission\n\n"
            "TCP retransmits missing data."
        ),
        metadata={"page": 0},
    )]

    sections = chunker._structure_sections(pages)

    assert any(s.path == "Networking > TCP" for s in sections)
    assert any(s.path == "Networking > TCP > Retransmission" for s in sections)


def test_semantic_split_creates_topic_blocks(chunker):
    paragraphs = [
        "TCP provides reliable ordered delivery.",
        "TCP retransmits missing packets.",
        "CPU registers store temporary values.",
        "CPU executes instructions.",
    ]

    blocks = chunker._semantic_blocks(paragraphs)

    assert len(blocks) >= 2
    assert any("TCP provides" in block and "TCP retransmits" in block for block in blocks)
    assert any("CPU registers" in block and "CPU executes" in block for block in blocks)


def test_semantic_chunking_falls_back_when_embeddings_fail(cfg):
    def broken_embeddings(texts):
        raise RuntimeError("embedding service unavailable")

    chunker = HierarchicalChunker(cfg, broken_embeddings)

    paragraphs = ["First paragraph.", "Second paragraph.", "Third paragraph."]
    blocks = chunker._semantic_blocks(paragraphs)

    assert blocks == paragraphs


def test_recursive_cap_respects_max_tokens(chunker):
    text = " ".join(["word"] * 150)

    pieces = chunker._recursive_cap(text, 30)

    assert len(pieces) > 1
    assert all(chunker._tokens(piece) <= 30 for piece in pieces)


def test_child_overlap_is_present(cfg):
    cfg.child_max_tokens = 20
    cfg.child_overlap_tokens = 4

    chunker = HierarchicalChunker(
        cfg,
        lambda texts: [[1.0, 0.0] for _ in texts],
    )

    text = " ".join(f"word{i}" for i in range(80))
    children = chunker._make_children(text)

    assert len(children) > 1

    first_words = children[0].split()
    second_words = children[1].split()

    assert second_words[:4] == first_words[-4:]


def test_chunk_creates_parent_child_relationships(chunker):
    pages = [Document(
        page_content=(
            "# Networking\n\n"
            "TCP provides reliable delivery.\n\n"
            "TCP retransmits missing data.\n\n"
            "TCP uses acknowledgements.\n\n"
            "Routing tables determine packet destinations."
        ),
        metadata={"page": 0},
    )]

    parents, children = chunker.chunk(pages, "doc-123")

    assert parents
    assert children

    parent_ids = {p.id for p in parents}

    for parent in parents:
        assert parent.chunk_type == "parent"
        assert parent.parent_id is None
        assert parent.chunk_index < 0
        assert parent.metadata["doc_id"] == "doc-123"

    for child in children:
        assert child.chunk_type == "child"
        assert child.parent_id in parent_ids
        assert child.chunk_index >= 0
        assert child.metadata["parent_id"] == child.parent_id


def test_parent_and_child_indexes_are_disjoint(chunker):
    pages = [Document(
        page_content="# Test\n\nThis is a short document.",
        metadata={"page": 0},
    )]

    parents, children = chunker.chunk(pages, "doc-123")

    parent_indexes = {p.chunk_index for p in parents}
    child_indexes = {c.chunk_index for c in children}

    assert parent_indexes.isdisjoint(child_indexes)


def test_chunk_preserves_page_numbers(chunker):
    pages = [
        Document(
            page_content="# First\n\nContent on page one.",
            metadata={"page": 0},
        ),
        Document(
            page_content="# Second\n\nContent on page two.",
            metadata={"page": 1},
        ),
    ]

    parents, children = chunker.chunk(pages, "doc-123")

    assert parents
    assert children
    assert all(p.start_page >= 1 for p in parents)
    assert all(c.start_page >= 1 for c in children)


def test_empty_document_does_not_crash(chunker):
    pages = [Document(page_content="", metadata={"page": 0})]

    parents, children = chunker.chunk(pages, "empty-doc")

    assert parents == []
    assert children == []


def test_multiple_sections_produce_distinct_section_paths(chunker):
    pages = [Document(
        page_content=(
            "# TCP\n\nTCP is reliable.\n\n"
            "# CPU\n\nCPU executes instructions."
        ),
        metadata={"page": 0},
    )]

    parents, children = chunker.chunk(pages, "doc-123")

    paths = {p.section_path for p in parents}

    assert "TCP" in paths
    assert "CPU" in paths


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
