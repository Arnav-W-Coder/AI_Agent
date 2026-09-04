"""
chunking.py — structure-aware, semantic-recursive hierarchical chunking.

Pipeline:
  PDF pages -> structural sections -> semantic blocks -> parent chunks -> child chunks

The hierarchy is intentional:
  - structure boundaries prevent unrelated sections from being mixed
  - semantic splitting follows topical changes inside a section
  - recursive token capping guarantees bounded chunks
  - parents preserve context; children are optimized for retrieval
"""
from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
import tiktoken
from langchain_core.documents import Document

log = logging.getLogger(__name__)


@dataclass
class Section:
    """A structural region that should normally not be crossed by a parent chunk."""

    title: str
    path: str
    text: str
    start_page: int
    end_page: int


@dataclass
class ChunkRecord:
    """Persistable parent/child chunk representation."""

    id: str
    text: str
    chunk_type: str
    chunk_index: int
    parent_id: str | None
    start_page: int
    end_page: int
    section_path: str
    metadata: dict = field(default_factory=dict)


class HierarchicalChunker:
    """Create structure-aware semantic parent/child chunks."""

    def __init__(self, cfg, embedding_fn: Callable[[list[str]], list[list[float]]]):
        self.cfg = cfg
        self.embedding_fn = embedding_fn
        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.encoder = None

    def _tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.encoder is not None:
            return len(self.encoder.encode(text))
        return max(1, len(text.split()))

    def _truncate(self, text: str, max_tokens: int) -> str:
        if self._tokens(text) <= max_tokens:
            return text.strip()
        if self.encoder is not None:
            ids = self.encoder.encode(text)[:max_tokens]
            return self.encoder.decode(ids).strip()
        return " ".join(text.split()[:max_tokens]).strip()

    @staticmethod
    def _looks_like_heading(line: str) -> bool:
        line = line.strip()
        if not line or len(line) > 140:
            return False
        if line.startswith("#"):
            return True
        if re.match(r"^(?:chapter|section|part|appendix)\b", line, re.I):
            return True
        if re.match(r"^\d+(?:\.\d+){0,3}[.)]?\s+\S+", line):
            return len(line.split()) <= 14
        words = line.split()
        if len(words) <= 10 and line[-1:] not in ".,;:!?" and line == line.upper():
            return True
        return False

    @staticmethod
    def _heading_level(line: str) -> int:
        line = line.strip()
        if line.startswith("#"):
            return min(len(line) - len(line.lstrip("#")), 6)
        if re.match(r"^(?:chapter|part)\b", line, re.I):
            return 1
        if re.match(r"^\d+\.?\s+", line):
            return 1
        if re.match(r"^\d+\.\d+", line):
            return 2
        return 3

    def _structure_sections(self, pages: Sequence[Document]) -> list[Section]:
        """Build lightweight Markdown-like sections from PDF page text.

        PyPDFLoader does not expose reliable font/layout semantics for every PDF, so
        this parser combines explicit Markdown/numbered headings with page boundaries.
        It is deliberately conservative: weak heading signals do not split a section.
        """
        sections: list[Section] = []
        current_title = "Document"
        current_path = "Document"
        current_lines: list[str] = []
        current_start = 0
        current_end = 0
        hierarchy: dict[int, str] = {}

        def flush() -> None:
            nonlocal current_lines
            text = "\n".join(current_lines).strip()
            if text:
                sections.append(Section(
                    title=current_title,
                    path=current_path,
                    text=text,
                    start_page=current_start,
                    end_page=current_end,
                ))
            current_lines = []

        for page_idx, page in enumerate(pages):
            page_no = int(page.metadata.get("page", page_idx)) + 1
            lines = page.page_content.replace("\r", "").splitlines()
            for raw in lines:
                line = re.sub(r"[ \t]+", " ", raw).strip()
                if not line:
                    if current_lines and current_lines[-1] != "":
                        current_lines.append("")
                    continue
                if self._looks_like_heading(line) and current_lines:
                    flush()
                    level = self._heading_level(line)
                    hierarchy[level] = line.lstrip("#").strip()
                    for k in list(hierarchy):
                        if k > level:
                            del hierarchy[k]
                    current_title = hierarchy[level]
                    current_path = " > ".join(hierarchy[k] for k in sorted(hierarchy))
                    current_start = page_no
                elif not current_lines:
                    current_start = page_no
                current_end = page_no
                current_lines.append(line)
        flush()

        if not sections:
            return [Section("Document", "Document", "\n".join(p.page_content for p in pages), 1, max(1, len(pages)))]
        return sections

    @staticmethod
    def _paragraphs(text: str) -> list[str]:
        blocks = re.split(r"\n\s*\n+", text)
        return [re.sub(r"\s+", " ", b).strip() for b in blocks if b.strip()]

    def _semantic_blocks(self, paragraphs: list[str]) -> list[str]:
        """Split paragraphs at strong semantic discontinuities.

        Paragraph embeddings are only used to find boundaries; child embeddings are
        generated later and are therefore not duplicated in the vector index.
        """
        if len(paragraphs) <= 1 or not self.cfg.semantic_chunking_enabled:
            return paragraphs

        # Avoid spending embedding calls on tiny fragments.
        usable = [p for p in paragraphs if self._tokens(p) >= self.cfg.semantic_min_paragraph_tokens]
        if len(usable) <= 1:
            return paragraphs
        if len(usable) != len(paragraphs):
            paragraphs = usable

        try:
            vectors = np.asarray(self.embedding_fn(paragraphs), dtype=np.float32)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norms, 1e-8)
            similarities = np.sum(vectors[:-1] * vectors[1:], axis=1)
            distances = 1.0 - similarities
            threshold = float(np.percentile(distances, self.cfg.semantic_breakpoint_percentile))
            threshold = max(threshold, self.cfg.semantic_min_distance)
        except Exception as exc:
            log.warning("[Chunking] Semantic boundary detection failed: %s", exc)
            return paragraphs

        blocks: list[str] = []
        current: list[str] = []
        for i, paragraph in enumerate(paragraphs):
            current.append(paragraph)
            should_break = (
                i < len(paragraphs) - 1
                and distances[i] >= threshold
                and self._tokens("\n\n".join(current)) >= self.cfg.semantic_min_block_tokens
            )
            if should_break:
                blocks.append("\n\n".join(current))
                current = []
        if current:
            blocks.append("\n\n".join(current))
        return blocks

    def _recursive_cap(self, text: str, max_tokens: int) -> list[str]:
        """Hard cap using paragraph -> sentence -> whitespace recursion."""
        if self._tokens(text) <= max_tokens:
            return [text.strip()]

        pieces = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
        if len(pieces) == 1:
            pieces = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
        if len(pieces) == 1:
            words = text.split()
            return [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]

        out: list[str] = []
        current: list[str] = []
        for piece in pieces:
            candidate = "\n\n".join(current + [piece])
            if current and self._tokens(candidate) > max_tokens:
                out.extend(self._recursive_cap("\n\n".join(current), max_tokens))
                current = [piece]
            else:
                current.append(piece)
        if current:
            out.extend(self._recursive_cap("\n\n".join(current), max_tokens))
        return out

    def _make_parents(self, section: Section) -> list[str]:
        semantic = self._semantic_blocks(self._paragraphs(section.text))
        parents: list[str] = []
        current: list[str] = []
        for block in semantic:
            candidate = "\n\n".join(current + [block])
            if current and self._tokens(candidate) > self.cfg.parent_target_tokens:
                parents.extend(self._recursive_cap("\n\n".join(current), self.cfg.parent_max_tokens))
                current = [block]
            else:
                current.append(block)
        if current:
            parents.extend(self._recursive_cap("\n\n".join(current), self.cfg.parent_max_tokens))
        return [p for p in parents if p.strip()]

    def _make_children(self, parent_text: str) -> list[str]:
        pieces = self._recursive_cap(parent_text, self.cfg.child_max_tokens)
        if self.cfg.child_overlap_tokens <= 0 or len(pieces) <= 1:
            return pieces
        children: list[str] = []
        for i, piece in enumerate(pieces):
            if i == 0:
                children.append(piece)
                continue
            previous_words = pieces[i - 1].split()
            overlap = " ".join(previous_words[-self.cfg.child_overlap_tokens:])
            children.append((overlap + "\n" + piece).strip())
        return children

    def chunk(self, pages: Sequence[Document], doc_id: str) -> tuple[list[ChunkRecord], list[ChunkRecord]]:
        """Return (parents, children), with children linked by parent_id."""
        parents: list[ChunkRecord] = []
        children: list[ChunkRecord] = []
        parent_index = 0
        child_index = 0

        for section in self._structure_sections(pages):
            for parent_text in self._make_parents(section):
                parent_id = str(uuid.uuid4())
                parent_text = parent_text.strip()
                # Breadcrumb is part of retrieval context without overwhelming embeddings.
                contextual_parent = f"{section.path}\n\n{parent_text}" if section.path else parent_text
                parent = ChunkRecord(
                    id=parent_id,
                    text=contextual_parent,
                    chunk_type="parent",
                    chunk_index=parent_index,
                    parent_id=None,
                    start_page=section.start_page,
                    end_page=section.end_page,
                    section_path=section.path,
                    metadata={"doc_id": doc_id, "source": "", "page": section.start_page},
                )
                parents.append(parent)
                for child_text in self._make_children(contextual_parent):
                    child = ChunkRecord(
                        id=str(uuid.uuid4()),
                        text=child_text,
                        chunk_type="child",
                        chunk_index=child_index,
                        parent_id=parent_id,
                        start_page=section.start_page,
                        end_page=section.end_page,
                        section_path=section.path,
                        metadata={
                            "doc_id": doc_id,
                            "parent_id": parent_id,
                            "chunk_type": "child",
                            "page": section.start_page,
                            "start_page": section.start_page,
                            "end_page": section.end_page,
                            "section_path": section.path,
                        },
                    )
                    children.append(child)
                    child_index += 1
                parent_index += 1

        return parents, children
