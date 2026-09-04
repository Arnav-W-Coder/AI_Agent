"""
chunking.py — structure-aware, semantic-recursive hierarchical chunking.
"""
from __future__ import annotations
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Callable, Sequence
import numpy as np
import tiktoken
from langchain_core.documents import Document

log = logging.getLogger(__name__)

@dataclass
class Section:
    title: str
    path: str
    text: str
    start_page: int
    end_page: int

@dataclass
class ChunkRecord:
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
    """Structure boundary -> semantic split -> recursive cap -> parent/child."""
    def __init__(self, cfg, embedding_fn: Callable[[list[str]], list[list[float]]]):
        self.cfg = cfg
        self.embedding_fn = embedding_fn
        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.encoder = None

    def _tokens(self, text: str) -> int:
        if self.encoder is not None:
            return len(self.encoder.encode(text or ""))
        return max(1, len((text or "").split()))

    @staticmethod
    def _looks_like_heading(line: str) -> bool:
        line = line.strip()
        if not line or len(line) > 140:
            return False
        if line.startswith("#") or re.match(r"^(?:chapter|section|part|appendix)\b", line, re.I):
            return True
        if re.match(r"^\d+(?:\.\d+){0,3}[.)]?\s+\S+", line):
            return len(line.split()) <= 14
        return len(line.split()) <= 10 and line[-1:] not in ".,;:!?" and line == line.upper()

    @staticmethod
    def _heading_level(line: str) -> int:
        line = line.strip()
        if line.startswith("#"):
            return min(len(line) - len(line.lstrip("#")), 6)
        if re.match(r"^(?:chapter|part)\b", line, re.I) or re.match(r"^\d+\.?\s+", line):
            return 1
        if re.match(r"^\d+\.\d+", line):
            return 2
        return 3

    def _structure_sections(self, pages: Sequence[Document]) -> list[Section]:
        sections: list[Section] = []
        title, path = "Document", "Document"
        lines: list[str] = []
        start = end = 1
        hierarchy: dict[int, str] = {}
        def flush():
            nonlocal lines
            text = "\n".join(lines).strip()
            if text:
                sections.append(Section(title, path, text, start, end))
            lines = []
        for page_idx, page in enumerate(pages):
            page_no = int(page.metadata.get("page", page_idx)) + 1
            for raw in page.page_content.replace("\r", "").splitlines():
                line = re.sub(r"[ \t]+", " ", raw).strip()
                if not line:
                    if lines and lines[-1] != "":
                        lines.append("")
                    continue
                if self._looks_like_heading(line):
                    flush()
                    level = self._heading_level(line)
                    hierarchy[level] = line.lstrip("#").strip()
                    for key in list(hierarchy):
                        if key > level:
                            del hierarchy[key]
                    title = hierarchy[level]
                    path = " > ".join(hierarchy[k] for k in sorted(hierarchy))
                    start = end = page_no
                    continue
                if not lines:
                    start = page_no
                end = page_no
                lines.append(line)
        flush()
        if not sections:
            text = "\n".join(p.page_content for p in pages).strip()
            return [Section("Document", "Document", text, 1, max(1, len(pages)))]
        return sections

    @staticmethod
    def _paragraphs(text: str) -> list[str]:
        return [re.sub(r"\s+", " ", b).strip() for b in re.split(r"\n\s*\n+", text) if b.strip()]

    def _semantic_blocks(self, paragraphs: list[str]) -> list[str]:
        if len(paragraphs) <= 1 or not self.cfg.semantic_chunking_enabled:
            return paragraphs
        try:
            vectors = np.asarray(self.embedding_fn(paragraphs), dtype=np.float32)
            vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
            distances = 1.0 - np.sum(vectors[:-1] * vectors[1:], axis=1)
            threshold = max(float(np.percentile(distances, self.cfg.semantic_breakpoint_percentile)), self.cfg.semantic_min_distance)
        except Exception as exc:
            log.warning("[Chunking] Semantic boundary detection failed: %s", exc)
            return paragraphs
        blocks, current = [], []
        for i, paragraph in enumerate(paragraphs):
            current.append(paragraph)
            if i < len(paragraphs) - 1 and distances[i] >= threshold and self._tokens("\n\n".join(current)) >= self.cfg.semantic_min_block_tokens:
                blocks.append("\n\n".join(current)); current = []
        if current:
            blocks.append("\n\n".join(current))
        return blocks

    def _recursive_cap(self, text: str, max_tokens: int) -> list[str]:
        if self._tokens(text) <= max_tokens:
            return [text.strip()]
        pieces = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
        if len(pieces) == 1:
            pieces = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
        if len(pieces) == 1:
            words = text.split()
            return [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]
        out, current = [], []
        for piece in pieces:
            candidate = "\n\n".join(current + [piece])
            if current and self._tokens(candidate) > max_tokens:
                out.extend(self._recursive_cap("\n\n".join(current), max_tokens)); current = [piece]
            else:
                current.append(piece)
        if current:
            out.extend(self._recursive_cap("\n\n".join(current), max_tokens))
        return out

    def _make_parents(self, section: Section) -> list[str]:
        parents, current = [], []
        for block in self._semantic_blocks(self._paragraphs(section.text)):
            candidate = "\n\n".join(current + [block])
            if current and self._tokens(candidate) > self.cfg.parent_target_tokens:
                parents.extend(self._recursive_cap("\n\n".join(current), self.cfg.parent_max_tokens)); current = [block]
            else:
                current.append(block)
        if current:
            parents.extend(self._recursive_cap("\n\n".join(current), self.cfg.parent_max_tokens))
        return [p for p in parents if p.strip()]

    def _make_children(self, text: str) -> list[str]:
        pieces = self._recursive_cap(text, self.cfg.child_max_tokens)
        if self.cfg.child_overlap_tokens <= 0:
            return pieces
        out = []
        for i, piece in enumerate(pieces):
            if i == 0:
                out.append(piece)
            else:
                overlap = " ".join(pieces[i - 1].split()[-self.cfg.child_overlap_tokens:])
                out.append((overlap + "\n" + piece).strip())
        return out

    def chunk(self, pages: Sequence[Document], doc_id: str) -> tuple[list[ChunkRecord], list[ChunkRecord]]:
        parents, children = [], []
        parent_index = child_index = 0
        for section in self._structure_sections(pages):
            for parent_text in self._make_parents(section):
                parent_id = str(uuid.uuid4())
                contextual = f"{section.path}\n\n{parent_text}" if section.path else parent_text
                # Negative indices keep parent and child namespaces distinct while
                # preserving adjacency for neighbor lookup in SQLite.
                parents.append(ChunkRecord(parent_id, contextual, "parent", -(parent_index + 1), None,
                                            section.start_page, section.end_page, section.path, {"doc_id": doc_id}))
                for child_text in self._make_children(contextual):
                    children.append(ChunkRecord(str(uuid.uuid4()), child_text, "child", child_index,
                                                parent_id, section.start_page, section.end_page, section.path,
                                                {"doc_id": doc_id, "parent_id": parent_id}))
                    child_index += 1
                parent_index += 1
        return parents, children
