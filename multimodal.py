"""Optional multimodal PDF indexing for image-aware RAG.

Captions and markdown are embedded for retrieval; rendered images remain on disk
and are loaded only when the answer-generation step needs visual evidence.
"""
from __future__ import annotations

import base64
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ImageRecord:
    doc_id: str
    page_number: int
    file_path: str
    caption: str
    markdown: str = ""
    width: int = 0
    height: int = 0
    chroma_id: str | None = None


class PageImageExtractor:
    def __init__(self, output_dir: str | Path, dpi: int = 144, min_drawing_count: int = 12, max_per_doc: int = 24):
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        self.min_drawing_count = min_drawing_count
        self.max_per_doc = max_per_doc

    def extract(self, pdf_path: str | Path, doc_id: str) -> list[ImageRecord]:
        try:
            import fitz
        except ImportError:
            return []
        pdf_path = Path(pdf_path)
        out = self.output_dir / doc_id
        out.mkdir(parents=True, exist_ok=True)
        records: list[ImageRecord] = []
        with fitz.open(pdf_path) as doc:
            for page_index, page in enumerate(doc):
                meaningful = bool(page.get_images(full=True)) or len(page.get_drawings()) >= self.min_drawing_count
                if not meaningful:
                    continue
                pix = page.get_pixmap(dpi=self.dpi, alpha=False)
                path = out / f"page_{page_index + 1}.png"
                pix.save(str(path))
                records.append(ImageRecord(doc_id, page_index + 1, str(path), "", "", pix.width, pix.height))
                if len(records) >= self.max_per_doc:
                    break
        return records


class VisionCaptioner:
    def __init__(self, model: str = "llama3.2-vision", enabled: bool = True):
        self.model = model
        self.enabled = enabled
        self._llm: Any = None

    def _get_llm(self):
        if self._llm is None:
            from langchain_ollama import ChatOllama
            self._llm = ChatOllama(model=self.model, temperature=0)
        return self._llm

    def caption(self, image_path: str | Path) -> tuple[str, str]:
        if not self.enabled:
            return "", ""
        data = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        prompt = "Describe this PDF page image precisely. Return exactly two labeled sections: CAPTION: a concise description; MARKDOWN: faithful markdown transcription of visible text, tables, equations, and labels. Do not invent unreadable content."
        message = {"role": "user", "content": prompt, "images": [data]}
        response = self._get_llm().invoke([message])
        text = response.content if hasattr(response, "content") else str(response)
        caption = re.search(r"CAPTION:\s*(.*?)(?=\nMARKDOWN:|$)", text, re.S | re.I)
        markdown = re.search(r"MARKDOWN:\s*(.*)$", text, re.S | re.I)
        return (caption.group(1).strip() if caption else text.strip(), markdown.group(1).strip() if markdown else "")


def extract_and_caption(pdf_path: str | Path, doc_id: str, output_dir: str | Path, dpi: int = 144, min_drawing_count: int = 12, max_per_doc: int = 24, model: str = "llama3.2-vision", workers: int = 2, enabled: bool = True) -> list[ImageRecord]:
    records = PageImageExtractor(output_dir, dpi, min_drawing_count, max_per_doc).extract(pdf_path, doc_id)
    if not records or not enabled:
        return records
    captioner = VisionCaptioner(model, enabled)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(captioner.caption, r.file_path): r for r in records}
        for future in as_completed(futures):
            record = futures[future]
            try:
                record.caption, record.markdown = future.result()
            except Exception:
                record.caption, record.markdown = "", ""
    return records
