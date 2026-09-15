"""checkpoints.py — Lightweight data-flow checkpoints for RAG debugging.

Checkpoints intentionally log structure and small previews instead of dumping
entire documents, prompts, embeddings, or model outputs. Enable them with
RAGConfig.debug_checkpoints=True.
"""
import json
import logging
import math
from typing import Any

log = logging.getLogger(__name__)


def _safe_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return str(value)
        return value
    return type(value).__name__


def _summarize(value: Any, preview_chars: int, sample_items: int) -> Any:
    if isinstance(value, str):
        return {"type": "str", "chars": len(value), "preview": value[:preview_chars]}
    if isinstance(value, dict):
        keys = list(value.keys())
        summary = {"type": "dict", "keys": [str(k) for k in keys], "size": len(value)}
        for key in ("text", "source_url", "filename", "title", "source_type", "chroma_id",
                    "parent_id", "section_path", "rerank_score", "rrf_score", "bm25_score",
                    "dense_score", "page_number"):
            if key in value:
                item = value[key]
                if key == "text":
                    summary["text"] = {"chars": len(item or ""), "preview": (item or "")[:preview_chars]}
                else:
                    summary[key] = _safe_scalar(item)
        return summary
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        return {"type": type(value).__name__, "size": len(items),
                "samples": [_summarize(x, preview_chars, sample_items) for x in items[:sample_items]]}
    return _safe_scalar(value)


def checkpoint(stage: str, data: Any = None, *, enabled: bool = True,
               preview_chars: int = 160, sample_items: int = 3, **details: Any) -> None:
    """Emit one compact, structured checkpoint describing data at `stage`."""
    if not enabled:
        return
    payload = {"stage": stage}
    if data is not None:
        payload["data"] = _summarize(data, preview_chars, sample_items)
    if details:
        payload["details"] = {k: _safe_scalar(v) for k, v in details.items()}
    log.info("[Checkpoint] %s", json.dumps(payload, default=str, ensure_ascii=True))
