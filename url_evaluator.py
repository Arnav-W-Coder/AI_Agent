"""URL safety and quality checks used before web content ingestion."""
from __future__ import annotations

import ipaddress
import socket
from pathlib import Path
from urllib.parse import urlsplit


BLOCKED_HOSTS = {
    "localhost", "localhost.localdomain", "metadata.google.internal",
    "metadata.google",
}
BLOCKED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".mp3", ".mp4",
    ".avi", ".mov", ".zip", ".rar", ".7z", ".exe", ".dmg", ".iso",
}


def normalize_url(url: str) -> str | None:
    try:
        parts = urlsplit((url or "").strip())
        host = (parts.hostname or "").lower().rstrip(".")
        if parts.scheme.lower() not in {"http", "https"} or not host:
            return None
        if parts.username or parts.password or parts.port not in (None, 80, 443):
            return None
        if host in BLOCKED_HOSTS or Path((parts.path or "/").lower()).suffix in BLOCKED_EXTENSIONS:
            return None
        return f"{parts.scheme.lower()}://{host}{parts.path or '/'}" + (f"?{parts.query}" if parts.query else "")
    except (TypeError, ValueError):
        return None


def public_host(host: str) -> bool:
    try:
        addresses = {x[4][0] for x in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)}
    except OSError:
        return False
    return bool(addresses) and all(ipaddress.ip_address(a).is_global for a in addresses)


def evaluate_url(url: str, *, min_domain_score: int = 65) -> dict:
    """Return a decision before fetching, embedding, or persisting a URL."""
    normalized = normalize_url(url)
    if not normalized:
        return {
            "approved": False,
            "reason": "invalid_or_blocked_url",
            "url": None,
            "normalized_url": None,
            "host": None,
            "domain_score": 0,
        }
    host = urlsplit(normalized).hostname or ""
    if not public_host(host):
        return {
            "approved": False,
            "reason": "non_public_host",
            "url": normalized,
            "normalized_url": normalized,
            "host": host,
            "domain_score": 0,
        }
    labels = host.split(".")
    score = 92 if labels[-1] in {"gov", "edu"} else 60
    known = {
        "cppreference.com": 100, "cplusplus.com": 90,
        "learn.microsoft.com": 95, "docs.python.org": 100,
        "developer.mozilla.org": 95, "docs.oracle.com": 95,
        "kernel.org": 92, "llvm.org": 92, "gnu.org": 92,
        "iso.org": 92, "arxiv.org": 96, "pubmed.ncbi.nlm.nih.gov": 100,
        "github.com": 82, "stackoverflow.com": 78,
        "wikipedia.org": 60,
    }
    for i in range(len(labels) - 1):
        score = known.get(".".join(labels[i:]), score)
        if score != 60:
            break
    return {
        "approved": score >= min_domain_score,
        "reason": "approved" if score >= min_domain_score else "low_domain_authority",
        "url": normalized,
        "normalized_url": normalized,
        "host": host,
        "domain_score": score,
    }
