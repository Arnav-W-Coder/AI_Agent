"""URL safety and quality checks used before web content ingestion."""
from __future__ import annotations

import ipaddress
import re
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


def _domain_score(host: str, url: str) -> tuple[int, list[str]]:
    """Score a domain from observable properties, not a site-name allowlist.

    This is a prior, not a claim that a page is true. The fetcher/reranker should
    still evaluate the actual page title, text quality, and agreement with sources.
    """
    labels = [part for part in host.split(".") if part]
    score = 50
    reasons: list[str] = []
    tld = labels[-1] if labels else ""
    registrable = ".".join(labels[-2:]) if len(labels) >= 2 else host
    path = urlsplit(url).path.lower()

    if tld in {"gov", "mil"}:
        score += 28
        reasons.append("government_tld")
    elif tld == "edu" or host.endswith(".ac.uk") or host.endswith(".ac.jp"):
        score += 25
        reasons.append("academic_tld")
    elif tld in {"org", "int"}:
        score += 8
        reasons.append("organization_tld")
    elif tld in {"com", "net", "io", "ai", "dev", "co"}:
        score += 2

    if len(labels) == 2:
        score += 8
        reasons.append("direct_registrable_domain")
    elif len(labels) >= 5:
        score -= 8
        reasons.append("deep_subdomain")

    if urlsplit(url).scheme == "https":
        score += 4
        reasons.append("https")
    if any(token in path for token in ("/docs", "/documentation", "/reference", "/api", "/manual", "/papers", "/research")):
        score += 6
        reasons.append("documentation_or_research_path")
    if any(token in host for token in ("blog", "forum", "paste", "wiki")):
        score -= 6
        reasons.append("user_generated_or_editorial_subdomain")
    if re.search(r"(^|[-.])(free|download|torrent|casino|adult)([-.]|$)", host):
        score -= 25
        reasons.append("high_risk_host_pattern")
    if registrable in {"wikipedia.org", "stackoverflow.com"}:
        score -= 5
        reasons.append("secondary_reference_source")

    return max(0, min(100, score)), reasons


def evaluate_url(url: str, *, min_domain_score: int = 55) -> dict:
    """Return a decision before fetching, embedding, or persisting a URL.

    Authority is estimated from generic, explainable signals. No vendor or
    documentation site is permanently trusted by hostname. Page-level quality,
    retrieval relevance, and cross-source agreement must be checked downstream.
    """
    normalized = normalize_url(url)
    if not normalized:
        return {
            "allowed": False, "approved": False,
            "reason": "invalid_or_blocked_url", "url": None,
            "normalized_url": None, "host": None, "domain_score": 0,
        }
    host = urlsplit(normalized).hostname or ""
    if not public_host(host):
        return {
            "allowed": False, "approved": False,
            "reason": "non_public_host", "url": normalized,
            "normalized_url": normalized, "host": host, "domain_score": 0,
        }
    score, reasons = _domain_score(host, normalized)
    approved = score >= min_domain_score
    return {
        "allowed": approved,
        "approved": approved,
        "reason": "approved" if approved else "low_domain_authority",
        "url": normalized,
        "normalized_url": normalized,
        "host": host,
        "domain_score": score,
        "domain_score_reasons": reasons,
    }
