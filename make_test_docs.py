"""
make_test_docs.py — Populate ./docs with varied test PDFs for local development.

Strategy (in priority order):
  1. Download real free PDFs from public URLs (arXiv, Project Gutenberg).
     These give realistic chunking behaviour — messy layouts, references, etc.
  2. Fall back to synthetic PDFs for any download that fails.
     Synthetic docs contain unique fictional facts so retrieval tests can
     assert specific phrases without ambiguity.

Run:
    python make_test_docs.py

Downloads ~3-4 MB total. All sources are public domain or open access.
"""
import sys
import time
import struct
import zlib
from pathlib import Path

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("requests not installed — all docs will be synthetic.")

DOCS_DIR = Path("./docs")
DOCS_DIR.mkdir(exist_ok=True)

# ── Real PDF sources (public domain / open access) ───────────────────────────
# Each entry: (filename, url, description)
REAL_PDFS = [
    (
        "rag_paper.pdf",
        "https://arxiv.org/pdf/2005.11401",
        "Lewis et al. 2020 — original RAG paper (arXiv open access)",
    ),
    (
        "attention_is_all_you_need.pdf",
        "https://arxiv.org/pdf/1706.03762",
        "Vaswani et al. 2017 — Transformer paper (arXiv open access)",
    ),
    (
        "constitution_usa.pdf",
        "https://www.constitutionfacts.com/content/constitution/files/USConstitution_FulText.pdf",
        "US Constitution — public domain",
    ),
    (
        "declaration_of_independence.pdf",
        "https://www.archives.gov/founding-docs/downloads/declaration_of_independence.pdf",
        "US Declaration of Independence — public domain (national archives)",
    ),
]

# ── Synthetic PDF content (unique facts for deterministic testing) ────────────
SYNTHETIC_DOCS = [
    {
        "filename": "zephyr_protocol.pdf",
        "content": (
            "The Zephyr Protocol was established in 2047 in Nova City. "
            "It governs the exchange of synthetic data between autonomous agents. "
            "Under the protocol, all agents must register their embedding models. "
            "The primary architect was Dr. Lena Voss, who also wrote the Nova Manifesto. "
            "The protocol has three tiers: Bronze, Silver, and Gold certification. "
            "Bronze requires a minimum accuracy of 82 percent on standard benchmarks. "
            "Silver requires 91 percent and at least two peer audits per quarter. "
            "Gold certification is reviewed by the Nova City Ethics Board annually. "
            "Total registered agents as of 2049: 14,302. "
            "The Kline Amendment of 2048 added a mandatory transparency clause."
        ),
    },
    {
        "filename": "helios_framework.pdf",
        "content": (
            "The Helios Framework was founded in 3099 on Mars Colony Alpha. "
            "Its creator was Engineer Petra Kline, formerly of the Zephyr Institute. "
            "The framework manages solar energy distribution across six colonies. "
            "It uses the Kline Algorithm for load balancing across grid nodes. "
            "Peak output capacity is 4.7 terawatts during the Martian summer. "
            "The grid operates on a 28-hour Martian sol cycle. "
            "Colony Beta draws the highest average load at 1.1 terawatts. "
            "Firmware version 9.4.2 introduced predictive load shedding in 3101. "
            "The Helios Council meets quarterly to review distribution fairness. "
            "Engineer Kline retired in 3107 after 8 years leading the project."
        ),
    },
    {
        "filename": "retrieval_augmented_generation_overview.pdf",
        "content": (
            "Retrieval-Augmented Generation (RAG) combines a retrieval system "
            "with a large language model to produce grounded, verifiable answers. "
            "The retrieval step fetches relevant document chunks from a vector store. "
            "The generation step conditions the language model on the retrieved context. "
            "RAG reduces hallucinations by anchoring responses to real source material. "
            "Dense retrieval uses embedding similarity to find semantically relevant chunks. "
            "Sparse retrieval methods like BM25 use keyword overlap for candidate selection. "
            "Hybrid retrieval combines both methods using Reciprocal Rank Fusion (RRF). "
            "Cross-encoder rerankers score each candidate against the query for precision. "
            "Production RAG systems typically retrieve 20 candidates and rerank to 5. "
            "Context window limits require careful chunking of source documents. "
            "Chunk size of 800 characters with 150-character overlap is a common default. "
            "Metadata filters allow retrieval to be scoped to specific document sets. "
            "Answer caches reduce latency for semantically similar repeated queries."
        ),
    },
    {
        "filename": "vector_databases_overview.pdf",
        "content": (
            "Vector databases store high-dimensional embeddings for similarity search. "
            "Common vector databases include ChromaDB, Pinecone, Weaviate, and Qdrant. "
            "ChromaDB is an open-source, local-first vector store suited for development. "
            "HNSW (Hierarchical Navigable Small World) is the dominant ANN index algorithm. "
            "ANN stands for Approximate Nearest Neighbor search. "
            "HNSW builds a layered graph where each node connects to its nearest neighbors. "
            "Query time scales as O(log n) with HNSW for typical workloads. "
            "Embedding models convert text into dense floating-point vectors. "
            "The nomic-embed-text model produces 768-dimensional embeddings. "
            "Cosine similarity measures the angle between two vectors in embedding space. "
            "A cosine similarity of 1.0 means vectors point in identical directions. "
            "Production vector stores support metadata filtering alongside ANN search. "
            "Re-indexing is required when the embedding model is changed or updated."
        ),
    },
    {
        "filename": "llm_evaluation_methods.pdf",
        "content": (
            "Evaluating large language models requires multiple complementary metrics. "
            "Faithfulness measures whether a generated answer is supported by the source. "
            "Relevance measures how well the answer addresses the original question. "
            "Fluency assesses grammatical correctness and natural phrasing. "
            "RAGAS is a popular framework for automated RAG evaluation. "
            "Human evaluation remains the gold standard for nuanced quality assessment. "
            "The BLEU score measures n-gram overlap between generated and reference text. "
            "BERTScore uses contextual embeddings for semantic similarity measurement. "
            "Hallucination detection involves checking claims against source documents. "
            "LLM-as-judge approaches use a second model to score answers. "
            "Calibration ensures model confidence correlates with actual accuracy. "
            "Benchmark leaderboards like MMLU and HellaSwag measure general capability. "
            "Domain-specific evaluation sets are essential for production deployments."
        ),
    },
]


# ── Minimal valid PDF writer (no external library needed) ─────────────────────

def _make_pdf(content: str) -> bytes:
    """
    Generate a minimal but valid PDF containing `content` as plain text.
    Uses only stdlib — no reportlab, fpdf, or pypdf write support required.
    Characters outside Latin-1 are replaced with '?' to stay in PDF Type1 range.
    """
    safe = (
        content
        .encode("latin-1", errors="replace")
        .decode("latin-1")
        .replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )

    # Split into lines of ~90 chars to prevent PDF stream overflow
    words = safe.split()
    lines, current = [], []
    for word in words:
        if sum(len(w) + 1 for w in current) + len(word) > 90:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))

    # Build text stream: BT ... Td for each line
    stream_parts = ["BT", "/F1 11 Tf", "50 750 Td", "14 TL"]
    for line in lines:
        stream_parts.append(f"({line}) Tj T*")
    stream_parts.append("ET")
    stream = "\n".join(stream_parts)
    stream_bytes = stream.encode("latin-1")

    # PDF object bodies
    catalog  = b"<< /Type /Catalog /Pages 2 0 R >>"
    pages    = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"
    page     = (
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n"
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
    )
    content_obj = (
        f"<< /Length {len(stream_bytes)} >>\nstream\n".encode()
        + stream_bytes
        + b"\nendstream"
    )
    font = (
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
        b"/Encoding /WinAnsiEncoding >>"
    )

    objects = [catalog, pages, page, content_obj, font]

    # Assemble PDF
    buf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj_body in enumerate(objects, 1):
        offsets.append(len(buf))
        header = f"{i} 0 obj\n".encode()
        buf += header + obj_body + b"\nendobj\n"

    xref_offset = len(buf)
    buf += b"xref\n"
    buf += f"0 {len(objects) + 1}\n".encode()
    buf += b"0000000000 65535 f \n"
    for off in offsets:
        buf += f"{off:010d} 00000 n \n".encode()

    buf += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode()

    return bytes(buf)


# ── Download helpers ──────────────────────────────────────────────────────────

def _download(url: str, dest: Path, timeout: int = 30) -> bool:
    if not HAS_REQUESTS:
        return False
    try:
        print(f"  Downloading {url[:70]}...")
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (RAG-test-downloader/1.0)"},
            timeout=timeout,
            stream=True,
        )
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "pdf" not in content_type and not url.endswith(".pdf"):
            print(f"  Warning: content-type '{content_type}' may not be PDF")
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        size_kb = dest.stat().st_size // 1024
        print(f"  Saved {dest.name} ({size_kb} KB)")
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Populating {DOCS_DIR.resolve()} with test documents...\n")

    downloaded, synthesized, skipped = 0, 0, 0

    # ── Real PDFs ──────────────────────────────────────────────────────────────
    print("── Real PDFs (downloading) ──────────────────────────────")
    for filename, url, desc in REAL_PDFS:
        dest = DOCS_DIR / filename
        if dest.exists():
            print(f"  Skipping {filename} (already exists)")
            skipped += 1
            continue
        print(f"\n{desc}")
        if _download(url, dest):
            downloaded += 1
        else:
            print(f"  Download failed — generating synthetic replacement: {filename}")
            # Use first synthetic doc as fallback content
            fallback_content = (
                f"This document ({filename}) could not be downloaded. "
                "It is a placeholder for testing purposes. "
                "The retrieval system should handle this gracefully."
            )
            dest.write_bytes(_make_pdf(fallback_content))
            synthesized += 1
        time.sleep(0.5)   # polite delay between requests

    # ── Synthetic PDFs ─────────────────────────────────────────────────────────
    print("\n── Synthetic PDFs (generating) ──────────────────────────")
    for doc in SYNTHETIC_DOCS:
        dest = DOCS_DIR / doc["filename"]
        if dest.exists():
            print(f"  Skipping {doc['filename']} (already exists)")
            skipped += 1
            continue
        pdf_bytes = _make_pdf(doc["content"])
        dest.write_bytes(pdf_bytes)
        size_kb = len(pdf_bytes) // 1024
        print(f"  Generated {doc['filename']} ({size_kb} KB)")
        synthesized += 1

    # ── Summary ────────────────────────────────────────────────────────────────
    all_pdfs = list(DOCS_DIR.glob("*.pdf"))
    print(f"\n{'─' * 52}")
    print(f"Done. {DOCS_DIR} now contains {len(all_pdfs)} PDFs:")
    for p in sorted(all_pdfs):
        print(f"  {p.stat().st_size // 1024:5d} KB  {p.name}")
    print(f"\nDownloaded: {downloaded}  |  Synthesized: {synthesized}  |  Skipped: {skipped}")
    print("\nRun main.py to ingest all documents into the RAG system.")


if __name__ == "__main__":
    main()