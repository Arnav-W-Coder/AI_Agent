# Production-Oriented Local RAG System

A local-first Retrieval-Augmented Generation (RAG) system built from the ground up in Python. The project focuses on improving retrieval quality, reducing hallucinations, and making RAG pipelines more reliable and observable.

## Architecture

```text
                         User Query
                              │
                              ▼
                  Query Classification
                    + Query Rewriting
                              │
                              ▼
                    Retrieval Planning
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
              BM25 Search           Dense Retrieval
                 │                    (ChromaDB)
                 └────────────┬────────────┘
                              ▼
                    Reciprocal Rank Fusion
                              │
                              ▼
                     Cross-Encoder Rerank
                              │
                              ▼
                    Parent Context Expansion
                              │
                              ▼
                       Answerability
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
                 Sufficient        Web Fallback
                    │                   │
                    └─────────┬─────────┘
                              ▼
                       Local LLM (Ollama)
                              │
                              ▼
                     Critic / Grounding
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
                   Pass            Repair / Abstain
                    │                   │
                    └─────────┬─────────┘
                              ▼
                         Final Answer
```

## Key Features

### Hybrid Retrieval

Combines **BM25 lexical search** and **dense vector retrieval**, then merges their rankings using **Reciprocal Rank Fusion (RRF)**.

### Cross-Encoder Reranking

Retrieved candidates are scored using a cross-encoder to select the most relevant context before generation.

### Hierarchical Chunking

Documents are split into small retrieval-oriented child chunks while maintaining larger parent chunks for contextual generation.

### Adaptive Query Processing

Queries are classified and rewritten before retrieval, allowing different retrieval strategies for research, comparisons, troubleshooting, recommendations, and other query types.

### Grounded Generation

The LLM is constrained to retrieved evidence, with **answerability checks** used to determine whether enough information exists to answer confidently.

### Critic & Repair

Generated responses are evaluated for grounding, relevance, and completeness. Unsupported responses can be repaired using the retrieved evidence or rejected.

### Production-Oriented Infrastructure

Includes:

* Asynchronous PDF ingestion
* Batched embeddings
* Hash-based incremental document updates
* Retrieval and answer caching
* SQLite-backed metrics
* Retrieval and latency monitoring
* Optional web retrieval and fallback

## Document Pipeline

```text
PDF
 │
 ▼
Structure-Aware Chunking
 │
 ├── Parent Chunks ──► SQLite
 │
 └── Child Chunks
          │
          ▼
     Embeddings
          │
          ▼
       ChromaDB
          │
          └──► BM25 Index
```

Only retrieval-sized child chunks are indexed, while parent chunks preserve additional context for generation.

## Tech Stack

* **Python**
* **LangChain**
* **Ollama**
* **ChromaDB**
* **SQLite**
* **BM25 / rank_bm25**
* **Sentence Transformers**
* **PyMuPDF / PyPDF**
* **BeautifulSoup / DDGS**

## Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd AI_Agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install local models

This project uses Ollama for local LLM inference and embeddings.

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 5. Add documents

Place PDF files in the configured `docs/` directory.

The ingestion pipeline automatically detects new or modified documents and updates the index.

### 6. Run

Use the project's main entry point:

```bash
python main.py
```

## Project Structure

```text
AI_Agent/
├── config.py          # Configuration
├── pipeline.py        # RAG orchestration
├── ingestion.py       # Document ingestion
├── chunking.py        # Semantic chunking
├── retrieval.py       # Hybrid retrieval + reranking
├── rewriter.py        # Query rewriting
├── critic.py          # Grounding + repair
├── cache.py           # Caching
├── metrics.py         # Monitoring
├── db.py              # SQLite persistence
├── web_store.py       # Web retrieval
└── main.py            # Application entry point
```

## Project Goals

This project explores how to build a RAG system beyond the basic **"vector database + LLM"** architecture.

The primary focus is:

* Improving retrieval precision
* Preserving document context
* Detecting insufficient evidence
* Reducing unsupported generation
* Making RAG systems measurable and maintainable

Built independently as an AI engineering project.
