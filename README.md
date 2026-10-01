# Multimodal Production-Oriented RAG System

A local-first Retrieval-Augmented Generation (RAG) system built from the ground up in Python. It combines hybrid retrieval, hierarchical chunking, multimodal document understanding, conversational memory, grounded generation, and automated response validation.

The system is designed to go beyond a basic **vector database + LLM** pipeline by treating retrieval quality, document structure, visual information, context management, and answer reliability as separate engineering problems.

## Architecture

```text
                              User Query
                                  │
                                  ▼
                    Conversation Context / Memory
                                  │
                                  ▼
                     Query Classification + Rewrite
                                  │
                                  ▼
                        Retrieval Planning
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
              BM25 Retrieval             Dense Retrieval
                    │                      ChromaDB
                    └─────────────┬─────────────┘
                                  ▼
                         RRF Rank Fusion
                                  │
                                  ▼
                       Cross-Encoder Rerank
                                  │
                                  ▼
                       Context Expansion
                                  │
                                  ▼
                         Answerability Check
                                  │
                         ┌────────┴────────┐
                         ▼                 ▼
                      Local           Web Retrieval
                      Evidence         / Fallback
                         │                 │
                         └────────┬────────┘
                                  ▼
                         Text + Visual Context
                                  │
                                  ▼
                         Ollama LLM / VLM
                                  │
                                  ▼
                       Critic + Grounding Check
                                  │
                         ┌────────┴────────┐
                         ▼                 ▼
                        Pass          Repair / Abstain
                                  │
                                  ▼
                             Final Answer
```

## Key Features

### Hybrid Retrieval
Combines **BM25 lexical search** and **dense vector retrieval**, then merges their rankings with **Reciprocal Rank Fusion (RRF)** before cross-encoder reranking.

### Hierarchical Semantic Chunking
Documents are split into retrieval-sized child chunks while preserving larger parent chunks and neighboring context for generation.

### Multimodal RAG
PDF pages containing meaningful images, tables, charts, or dense vector drawings are rendered and processed by a vision-language model. The VLM produces:

- Image captions
- Markdown representations of visible content and tables

These textual representations are embedded for retrieval while the original rendered images are stored separately and supplied to the vision-capable LLM when visual evidence is relevant.

```text
PDF Page
   │
   ├── Text ───────────────► Semantic Chunking ──► Vector Store
   │
   └── Image / Table
            │
            ▼
       Page Rendering
            │
            ▼
       Vision-Language Model
            │
       ┌────┴────┐
       ▼         ▼
    Caption   Markdown
       │         │
       └────┬────┘
            ▼
       Vector Store
            │
            └────► Original Image ──► Multimodal Generation
```

### Conversational Memory
Short-term conversation history is persisted in SQLite and used during query rewriting and retrieval, allowing follow-up questions to retain context from previous turns.

### Adaptive Query Processing
Queries are classified and rewritten before retrieval, allowing the pipeline to adapt retrieval behavior for research, comparisons, troubleshooting, recommendations, and other query types.

### Grounded Generation & Repair
The pipeline evaluates whether retrieved evidence is sufficient before generation. Generated responses are then checked for unsupported claims and can be repaired or rejected when grounding fails.

### Production-Oriented Infrastructure
- Asynchronous PDF ingestion and batched embeddings
- Hash-based incremental document updates
- Persistent ChromaDB + SQLite storage
- Retrieval and answer caching
- Web retrieval with source-quality filtering
- Retrieval, latency, and grounding metrics
- Deterministic multimodal pipeline tests

## Document Ingestion

```text
PDF
 │
 ├── Text ──► Hierarchical Chunking ──► Embeddings ──► ChromaDB / BM25
 │
 └── Visual Pages ──► VLM Caption + Markdown ──► Embeddings ──► ChromaDB
                                  │
                                  └── Original Image ──► Image Store
```

Documents are identified by file hash so unchanged files can be skipped and modified documents can be re-indexed without rebuilding the entire corpus.

## Tech Stack

- **Python**
- **LangChain**
- **Ollama** — LLM, embedding, and vision inference
- **ChromaDB** — vector retrieval
- **SQLite** — chunks, images, conversations, cache, and metrics
- **BM25 / rank_bm25** — lexical retrieval
- **Sentence Transformers** — cross-encoder reranking
- **PyMuPDF / PyPDF** — PDF processing and page rendering
- **BeautifulSoup / DDGS** — web retrieval

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/Arnav-W-Coder/AI_Agent.git
cd AI_Agent
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

```bash
pip install -r requirements.txt
```

### 2. Install Ollama models

The default local models are:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
ollama pull llama3.2-vision
```

### 3. Add documents

Place PDF files in the configured `docs/` directory. The ingestion pipeline automatically processes new or modified documents and indexes both textual and visual evidence.

### 4. Run

```bash
python main.py
```

## Project Structure

```text
AI_Agent/
├── config.py          # Central configuration
├── pipeline.py        # RAG orchestration
├── ingestion.py       # PDF + multimodal ingestion
├── multimodal.py      # Page extraction and VLM captioning
├── chunking.py        # Hierarchical semantic chunking
├── retrieval.py       # Hybrid retrieval + reranking
├── rewriter.py        # Query rewriting
├── memory.py          # SQLite conversation memory
├── critic.py          # Grounding and response repair
├── cache.py           # Retrieval / answer caching
├── metrics.py         # Observability and metrics
├── db.py              # SQLite persistence
├── web_store.py       # Web evidence retrieval
└── main.py            # Application entry point
```

## Project Goals

This project explores how to build RAG systems beyond the basic **"vector database + LLM"** pattern, with an emphasis on:

- High-quality hybrid retrieval
- Preserving document and visual context
- Conversational query understanding
- Evidence-aware generation
- Hallucination detection and repair
- Local inference and data ownership
- Measurable, maintainable AI infrastructure

Built independently as an AI engineering project.
