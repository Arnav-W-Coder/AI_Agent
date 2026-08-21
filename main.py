# Agentic RAG — Revision Loop + Auto Web Scraping + Source Reliability
# Stack: LangChain, ChromaDB, Ollama, BeautifulSoup, DuckDuckGo
#
# pip install langchain langchain-ollama langchain-chroma langchain-community
#             langchain-text-splitters chromadb pypdf requests beautifulsoup4
#             ddgs

import mimetypes
mimetypes.add_type("application/pdf", ".pdf")  # Windows Store Python fix

# ── Imports ───────────────────────────────────────────────────────────────────

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.documents import Document
from langchain.agents import create_agent

import requests
import time
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from ddgs import DDGS

# ── Config ────────────────────────────────────────────────────────────────────

EMBED_MODEL      = "nomic-embed-text"
LLM_MODEL        = "llama3.2"
CHROMA_DIR       = "./chroma_db"
DOCS_DIR         = "./docs"
CTX_WINDOW       = 16384
TOP_K_CHUNKS     = 4
MAX_REVISIONS    = 2
MAX_SCRAPE_URLS  = 5      # fetch more candidates so filtering has room to work
TOP_WEB_CHUNKS   = 6      # how many web chunks to retrieve after embedding
DDG_RETRIES      = 3
MIN_DOMAIN_SCORE = 30     # skip URLs scoring below this (0–100)

UNCERTAINTY_PHRASES = [
    "i don't have enough information",
    "i don't know",
    "i cannot find",
    "i was unable to find",
    "the documents do not",
    "the context does not",
    "no information",
    "not mentioned",
    "not present in",
    "cannot answer",
    "not provided",
]

# ── Domain reliability scoring ────────────────────────────────────────────────
#
# Score 0–100. URLs below MIN_DOMAIN_SCORE are skipped before scraping.
# Add your own trusted/blocked domains here.

DOMAIN_SCORES = {
    # === Tier 1: Primary sources (90–100) ===
    "wikipedia.org":         95,
    "britannica.com":        90,
    # Government & education
    "*.gov":                 95,
    "*.edu":                 90,
    # Major tech documentation
    "docs.python.org":       95,
    "developer.mozilla.org": 92,
    "arxiv.org":             92,
    "pubmed.ncbi.nlm.nih.gov": 95,
    # === Tier 2: Reputable publishers (70–89) ===
    "aws.amazon.com":        85,
    "cloud.google.com":      85,
    "learn.microsoft.com":   85,
    "research.ibm.com":      88,
    "openai.com":            82,
    "anthropic.com":         85,
    "langchain.com":         80,
    "huggingface.co":        80,
    "towardsdatascience.com":72,
    "medium.com":            65,
    "stackoverflow.com":     75,
    "github.com":            78,
    # === Tier 3: Generally ok (50–69) ===
    "reddit.com":            50,
    "quora.com":             48,
    "linkedin.com":          55,
    # === Blocked / low quality (0) ===
    "pinterest.com":          0,
    "facebook.com":           0,
    "twitter.com":            0,
    "x.com":                  0,
    "tiktok.com":             0,
    "fandom.com":            20,
    "grokipedia.com":        15,   # mirror/scraper site seen in your output
}

def _score_domain(url: str) -> tuple[int, str]:
    """
    Return (score 0–100, reason) for a URL.
    Checks exact domain, wildcard TLD patterns (.gov, .edu), then defaults to 60.
    """
    try:
        host = urlparse(url).netloc.lower()
        # Strip www.
        if host.startswith("www."):
            host = host[4:]
    except Exception:
        return 0, "unparseable URL"

    # Exact match
    if host in DOMAIN_SCORES:
        return DOMAIN_SCORES[host], f"known domain ({host})"

    # Subdomain match — e.g. research.ibm.com against ibm.com
    parts = host.split(".")
    for i in range(len(parts) - 1):
        parent = ".".join(parts[i:])
        if parent in DOMAIN_SCORES:
            return DOMAIN_SCORES[parent], f"subdomain of {parent}"

    # Wildcard TLD — *.gov, *.edu
    tld = "." + parts[-1] if parts else ""
    wildcard = f"*{tld}"
    if wildcard in DOMAIN_SCORES:
        return DOMAIN_SCORES[wildcard], f"trusted TLD ({tld})"

    # Default: unknown but not blocked
    return 60, "unknown domain (default score)"


def _filter_and_rank_urls(results: list[dict]) -> list[dict]:
    """
    Score, filter, and rank DDG results by domain reliability.
    Attaches score and reason to each result for debug output.
    """
    scored = []
    for r in results:
        score, reason = _score_domain(r["href"])
        r["_score"]  = score
        r["_reason"] = reason
        scored.append(r)

    scored.sort(key=lambda r: r["_score"], reverse=True)

    print("\n[URL Scoring] All candidates:")
    for r in scored:
        status = "✓ KEEP" if r["_score"] >= MIN_DOMAIN_SCORE else "✗ SKIP"
        print(f"  [{r['_score']:3d}] {status} — {r['href']}")
        print(f"         Reason: {r['_reason']}")

    filtered = [r for r in scored if r["_score"] >= MIN_DOMAIN_SCORE]
    if not filtered:
        print("[URL Scoring] No URLs passed the reliability threshold.")
    return filtered


# ── Step 1: Ingest PDFs ───────────────────────────────────────────────────────

print("Loading and indexing documents...")

loader    = DirectoryLoader(DOCS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
splitter  = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
chunks    = splitter.split_documents(documents)

embeddings  = OllamaEmbeddings(model=EMBED_MODEL)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_CHUNKS})

print(f"Indexed {len(chunks)} chunks.\n")

# ── Step 2: LLM ──────────────────────────────────────────────────────────────

llm = ChatOllama(model=LLM_MODEL, num_ctx=CTX_WINDOW)

# ── Step 3: Tools ─────────────────────────────────────────────────────────────

@tool
def search_documents(query: str) -> str:
    """Search the local knowledge base (indexed PDFs) for relevant information.
    Use this when the question is about documents the user has provided."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant documents found in the knowledge base."
    return "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{d.page_content}" for i, d in enumerate(docs)
    )


def _scrape_url(url: str, char_limit: int = 3000) -> str:
    """Fetch, clean, and return text from one URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        main = soup.find("main") or soup.find("article") or soup.body
        text = main.get_text(separator="\n", strip=True) if main else ""
        if len(text) > char_limit:
            text = text[:char_limit] + "\n[...truncated...]"
        return text if text else "Could not extract readable content."
    except requests.exceptions.Timeout:
        return "Error: request timed out."
    except requests.exceptions.HTTPError as e:
        return f"Error: HTTP {e.response.status_code}."
    except Exception as e:
        return f"Error: {str(e)}"


def _embed_web_content(scraped: list[dict], query: str) -> str:
    """
    FIX: Instead of dumping all scraped text in order (which biases the LLM
    toward Source 1), chunk every source, embed them all into a temporary
    in-memory Chroma collection, then similarity-search for the most relevant
    chunks across ALL sources.

    Returns the top TOP_WEB_CHUNKS chunks, potentially from different sources.
    """
    if not scraped:
        return "No web content available."

    # Build LangChain Documents from scraped pages
    raw_docs = []
    for item in scraped:
        raw_docs.append(Document(
            page_content=item["text"],
            metadata={"source": item["url"], "title": item["title"]}
        ))

    # Chunk each page so long articles don't hog the context
    web_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=100
    )
    web_chunks = web_splitter.split_documents(raw_docs)

    if not web_chunks:
        return "No content could be extracted from web sources."

    # Embed into a temporary in-memory collection (no persist_directory)
    temp_store = Chroma.from_documents(
        documents=web_chunks,
        embedding=embeddings,    # same embedding model as the PDF store
    )

    # Similarity search — pulls relevant chunks from whichever source has them
    relevant = temp_store.similarity_search(query, k=TOP_WEB_CHUNKS)

    # Format with source attribution so the LLM (and critic) can trace claims
    parts = []
    for i, doc in enumerate(relevant, 1):
        title = doc.metadata.get("title", "Unknown")
        url   = doc.metadata.get("source", "Unknown")
        parts.append(
            f"[Web Chunk {i} | {title}]\nURL: {url}\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(parts)


@tool
def search_and_scrape(query: str) -> str:
    """Search the web for a query, score URLs for reliability, scrape the best
    sources, and return the most relevant content from across all of them.
    Use this for any question needing current or general web information.
    Pass a plain English query — URLs are found and filtered automatically."""

    print(f"\n[Web Search] Querying DuckDuckGo: '{query}'")

    # ── DDG search with backoff ───────────────────────────────────────────────
    raw_results = []
    for attempt in range(DDG_RETRIES):
        try:
            with DDGS() as ddgs:
                raw_results = list(ddgs.text(query, max_results=MAX_SCRAPE_URLS))
            if raw_results:
                break
            print(f"  [DDG] No results on attempt {attempt+1}, retrying...")
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"  [DDG] Attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)

    if not raw_results:
        return "Web search returned no results after retries."

    # ── Score and filter URLs ─────────────────────────────────────────────────
    filtered = _filter_and_rank_urls(raw_results)
    if not filtered:
        return (
            "All candidate URLs were below the reliability threshold. "
            "No trustworthy sources found for this query."
        )

    # ── Scrape each approved URL ──────────────────────────────────────────────
    scraped = []
    for r in filtered:
        url, title = r["href"], r.get("title", "Untitled")
        print(f"[Scraping | score={r['_score']}] {url}")
        text = _scrape_url(url)
        if not text.startswith("Error:"):
            scraped.append({"url": url, "title": title, "text": text})
        else:
            print(f"  → {text}")

    if not scraped:
        return "All approved URLs failed to scrape."

    # ── Embed all sources, retrieve best chunks across all of them ────────────
    print(f"\n[Embedding] {sum(1 for _ in scraped)} sources → similarity search "
          f"for top {TOP_WEB_CHUNKS} chunks across all...")
    return _embed_web_content(scraped, query)


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic math expression, e.g. '12.36 + 8.92' or '21.28 * 2'.
    Only use this for arithmetic — do not pass arbitrary code."""
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: Only basic arithmetic is allowed."
        return str(round(eval(expression, {"__builtins__": {}}), 4))
    except Exception as e:
        return f"Error: {e}"


# ── Step 4: Critic ───────────────────────────────────────────────────────────
#
# Improvement: instead of binary GROUNDED/HALLUCINATED, ask the critic to
# list the SPECIFIC unsupported claims. This lets the revision prompt tell
# the agent exactly what it got wrong, rather than "you hallucinated, try again."

CLAIM_CRITIC_PROMPT = ChatPromptTemplate.from_template("""
You are a strict fact-checker. Compare the answer against the retrieved context.

RETRIEVED CONTEXT:
{context}

AGENT ANSWER:
{answer}

Task: Find every specific claim in the answer that CANNOT be traced word-for-word
or by clear paraphrase to the retrieved context above.

Rules:
- List unsupported claims as short bullet points. Be specific (quote the claim).
- If ALL claims are supported, reply with exactly: FULLY_GROUNDED
- If there are unsupported claims, start your reply with UNSUPPORTED_CLAIMS:
  followed by the bullet list. Do not add any other text.
- General summary statements that broadly reflect the context are acceptable.
  Only flag claims that introduce NEW specific facts (names, numbers, dates,
  organisations) not present in the context.
""")

claim_critic_chain = CLAIM_CRITIC_PROMPT | llm | StrOutputParser()


def _is_uncertainty_response(answer: str) -> bool:
    lower = answer.lower()
    return any(phrase in lower for phrase in UNCERTAINTY_PHRASES)


def _run_critic(context: str, answer: str) -> tuple[str, str]:
    """
    Returns (verdict, unsupported_claims_text).
    verdict = 'GROUNDED' | 'HALLUCINATED'
    unsupported_claims_text = bullet list of bad claims, or '' if grounded.
    """
    # Stage 1: no tools called
    if not context:
        print("[Critic] No tool context — HALLUCINATED.")
        return "HALLUCINATED", "No tools were called. Answer came from model memory."

    # Stage 2: honest "I don't know"
    if _is_uncertainty_response(answer):
        print("[Critic] Uncertainty admission — auto GROUNDED.")
        return "GROUNDED", ""

    # Stage 3: LLM claim-level check
    raw = claim_critic_chain.invoke({"context": context, "answer": answer}).strip()

    if "FULLY_GROUNDED" in raw.upper():
        print("[Critic] FULLY GROUNDED.")
        return "GROUNDED", ""

    if "UNSUPPORTED_CLAIMS" in raw.upper():
        claims = raw.replace("UNSUPPORTED_CLAIMS:", "").strip()
        print(f"[Critic] HALLUCINATED. Unsupported claims:\n{claims}")
        return "HALLUCINATED", claims

    # Unexpected format — be lenient
    print(f"[Critic] Unexpected response '{raw[:80]}' — defaulting GROUNDED.")
    return "GROUNDED", ""


# ── Step 5: Revision loop ────────────────────────────────────────────────────

BASE_SYSTEM_PROMPT = (
    "You are a precise research assistant. "
    "Use search_documents for questions about local PDF documents. "
    "Use search_and_scrape for any question needing live or general web knowledge "
    "— pass a plain English query, URLs are found and scored automatically. "
    "Use calculator only for arithmetic. "
    "Only state facts explicitly present in what the tools return. "
    "Do not introduce names, numbers, organisations, or dates that you do not "
    "see in the tool output. "
    "If the tools do not contain the answer, say: "
    "'I don't have enough information to answer this confidently.'"
)


def _build_retry_prompt(unsupported_claims: str) -> str:
    """
    FIX: Tell the agent exactly which claims were wrong, not just 'you hallucinated'.
    This gives the model something actionable to correct on the next attempt.
    """
    return (
        BASE_SYSTEM_PROMPT +
        f"\n\nWARNING — your previous answer contained claims not found in the "
        f"retrieved context. The fact-checker identified these specific problems:\n"
        f"{unsupported_claims}\n\n"
        "Please search again and answer using ONLY what the tools explicitly return. "
        "If you cannot find support for a claim, leave it out entirely."
    )


def run_with_revision(question: str) -> str:
    """Run agent → critique → targeted retry, up to MAX_REVISIONS times."""

    last_answer      = ""
    last_context     = ""
    unsupported      = ""

    for attempt in range(MAX_REVISIONS + 1):
        label  = f"Attempt {attempt+1}/{MAX_REVISIONS+1}"
        prompt = BASE_SYSTEM_PROMPT if attempt == 0 else _build_retry_prompt(unsupported)

        agent = create_agent(
            model=llm,
            tools=[search_documents, search_and_scrape, calculator],
            system_prompt=prompt,
        )

        response = agent.invoke({
            "messages": [{"role": "user", "content": question}]
        })

        tool_outputs = [
            msg.content for msg in response["messages"]
            if isinstance(msg, ToolMessage)
        ]
        last_context = "\n\n".join(tool_outputs)
        last_answer  = response["messages"][-1].content

        if tool_outputs:
            print(f"\n[{label}] Tool outputs ({len(tool_outputs)} call(s)):")
            for i, out in enumerate(tool_outputs, 1):
                preview = out[:300] + ("..." if len(out) > 300 else "")
                print(f"  Tool {i}: {preview}")
        else:
            print(f"\n[{label}] No tools called.")

        print(f"\n[{label}] Draft answer:\n{last_answer}")

        verdict, unsupported = _run_critic(last_context, last_answer)
        print(f"[Critic] Verdict: {verdict}")

        if verdict == "GROUNDED":
            return last_answer

        if attempt < MAX_REVISIONS:
            print(f"[Revision] Retrying with targeted feedback...\n")

    print("[Revision] Max retries reached — returning safe fallback.")
    return (
        "I was unable to produce a fully grounded answer. "
        "The available sources may not contain the information needed."
    )


# ── Step 6: Run ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("TEST 1: Local document query")
    print("=" * 60)
    print(run_with_revision("What are the main topics covered in the documents?"))

    print("\n" + "=" * 60)
    print("TEST 2: Web search — multi-source RAG")
    print("=" * 60)
    print(run_with_revision(
        "What is retrieval-augmented generation and how does it work?"
    ))

    print("\n" + "=" * 60)
    print("TEST 3: Math requiring retrieval")
    print("=" * 60)
    print(run_with_revision(
        "What's the total cost mentioned in the docs, doubled?"
    ))