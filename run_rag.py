"""
run_rag.py — Local command-line runner for ProductionRAGPipeline.

Run from the project root:

    python run_rag.py

The script:
1. Creates the production RAG pipeline.
2. Runs async setup (PDF ingestion, BM25 index, reranker).
3. Opens an interactive question loop.
4. Prints answers, sources, and useful retrieval metrics.

This file is intentionally a local runner and is NOT added to GitHub.
"""

import asyncio
import logging
import sys

from config import RAGConfig
from pipeline import ProductionRAGPipeline


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


async def build_pipeline() -> ProductionRAGPipeline:
    print("\n" + "=" * 70)
    print("  Production RAG Pipeline")
    print("=" * 70)
    print("\nStarting setup...")
    print("This may take a while on the first run because models may need")
    print("to be loaded/downloaded and PDFs may need to be ingested.\n")

    cfg = RAGConfig()
    pipeline = ProductionRAGPipeline(cfg)

    try:
        setup_result = await pipeline.setup()
    except Exception:
        logging.exception("\nPipeline setup failed.")
        print("\nCheck the error above before trying again.")
        print("Common causes:")
        print("  - Ollama is not running")
        print("  - Required Ollama models are missing")
        print("  - A Python dependency is missing")
        print("  - A PDF/database/vector-store configuration is invalid")
        raise

    ingested = setup_result.get("ingested_files", [])
    print("\nSetup complete.")

    if ingested:
        print("\nIngestion summary:")
        for item in ingested:
            print(f"  • {item}")
    else:
        print("No new PDF ingestion was reported.")

    return pipeline


def print_result(result: dict) -> None:
    print("\n" + "-" * 70)
    print("ANSWER")
    print("-" * 70)
    print(result.get("answer", "(no answer returned)"))

    sources = result.get("sources", [])
    print("\n" + "-" * 70)
    print("SOURCES")
    print("-" * 70)

    if not sources:
        print("No sources returned.")
    else:
        for i, source in enumerate(sources, start=1):
            filename = source.get("filename", "unknown")
            page = source.get("page", 0)
            score = source.get("rerank_score", 0.0)
            print(f"{i}. {filename} | page {page} | rerank={score}")

    metrics = result.get("metrics", {})
    if metrics:
        print("\n" + "-" * 70)
        print("METRICS")
        print("-" * 70)
        for key, value in metrics.items():
            print(f"{key}: {value}")

    rewritten = result.get("rewritten_query")
    if rewritten:
        print(f"\nRewritten query: {rewritten}")

    if result.get("from_cache"):
        print("Answer source: ANSWER CACHE")

    if result.get("drift_alert"):
        print(f"\nDRIFT ALERT: {result['drift_alert']}")

    print("-" * 70)


async def main() -> None:
    configure_logging()

    try:
        pipeline = await build_pipeline()
    except Exception:
        sys.exit(1)

    print(
        "\nEnter a question to query your documents."
        "\nCommands:"
        "\n  /quit     Exit"
        "\n  /exit     Exit"
        "\n  /help     Show commands"
        "\n"
    )

    while True:
        try:
            question = input("RAG> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting.")
            break

        if not question:
            continue

        command = question.lower()

        if command in {"/quit", "/exit"}:
            print("Goodbye.")
            break

        if command == "/help":
            print(
                "\nCommands:"
                "\n  /quit or /exit  Exit the program"
                "\n  /help           Show this help"
                "\n\nAny other text is treated as a RAG question."
            )
            continue

        print("\nSearching and generating answer...")

        try:
            result = pipeline.query(
                question,
                metadata_filter=None,
                use_web_fallback=True,
            )
            print_result(result)
        except Exception:
            logging.exception("\nQuery failed.")
            print(
                "\nThe pipeline encountered an error. "
                "The program is still running; you can try another question."
            )


if __name__ == "__main__":
    asyncio.run(main())
