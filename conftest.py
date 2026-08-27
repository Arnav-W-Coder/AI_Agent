# conftest.py — registers custom pytest marks so they don't produce warnings.
import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "layer1: infrastructure tests — no LLM needed")
    config.addinivalue_line("markers", "layer2: component tests — Ollama must be running")
    config.addinivalue_line("markers", "e2e: full pipeline tests — Ollama + all deps required")