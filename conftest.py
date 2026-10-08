# conftest.py — registers custom pytest marks so they don't produce warnings.
import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "layer1: infrastructure tests — no LLM needed")
    config.addinivalue_line("markers", "layer2: component tests — Ollama must be running")
    config.addinivalue_line("markers", "e2e: full pipeline tests — Ollama + all deps required")
    config.addinivalue_line("markers", "multimodal: deterministic multimodal pipeline tests")
    config.addinivalue_line("markers", "embedding: embedding/vector-store contract tests")
    config.addinivalue_line("markers", "integration: component integration tests with external boundaries mocked")
    config.addinivalue_line("markers", "live_ollama: optional tests that require a running Ollama server")
