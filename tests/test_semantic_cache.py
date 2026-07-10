"""Tests for the semantic cache (MiniLM + faiss).

Skipped entirely when the embedding stack isn't installed — the exact-match
fallback is covered by test_cache.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("sentence_transformers")
pytest.importorskip("faiss")

from app.cache import SemanticCache  # noqa: E402


@pytest.fixture(scope="module")
def warm_cache_cls():
    """Trigger the one-off encoder load so per-test timings stay honest."""
    c: SemanticCache = SemanticCache()
    c.put("warmup", 0)
    return SemanticCache


def test_exact_duplicate_hits(warm_cache_cls):
    cache: SemanticCache = warm_cache_cls()
    cache.put("What is the capital of France?", {"answer": "Paris"})
    assert cache.get("what  is the capital of france?") == {"answer": "Paris"}


def test_different_questions_do_not_collide(warm_cache_cls):
    # The whole point of the conservative 0.90 threshold: France vs Germany
    # are near-duplicates lexically but different questions semantically.
    cache: SemanticCache = warm_cache_cls()
    cache.put("What is the capital of France?", {"answer": "Paris"})
    assert cache.get("What is the capital of Germany?") is None


def test_paraphrase_hits_above_threshold(warm_cache_cls):
    cache: SemanticCache = warm_cache_cls()
    cache.similarity_threshold = 0.75  # loosened deliberately for this test
    cache.put("What is the capital of France?", {"answer": "Paris"})
    hit = cache.get("What's the capital city of France?")
    assert hit == {"answer": "Paris"}
    assert cache.stats.semantic_hits == 1


def test_lru_eviction_rebuilds_index(warm_cache_cls):
    cache: SemanticCache = warm_cache_cls()
    cache.max_entries = 2
    cache.put("first question about dogs", 1)
    cache.put("second question about cats", 2)
    cache.put("third question about birds", 3)  # evicts the dogs entry
    assert len(cache) == 2
    assert cache.get("first question about dogs") is None
    assert cache.get("third question about birds") == 3


def test_semantic_miss_below_threshold(warm_cache_cls):
    cache: SemanticCache = warm_cache_cls()
    cache.put("Explain the theory of relativity", "physics")
    assert cache.get("What is the best pizza topping?") is None
    assert cache.stats.misses >= 1
