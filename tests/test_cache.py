"""Tests for the exact-match query cache."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cache import QueryCache  # noqa: E402


def test_miss_then_hit():
    cache: QueryCache = QueryCache()
    assert cache.get("what is 2+2?") is None
    cache.put("what is 2+2?", {"answer": "4"})
    assert cache.get("what is 2+2?") == {"answer": "4"}


def test_normalization_collides_casing_and_whitespace():
    cache: QueryCache = QueryCache()
    cache.put("What   is  2+2?", {"answer": "4"})
    # Different spacing / casing must hit the same entry.
    assert cache.get("what is 2+2?") == {"answer": "4"}


def test_lru_eviction():
    cache: QueryCache = QueryCache(max_entries=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")  # touch 'a' so 'b' is now least-recently used
    cache.put("c", 3)  # evicts 'b'
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.get("b") is None


def test_stats_track_hits_and_misses():
    cache: QueryCache = QueryCache()
    cache.get("x")  # miss
    cache.put("x", 1)
    cache.get("x")  # hit
    assert cache.stats.misses == 1
    assert cache.stats.hits == 1
    assert cache.stats.hit_rate == 0.5
