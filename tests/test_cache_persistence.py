"""Tests for cache persistence, normalize-before-cache, and the warm()/hit-info API."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cache import QueryCache, SemanticCache, SqliteStore, create_cache  # noqa: E402
from app.config import load_config  # noqa: E402
from app.router.agent import Agent  # noqa: E402
from tests.test_agent_layers import FakeLocal, FakeRemote  # noqa: E402


def test_get_with_info_reports_exact_hit():
    cache = QueryCache()
    cache.put("hello world", {"answer": "hi"})
    value, kind = cache.get_with_info("hello world")
    assert value == {"answer": "hi"}
    assert kind == "exact"
    assert cache.get_with_info("nope")[1] == "none"


def test_sqlite_persistence_replays_across_restart(tmp_path):
    db = str(tmp_path / "cache.db")

    class Cfg:
        cache_backend = "exact"
        cache_max_entries = 128
        cache_similarity_threshold = 0.95
        cache_persist_path = db

    c1 = create_cache(Cfg())
    c1.put("what is the capital of france", {"answer": "Paris", "remote_tokens": 40})

    # A fresh cache built on the same file replays the stored entry.
    c2 = create_cache(Cfg())
    assert c2.get("what is the capital of france") == {"answer": "Paris", "remote_tokens": 40}


def test_sqlite_store_prunes_to_limit(tmp_path):
    store = SqliteStore(str(tmp_path / "c.db"))
    for i in range(10):
        store.save(f"k{i}", f"q{i}", {"answer": i})
    kept = store.load(limit=3)
    assert len(kept) == 3  # only the newest 3 survive


def test_warm_returns_bool():
    # Exact cache has nothing to warm; semantic warms the encoder (or degrades).
    assert QueryCache().warm() is False
    assert isinstance(SemanticCache().warm(), bool)


def test_greeting_variant_hits_the_same_cache_entry():
    # Normalize-before-cache: a politeness-only variant collapses to the same
    # normalized prompt and hits the exact cache (no second remote call).
    cfg = load_config()
    remote = FakeRemote(tokens=77)
    agent = Agent(config=cfg, local_model=FakeLocal(), remote_model=remote,
                  cheap_remote_model=None)

    agent.route("Prove that the square root of 2 is irrational, step by step.")
    hit = agent.route("Hi there! Please prove that the square root of 2 is "
                      "irrational, step by step. Thanks!")
    assert hit.cached is True
    assert hit.cache_hit_type == "exact"
    assert len(remote.calls) == 1  # the second query never reached the remote
