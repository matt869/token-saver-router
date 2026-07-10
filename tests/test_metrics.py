"""Tests for the metrics registry and the price table."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.metrics import MetricsRegistry, RequestRecord  # noqa: E402
from app.pricing import price_per_1m  # noqa: E402


def test_price_table_and_local_is_free():
    assert price_per_1m("accounts/fireworks/models/gemma-2-9b-it") == 0.20
    assert price_per_1m("accounts/fireworks/models/llama-v3p1-70b-instruct") == 0.90
    # Unknown Fireworks model -> default; local HF id -> free.
    assert price_per_1m("accounts/fireworks/models/unknown", default=0.5) == 0.5
    assert price_per_1m("Qwen/Qwen2.5-7B-Instruct") == 0.0
    assert price_per_1m("") == 0.0


def test_preflight_saving_priced_at_main_model():
    reg = MetricsRegistry(main_model="accounts/fireworks/models/gemma-2-9b-it")
    rec = reg.record(RequestRecord(
        route="remote", model_routed_to="accounts/fireworks/models/gemma-2-9b-it",
        tokens_saved_preflight=1000, remote_tokens_spent=500,
    ))
    # 1000 tokens * $0.20 / 1e6
    assert rec.est_cost_saved_usd == 0.0002
    snap = reg.snapshot()
    assert snap["layers"]["compression"]["tokens_saved"] == 1000
    assert snap["totals"]["est_cost_spent_usd"] == 0.0001  # 500 * 0.20/1e6


def test_cache_hit_split_and_stats_are_sourced_from_cachestats():
    reg = MetricsRegistry(main_model="accounts/fireworks/models/gemma-2-9b-it")
    reg.record(RequestRecord(route="cache", model_routed_to="cache",
                             cache_hit_type="exact", tokens_avoided_cache=120))
    reg.record(RequestRecord(route="cache", model_routed_to="cache",
                             cache_hit_type="semantic", tokens_avoided_cache=80))

    class FakeStats:
        hits, misses, semantic_hits = 2, 3, 1
        hit_rate = 0.4

    snap = reg.snapshot(cache_stats=FakeStats())
    assert snap["layers"]["exact_cache"]["hits"] == 1
    assert snap["layers"]["semantic_cache"]["tokens_avoided"] == 80
    # Hit rates come straight from CacheStats, not re-counted.
    assert snap["cache"]["hit_rate"] == 0.4
    assert snap["cache"]["exact_hits"] == 1
    assert snap["cache"]["semantic_hits"] == 1


def test_routing_saving_is_full_price_local_vs_delta_cheap():
    reg = MetricsRegistry(
        main_model="accounts/fireworks/models/llama-v3p1-70b-instruct",  # 0.90
        cheap_model="accounts/fireworks/models/llama-v3p2-3b-instruct",  # 0.10
    )
    # Local answer avoids the full main-model price.
    local = reg.record(RequestRecord(route="local", model_routed_to="Qwen/local",
                                     routing_tier="local", est_tokens_avoided_routing=1000))
    assert local.est_cost_saved_usd == 0.0009  # 1000 * 0.90/1e6
    # Cheap-remote answer only saves the price delta (0.90 - 0.10).
    cheap = reg.record(RequestRecord(
        route="cheap-remote", model_routed_to="accounts/fireworks/models/llama-v3p2-3b-instruct",
        routing_tier="cheap_remote", est_tokens_avoided_routing=1000, remote_tokens_spent=1000,
    ))
    assert cheap.est_cost_saved_usd == 0.0008  # 1000 * (0.90-0.10)/1e6
