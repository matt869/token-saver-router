"""Cumulative savings metrics — one record per routed request.

The agent calls :meth:`MetricsRegistry.record` exactly once per request; the
registry turns those records into the running totals and the per-layer
breakdown served by ``GET /metrics`` and the ``/dashboard`` view:

* **compression** — real-tokenizer prompt tokens the pre-flight optimizer
  removed before a live remote call.
* **exact_cache / semantic_cache** — remote tokens a cache hit avoided
  spending (the stored cost of the original call).
* **routing** — an *estimate* of the main-remote tokens avoided by answering
  on a cheaper tier instead (proxied by the cheap call's own token count,
  hence the ``est_`` prefix). A local answer saves the full main-model price;
  a cheap-remote answer saves the price *delta* between the two models.

Dollar figures come from the per-model price table in ``app/pricing.py``
(fallback ``REMOTE_PRICE_PER_1M_USD``); they are estimates and labelled as
such.

Cache hit/miss *rates* are NOT re-counted here — ``CacheStats`` on the cache
backend already tracks them, and :meth:`snapshot` accepts that object and
reports its numbers directly so the two can never drift apart.

In-process only, same as the cache: totals reset on restart.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import asdict, dataclass
from typing import Deque, Dict, Optional

from app.pricing import price_per_1m
from app.tokens import counter_name

_RECENT_LIMIT = 50  # per-request rows kept for the dashboard table


@dataclass
class RequestRecord:
    """What one routed request cost and saved."""

    route: str  # RouteResult.route label ("local", "remote", "cache", ...)
    model_routed_to: str  # concrete model name that answered, or "cache"
    cache_hit_type: str = "none"  # none | exact | semantic
    prompt_tokens_before: int = 0  # real-tokenizer count of the raw query
    prompt_tokens_after: int = 0  # prompt tokens actually billed remotely (0 = no remote call)
    tokens_saved_preflight: int = 0
    tokens_avoided_cache: int = 0
    est_tokens_avoided_routing: int = 0
    routing_tier: str = ""  # "local" | "cheap_remote" when routing saved money
    remote_tokens_spent: int = 0  # tokens billed on model_routed_to
    est_cost_saved_usd: float = 0.0  # filled in by the registry from its prices


class MetricsRegistry:
    """Thread-safe accumulator for per-request savings records.

    ``main_model`` is the strong (expensive) remote model — the counterfactual
    every saving is priced against.
    """

    def __init__(
        self,
        main_model: str = "",
        cheap_model: str = "",
        default_price_per_1m: float = 0.20,
    ):
        self.main_model = main_model
        self.cheap_model = cheap_model
        self.default_price_per_1m = default_price_per_1m
        self.main_price = price_per_1m(main_model, default_price_per_1m) if main_model else default_price_per_1m
        self.cheap_price = price_per_1m(cheap_model, default_price_per_1m) if cheap_model else 0.0

        self._lock = threading.Lock()
        self._recent: Deque[Dict[str, object]] = deque(maxlen=_RECENT_LIMIT)
        self._requests = 0
        self._totals: Dict[str, float] = {
            "prompt_tokens_before": 0,
            "prompt_tokens_after": 0,
            "tokens_saved_preflight": 0,
            "tokens_avoided_cache": 0,
            "est_tokens_avoided_routing": 0,
            "remote_tokens_spent": 0,
            "est_cost_saved_usd": 0.0,
            "est_cost_spent_usd": 0.0,
        }
        # Per-layer contribution (tokens + $), cache split by hit type.
        self._layers: Dict[str, Dict[str, float]] = {
            "compression": {"tokens_saved": 0, "requests_touched": 0, "est_cost_saved_usd": 0.0},
            "exact_cache": {"tokens_avoided": 0, "hits": 0, "est_cost_saved_usd": 0.0},
            "semantic_cache": {"tokens_avoided": 0, "hits": 0, "est_cost_saved_usd": 0.0},
            "routing": {"est_tokens_avoided": 0, "answers": 0, "est_cost_saved_usd": 0.0},
        }
        self._routes: Dict[str, int] = {}  # model_routed_to -> count

    # ------------------------------------------------------------------ #
    @staticmethod
    def _usd(tokens: float, price_per_1m_usd: float) -> float:
        return round(tokens * price_per_1m_usd / 1_000_000, 6)

    def _routing_saved_usd(self, rec: RequestRecord) -> float:
        """$ avoided by answering on a cheaper tier than the main model.

        Local answers are free, so they save the full main-model price; a
        cheap-remote answer still bills at its own rate, so it saves only the
        price *delta* (clamped at zero if misconfigured upside-down).
        """

        if rec.routing_tier == "cheap_remote":
            delta = max(0.0, self.main_price - self.cheap_price)
            return self._usd(rec.est_tokens_avoided_routing, delta)
        return self._usd(rec.est_tokens_avoided_routing, self.main_price)

    def record(self, rec: RequestRecord) -> RequestRecord:
        """Fold one request into the running totals. Returns ``rec`` with
        ``est_cost_saved_usd`` filled in."""

        preflight_usd = self._usd(rec.tokens_saved_preflight, self.main_price)
        cache_usd = self._usd(rec.tokens_avoided_cache, self.main_price)
        routing_usd = self._routing_saved_usd(rec) if rec.est_tokens_avoided_routing else 0.0
        rec.est_cost_saved_usd = round(preflight_usd + cache_usd + routing_usd, 6)
        spent_usd = self._usd(
            rec.remote_tokens_spent,
            price_per_1m(rec.model_routed_to, self.default_price_per_1m),
        )

        with self._lock:
            self._requests += 1
            self._totals["prompt_tokens_before"] += rec.prompt_tokens_before
            self._totals["prompt_tokens_after"] += rec.prompt_tokens_after
            self._totals["tokens_saved_preflight"] += rec.tokens_saved_preflight
            self._totals["tokens_avoided_cache"] += rec.tokens_avoided_cache
            self._totals["est_tokens_avoided_routing"] += rec.est_tokens_avoided_routing
            self._totals["remote_tokens_spent"] += rec.remote_tokens_spent
            self._totals["est_cost_saved_usd"] = round(
                self._totals["est_cost_saved_usd"] + rec.est_cost_saved_usd, 6
            )
            self._totals["est_cost_spent_usd"] = round(
                self._totals["est_cost_spent_usd"] + spent_usd, 6
            )

            if rec.tokens_saved_preflight > 0:
                layer = self._layers["compression"]
                layer["tokens_saved"] += rec.tokens_saved_preflight
                layer["requests_touched"] += 1
                layer["est_cost_saved_usd"] = round(layer["est_cost_saved_usd"] + preflight_usd, 6)
            if rec.cache_hit_type in ("exact", "semantic"):
                layer = self._layers[f"{rec.cache_hit_type}_cache"]
                layer["tokens_avoided"] += rec.tokens_avoided_cache
                layer["hits"] += 1
                layer["est_cost_saved_usd"] = round(layer["est_cost_saved_usd"] + cache_usd, 6)
            if rec.est_tokens_avoided_routing > 0:
                layer = self._layers["routing"]
                layer["est_tokens_avoided"] += rec.est_tokens_avoided_routing
                layer["answers"] += 1  # answered off the main model (local or cheap tier)
                layer["est_cost_saved_usd"] = round(layer["est_cost_saved_usd"] + routing_usd, 6)

            self._routes[rec.model_routed_to] = self._routes.get(rec.model_routed_to, 0) + 1
            self._recent.append(asdict(rec))
        return rec

    # ------------------------------------------------------------------ #
    def snapshot(self, cache_stats=None) -> Dict[str, object]:
        """Cumulative totals + per-layer breakdown, JSON-ready.

        ``cache_stats`` is the live :class:`app.cache.CacheStats` from the
        agent's cache backend — passed in so hit rates come from the single
        source of truth instead of being re-counted here.
        """

        with self._lock:
            totals = dict(self._totals)
            layers = {name: dict(vals) for name, vals in self._layers.items()}
            routes = dict(self._routes)
            recent = list(self._recent)
            requests = self._requests

        totals["tokens_saved_total"] = int(
            totals["tokens_saved_preflight"]
            + totals["tokens_avoided_cache"]
            + totals["est_tokens_avoided_routing"]
        )

        cache_block: Optional[Dict[str, object]] = None
        if cache_stats is not None:
            semantic_hits = int(getattr(cache_stats, "semantic_hits", 0))
            cache_block = {
                "hits": cache_stats.hits,
                "misses": cache_stats.misses,
                "hit_rate": cache_stats.hit_rate,
                "exact_hits": cache_stats.hits - semantic_hits,
                "semantic_hits": semantic_hits,
            }

        return {
            "requests": requests,
            "token_counter": counter_name(),
            "pricing": {
                "main_model": self.main_model,
                "main_price_per_1m_usd": self.main_price,
                "cheap_model": self.cheap_model,
                "cheap_price_per_1m_usd": self.cheap_price,
            },
            "totals": totals,
            "layers": layers,
            "cache": cache_block,
            "routes": routes,
            "recent": recent,
        }
