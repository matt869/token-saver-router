"""Model price table — turns token counts into estimated dollars.

Fireworks bills serverless inference per 1M tokens (prompt + completion at the
same rate), tiered by parameter count. The table below carries the list prices
for the models this project is likely to route between; anything unknown falls
back to the tier default or ``REMOTE_PRICE_PER_1M_USD``.

Prices are list prices as of mid-2026 and exist to make *relative* savings
visible ("this layer paid for itself") — they are estimates, always labelled
``est_`` downstream, never billing truth.
"""

from __future__ import annotations

from typing import Dict

# $ per 1M tokens (prompt and completion billed at the same rate on Fireworks
# serverless). Keyed by the full Fireworks model path.
DEFAULT_PRICE_TABLE: Dict[str, float] = {
    # <16B tier — $0.20 / 1M
    "accounts/fireworks/models/gemma-2-9b-it": 0.20,
    "accounts/fireworks/models/llama-v3p1-8b-instruct": 0.20,
    "accounts/fireworks/models/llama-v3p2-3b-instruct": 0.10,
    "accounts/fireworks/models/llama-v3p2-1b-instruct": 0.10,
    "accounts/fireworks/models/qwen2p5-7b-instruct": 0.20,
    # 16.1B–80B tier — $0.90 / 1M
    "accounts/fireworks/models/llama-v3p1-70b-instruct": 0.90,
    "accounts/fireworks/models/llama-v3p3-70b-instruct": 0.90,
    "accounts/fireworks/models/qwen2p5-72b-instruct": 0.90,
    # MoE / frontier
    "accounts/fireworks/models/mixtral-8x22b-instruct": 1.20,
    "accounts/fireworks/models/deepseek-v3": 0.90,
    "accounts/fireworks/models/llama-v3p1-405b-instruct": 3.00,
}


def price_per_1m(model: str, default: float = 0.20, table: Dict[str, float] | None = None) -> float:
    """Price for ``model`` in $/1M tokens, falling back to ``default``.

    A local HF model (or an empty name) prices at 0.0 — the local path is free.
    """

    # Local models are referenced by HF id ("Qwen/...") but never billed;
    # only Fireworks paths ("accounts/...") carry a price.
    if not model or not model.startswith("accounts/"):
        return 0.0
    return (table or DEFAULT_PRICE_TABLE).get(model, default)
