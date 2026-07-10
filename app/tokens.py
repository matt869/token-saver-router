"""Real token counting for honest before/after measurement.

The pre-flight optimizer (and the metrics layer) need to say how many tokens a
compression pass actually saved. A chars/4 guess is fine for a trace note but
useless as a headline metric, so this module provides :func:`count_tokens`
backed by a real BPE tokenizer:

* **tiktoken** (``cl100k_base``) when installed — milliseconds, no ML stack.
  Fireworks models use their own tokenizers, so counts are approximate for
  billing purposes, but *consistent*: before and after are measured with the
  same vocabulary, which is what makes "tokens saved" honest.
* **chars/4 estimate** as the last-resort fallback, so every caller keeps
  working on a machine with nothing installed (the project-wide degradation
  rule). :func:`counter_name` reports which backend is active so metrics can
  label the numbers.

The encoder is loaded lazily and cached at module level (same pattern as the
MiniLM encoder in ``app/cache.py``).
"""

from __future__ import annotations

import threading

_CHARS_PER_TOKEN = 4  # fallback estimate for English prose

# None = untried, otherwise the resolved (name, encode_fn) pair.
_BACKEND: "tuple[str, object] | None" = None
_BACKEND_LOCK = threading.Lock()


def _resolve_backend() -> "tuple[str, object]":
    """Pick the best available counter, once per process."""

    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    with _BACKEND_LOCK:
        if _BACKEND is not None:
            return _BACKEND
        try:
            import tiktoken  # noqa: WPS433 (lazy import is intentional)

            enc = tiktoken.get_encoding("cl100k_base")
            _BACKEND = ("tiktoken/cl100k_base", enc.encode)
        except Exception:  # noqa: BLE001 — missing dep or no network for the BPE file
            _BACKEND = ("estimate/chars-div-4", None)
    return _BACKEND


def counter_name() -> str:
    """Which backend produced the counts (for metrics labels)."""

    return _resolve_backend()[0]


def count_tokens(text: str) -> int:
    """Count tokens in ``text`` with the best available tokenizer."""

    if not text:
        return 0
    name, encode = _resolve_backend()
    if encode is not None:
        try:
            return len(encode(text))
        except Exception:  # noqa: BLE001 — never let counting break a request
            pass
    return max(1, len(text) // _CHARS_PER_TOKEN)
