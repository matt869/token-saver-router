"""Environment-driven configuration for TokenSaver.

Every knob is read from an environment variable with a sensible default so the
service runs out of the box and can be tuned without code changes. Load once at
startup via :func:`load_config`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _get_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Config:
    """Immutable runtime configuration."""

    # --- Remote (scored) model: Fireworks AI ---
    fireworks_api_key: str
    fireworks_base_url: str
    remote_model: str
    cheap_remote_model: str  # optional smaller Fireworks model for "easy" tasks ("" = off)
    remote_price_per_1m_usd: float  # fallback $/1M tokens for models not in pricing.py

    # --- Local (free / zero-scored) model: HuggingFace transformers ---
    local_model: str
    device: str

    # --- Routing behaviour ---
    classifier: str  # "llm" (judge via the local model) or "heuristic" (regex/length)
    confidence_threshold: float  # verifier self-check cutoff (0..1)
    complexity_length_cutoff: int  # word count that biases "hard"

    # --- Generation limits ---
    max_new_tokens: int
    remote_timeout: int
    remote_retries: int  # extra attempts on transient (429/5xx/network) errors
    warmup_enabled: bool  # preload local model + cache encoder at server boot
    remote_stub: bool  # keyless demo: swap the scored remote for a proportional stub

    # --- Token-saving layers (all in front of the scored remote path) ---
    preflight_enabled: bool  # strip whitespace/politeness/filler before remote
    cache_enabled: bool  # serve (near-)duplicate queries for zero remote tokens
    cache_backend: str  # "semantic" (MiniLM + faiss) or "exact" (hash only)
    cache_max_entries: int  # LRU cap on the query cache
    cache_similarity_threshold: float  # cosine cutoff for a semantic cache hit
    cache_persist_path: str  # sqlite file for cross-restart persistence ("" = in-memory only)
    embed_backend: str  # "local" (MiniLM) or "remote" (Fireworks embeddings API)
    embed_model: str  # local sentence-transformers model
    embed_remote_model: str  # Fireworks embeddings model for the remote backend
    ids_enabled: bool  # regex inspection for heavy-compute signatures
    ids_threshold: float  # severity (0..1) at which IDS forces the remote route
    failover_enabled: bool  # on remote failure, serve from the local model


def load_config() -> Config:
    """Build a :class:`Config` from the process environment."""

    return Config(
        fireworks_api_key=_get_str("FIREWORKS_API_KEY", ""),
        fireworks_base_url=_get_str(
            "FIREWORKS_BASE_URL",
            "https://api.fireworks.ai/inference/v1",
        ),
        remote_model=_get_str(
            "REMOTE_MODEL",
            "accounts/fireworks/models/gemma-2-9b-it",
        ),
        # Cheap-remote tier: a smaller Fireworks model for "easy" tasks so the
        # router still saves money on machines with no GPU. Empty = disabled;
        # easy tasks then use the local model (or the main remote if none).
        cheap_remote_model=_get_str("CHEAP_REMOTE_MODEL", ""),
        remote_price_per_1m_usd=_get_float("REMOTE_PRICE_PER_1M_USD", 0.20),
        # Qwen 7B in bf16 needs ~15-16 GB VRAM — the scoring environment must
        # have it. Override with LOCAL_MODEL=Qwen/Qwen2.5-3B-Instruct (or the
        # 1.5B/2B class) if VRAM is tight; everything stays env-overridable.
        local_model=_get_str("LOCAL_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        # ROCm exposes AMD GPUs as "cuda" in PyTorch, hence the default.
        device=_get_str("LOCAL_DEVICE", "cuda"),
        classifier=_get_str("CLASSIFIER", "llm"),
        confidence_threshold=_get_float("CONFIDENCE_THRESHOLD", 0.6),
        complexity_length_cutoff=_get_int("COMPLEXITY_LENGTH_CUTOFF", 24),
        max_new_tokens=_get_int("MAX_NEW_TOKENS", 256),
        remote_timeout=_get_int("REMOTE_TIMEOUT", 60),
        remote_retries=_get_int("REMOTE_RETRIES", 1),
        warmup_enabled=_get_bool("WARMUP_ENABLED", True),
        remote_stub=_get_bool("REMOTE_STUB", False),
        preflight_enabled=_get_bool("PREFLIGHT_ENABLED", True),
        cache_enabled=_get_bool("CACHE_ENABLED", True),
        cache_backend=_get_str("CACHE_BACKEND", "semantic"),
        cache_max_entries=_get_int("CACHE_MAX_ENTRIES", 512),
        # 0.95 default: strict enough that "capital of Austria" never serves the
        # Australia answer. Loosen via env only with the --validate guardrail.
        cache_similarity_threshold=_get_float("CACHE_SIMILARITY_THRESHOLD", 0.95),
        cache_persist_path=_get_str("CACHE_PERSIST_PATH", ""),
        embed_backend=_get_str("EMBED_BACKEND", "local"),
        embed_model=_get_str("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        embed_remote_model=_get_str("EMBED_REMOTE_MODEL", "nomic-ai/nomic-embed-text-v1.5"),
        ids_enabled=_get_bool("IDS_ENABLED", True),
        ids_threshold=_get_float("IDS_THRESHOLD", 0.5),
        failover_enabled=_get_bool("FAILOVER_ENABLED", True),
    )
