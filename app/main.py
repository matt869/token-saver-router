"""FastAPI server for TokenSaver.

Endpoints
---------
* ``GET  /health`` — status + configured model names.
* ``POST /route``  — route a single query and return the answer, the route
  taken, token accounting (remote vs local), and the decision trace.

The heavy local model is loaded lazily on first use, so the server starts fast
and ``/health`` works even before any model is materialised.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.config import Config, load_config
from app.models.local_model import LocalModel
from app.models.remote_model import RemoteModel
from app.router.agent import Agent
from app.router.classifier import LLMClassifier

_DASHBOARD_HTML = Path(__file__).with_name("dashboard.html")


def _warmup() -> None:
    """Preload the local model + cache encoder so request #1 isn't a cold start.

    Loading a 7B model takes minutes; doing it lazily inside the first request
    risks tripping the caller's HTTP timeout. Runs in a daemon thread so
    ``/health`` responds immediately while the weights stream in. Must never
    crash the server — a machine without the ML stack just logs and moves on.
    """

    try:
        agent = get_agent()
        cache = agent.cache
        if cache is not None and hasattr(cache, "warm"):
            # Loads the embedding model once and logs the active backend
            # (local MiniLM vs Fireworks) so the demo box's mode is obvious.
            cache.warm()
        if agent.local_model is not None:
            agent.local_model.load()  # the big one
        print("[warmup] local model + cache encoder ready")
    except Exception as exc:  # noqa: BLE001 — warmup is best-effort
        print(f"[warmup] skipped: {exc}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Emit per-request routing decisions (tokensaver.router logger) to the
    # console. basicConfig is a no-op if uvicorn already installed handlers,
    # in which case our INFO records still propagate to them.
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
    logging.getLogger("tokensaver").setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    if get_config().warmup_enabled:
        threading.Thread(target=_warmup, name="tokensaver-warmup", daemon=True).start()
    yield


app = FastAPI(title="TokenSaver", version="0.1.0", lifespan=lifespan)


@lru_cache(maxsize=1)
def get_config() -> Config:
    return load_config()


@lru_cache(maxsize=1)
def get_agent() -> Agent:
    cfg = get_config()
    local = LocalModel(
        model_name=cfg.local_model,
        device=cfg.device,
        max_new_tokens=cfg.max_new_tokens,
    )
    if cfg.remote_stub:
        # Keyless demo: proportional stub so the dashboard shows real savings.
        from app.models.stub_remote import StubRemoteModel

        remote = StubRemoteModel(model="stub/echo")
        print("[config] REMOTE_STUB=true -> using keyless proportional stub remote")
    else:
        remote = RemoteModel(
            api_key=cfg.fireworks_api_key,
            model=cfg.remote_model,
            base_url=cfg.fireworks_base_url,
            max_new_tokens=cfg.max_new_tokens,
            timeout=cfg.remote_timeout,
            retries=cfg.remote_retries,
        )
    # Optional cheap-remote tier: a smaller Fireworks model for easy tasks.
    cheap_remote = None
    if cfg.cheap_remote_model:
        cheap_remote = RemoteModel(
            api_key=cfg.fireworks_api_key,
            model=cfg.cheap_remote_model,
            base_url=cfg.fireworks_base_url,
            max_new_tokens=cfg.max_new_tokens,
            timeout=cfg.remote_timeout,
            retries=cfg.remote_retries,
        )
    # CLASSIFIER=llm: the judge shares the *same* LocalModel instance as the
    # answering path, so the weights are loaded exactly once per process.
    classifier = LLMClassifier(local_model=local) if cfg.classifier == "llm" else None
    return Agent(
        config=cfg,
        classifier=classifier,
        local_model=local,
        remote_model=remote,
        cheap_remote_model=cheap_remote,
    )


class RouteRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user query to route.")


class RouteResponse(BaseModel):
    # model_used isn't a pydantic "model_" config field — opt out of the guard.
    model_config = {"protected_namespaces": ()}

    answer: str
    route: str
    remote_tokens: int
    local_tokens: int
    trace: list
    cached: bool = False
    cache_hit_type: str = "none"
    preflight_tokens_saved: int = 0
    remote_tokens_avoided: int = 0
    prompt_tokens_before: int = 0
    prompt_tokens_after: int = 0
    model_used: str = ""


@app.get("/health")
def health() -> dict:
    cfg = get_config()
    # Resolved device/dtype: shows what the local model is REALLY on (e.g.
    # cpu/fp32 after a silent GPU fallback) vs. what LOCAL_DEVICE requested.
    # Building the agent is cheap — it does NOT load the 7B weights.
    local = get_agent().local_model
    local_runtime = local.runtime_info() if local is not None else None
    return {
        "status": "ok",
        "local_model": cfg.local_model,
        "remote_model": cfg.remote_model,
        "device": cfg.device,
        "local_runtime": local_runtime,
        "confidence_threshold": cfg.confidence_threshold,
        "complexity_length_cutoff": cfg.complexity_length_cutoff,
        "remote_api_key_configured": bool(cfg.fireworks_api_key),
        "classifier": cfg.classifier,
        "preflight_enabled": cfg.preflight_enabled,
        "cache_enabled": cfg.cache_enabled,
        "cache_backend": cfg.cache_backend,
        "cache_similarity_threshold": cfg.cache_similarity_threshold,
        "ids_enabled": cfg.ids_enabled,
        "failover_enabled": cfg.failover_enabled,
        "cheap_remote_model": cfg.cheap_remote_model or None,
        "cache_persist_path": cfg.cache_persist_path or None,
    }


@app.post("/route", response_model=RouteResponse)
def route(req: RouteRequest) -> RouteResponse:
    result = get_agent().route(req.query)
    return RouteResponse(**result.as_dict())


@app.get("/metrics")
def metrics() -> dict:
    """Cumulative savings totals + per-layer breakdown.

    Cache hit rates are read straight off the live ``CacheStats`` so the
    numbers can never drift from what the cache itself counted.
    """

    agent = get_agent()
    cache_stats = agent.cache.stats if agent.cache is not None else None
    return agent.metrics.snapshot(cache_stats=cache_stats)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    """Minimal single-file HTML view that polls /metrics."""

    try:
        return _DASHBOARD_HTML.read_text(encoding="utf-8")
    except OSError:
        return "<h1>TokenSaver</h1><p>dashboard.html not found.</p>"


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
