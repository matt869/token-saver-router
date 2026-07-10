"""Latency benchmark — RUN ON THE ROCm BOX.

Adds a speed dimension to the cost story: local easy-path generation time vs.
remote round-trip time. It reuses the existing model wrappers unchanged; the only
added logic is a ``time.perf_counter()`` wrap around each ``generate()`` call.

4 easy tasks -> local model, 4 hard tasks -> remote. Remote is REAL Fireworks when
``FIREWORKS_API_KEY`` is set and ``REMOTE_STUB`` isn't; otherwise the keyless stub,
which is labeled UNREPRESENTATIVE (no real network hop) so a fast stub number is
never mistaken for a real one.

Run on the ROCm box at your scored config:
    ENV=production python -m app.demo.latency_bench
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from app.config import load_config
from app.models.local_model import LocalModel

SAMPLE_TASKS = Path("app/eval/sample_tasks.json")


def _fmt(ms: float) -> str:
    return f"{ms:8.1f} ms"


def _build_remote(cfg):
    """Real Fireworks when a key is set (and stub not forced), else keyless stub."""
    if not cfg.remote_stub and cfg.fireworks_api_key:
        from app.models.remote_model import RemoteModel

        rm = RemoteModel(
            api_key=cfg.fireworks_api_key, model=cfg.remote_model,
            base_url=cfg.fireworks_base_url, max_new_tokens=cfg.max_new_tokens,
            timeout=cfg.remote_timeout, retries=cfg.remote_retries,
        )
        return rm, f"REAL Fireworks ({cfg.remote_model})", True
    from app.models.stub_remote import StubRemoteModel

    return StubRemoteModel(), "STUB (keyless) - UNREPRESENTATIVE, no real network hop", False


def main() -> None:
    cfg = load_config()
    tasks = json.loads(SAMPLE_TASKS.read_text(encoding="utf-8"))
    easy = [t for t in tasks if t["expected_route"] == "local"]
    hard = [t for t in tasks if t["expected_route"] == "remote"]

    local = LocalModel(cfg.local_model, device=cfg.device, max_new_tokens=cfg.max_new_tokens)
    remote, remote_label, remote_real = _build_remote(cfg)

    print("=" * 74)
    print("TokenSaver latency benchmark  |  easy=LOCAL gen   hard=REMOTE round-trip")
    print(f"local model : {cfg.local_model}  device={cfg.device}  max_new_tokens={cfg.max_new_tokens}")
    print(f"remote      : {remote_label}")
    print("=" * 74)

    # Warm the local model FIRST so a one-time weight load / kernel compile is not
    # charged to task 1 — we want steady-state generation latency, not cold start.
    print("[warm] loading local model + one throwaway generate ...", flush=True)
    local.load()
    local.generate("warmup", max_new_tokens=8)
    print(f"[warm] resolved: {local.runtime_info()}", flush=True)

    print("\n-- LOCAL (easy path: local generation) --")
    local_ms = []
    for t in easy:
        s = time.perf_counter()
        r = local.generate(t["query"])
        ms = (time.perf_counter() - s) * 1000
        local_ms.append(ms)
        print(f"  id={t['id']:>2}  {_fmt(ms)}  tokens={r.tokens}")

    print("\n-- REMOTE (hard path: network round-trip) --")
    remote_ms = []
    for t in hard:
        s = time.perf_counter()
        r = remote.generate(t["query"])
        ms = (time.perf_counter() - s) * 1000
        remote_ms.append(ms)
        print(f"  id={t['id']:>2}  {_fmt(ms)}  tokens={r.tokens}")

    lmed = statistics.median(local_ms)
    rmed = statistics.median(remote_ms)
    speed = (rmed / lmed) if lmed else float("inf")

    print("\n" + "-" * 74)
    print(f"local  median: {_fmt(lmed)}")
    print(f"remote median: {_fmt(rmed)}")
    print("=" * 74)
    note = "" if remote_real else "   [remote=STUB -> latency NOT representative; rerun with a real key]"
    print(
        f"SUMMARY: local median {lmed:.0f}ms vs remote median {rmed:.0f}ms - "
        f"easy traffic answered {speed:.1f}x faster with zero network hop.{note}"
    )
    print("=" * 74)


if __name__ == "__main__":
    main()
