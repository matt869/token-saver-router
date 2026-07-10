"""Live "watch it fail and recover" demo for TokenSaver.

This adds NO routing logic. It wraps the *configured* local model in a shim that
raises from ``generate()`` when ``FORCE_LOCAL_FAILURE`` is set, then lets the
EXISTING remote-failover path in ``app/router/agent.py`` handle recovery. The
router, classifier, and model wrappers are all untouched.

Trigger (toggle in one action while presenting):
    python -m app.demo.failover_demo                       # NORMAL  -> routes local, succeeds
    FORCE_LOCAL_FAILURE=1 python -m app.demo.failover_demo # FAILURE -> local throws -> remote

Remote answer source: real Fireworks when FIREWORKS_API_KEY is set, otherwise the
keyless proportional stub (so the demo works with zero credentials).
"""
from __future__ import annotations

import os
import sys

from app.config import load_config
from app.models.local_model import LocalModel
from app.router.agent import Agent
from app.router.classifier import HeuristicClassifier

_TRUTHY = ("1", "true", "yes", "on")


def _failure_armed() -> bool:
    return os.getenv("FORCE_LOCAL_FAILURE", "").strip().lower() in _TRUTHY


class FailOnDemandLocal:
    """Wraps a real local model; raises from ``generate()`` when the demo trigger
    is armed, otherwise delegates untouched. ``__getattr__`` passes ``load()``,
    ``runtime_info()``, ``model_name``, etc. straight through to the real model."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def generate(self, prompt, max_new_tokens=None):
        if _failure_armed():
            raise RuntimeError(
                "FORCE_LOCAL_FAILURE: simulated local GPU/model failure (demo trigger)"
            )
        return self._inner.generate(prompt, max_new_tokens)


def _build_remote(cfg):
    """Same selection main.py uses: real Fireworks when a key is set, else the
    keyless proportional stub so the demo runs with no credentials."""
    if not cfg.remote_stub and cfg.fireworks_api_key:
        from app.models.remote_model import RemoteModel

        return RemoteModel(
            api_key=cfg.fireworks_api_key, model=cfg.remote_model,
            base_url=cfg.fireworks_base_url, max_new_tokens=cfg.max_new_tokens,
            timeout=cfg.remote_timeout, retries=cfg.remote_retries,
        )
    from app.models.stub_remote import StubRemoteModel

    return StubRemoteModel()


def build_demo_agent() -> Agent:
    cfg = load_config()
    local = FailOnDemandLocal(
        LocalModel(cfg.local_model, device=cfg.device, max_new_tokens=cfg.max_new_tokens)
    )
    # Heuristic classifier: deterministic routing, and the ONLY local call is the
    # answering one — so the trigger yields exactly one visible failure on screen.
    agent = Agent(
        config=cfg,
        classifier=HeuristicClassifier(cfg.complexity_length_cutoff),
        local_model=local,
        remote_model=_build_remote(cfg),
    )
    agent.cache = None  # no cache short-circuit, so the failover path always runs
    return agent


def _print_result(prompt, result) -> None:
    print(f"PROMPT : {prompt}")
    print(f"ROUTE  : {result.route}")
    print(f"ANSWER : {result.answer!r}")
    print(f"TOKENS : remote={result.remote_tokens}  local={result.local_tokens}")
    print("TRACE  :")
    for s in result.trace:
        step = s.get("step")
        if step == "local" and str(s.get("status", "")).startswith("failed"):
            print(f"   [local ] FAILED  -> {s.get('status')}")
            print(f"            {s.get('note')}")
        elif step == "local" and "tokens" in s:
            print(f"   [local ] OK      -> answered locally ({s.get('tokens')} tokens, free)")
        elif step == "local":
            print(f"   [local ] {s.get('status')}")
        elif step == "remote" and "tokens" in s:
            print(f"   [remote] SUCCESS -> {s.get('model')} ({s.get('tokens')} tokens) "
                  f"reason={s.get('reason')!r}")
        elif step == "verify":
            print(f"   [verify] confidence={s.get('confidence')} escalate={s.get('escalate')}")


class SessionTotals:
    """Accumulates the demo's OWN per-request numbers into one recap line.

    Reuses ``result.remote_tokens`` / ``result.route`` exactly as the agent
    computed them — nothing is recomputed. The only extra quantity is the
    all-remote baseline, derived the same way as ``baseline_delta.py``: a keyless
    ``StubRemoteModel``'s token count of the RAW prompt (the counterfactual cost
    if that prompt had been sent straight to remote).
    """

    def __init__(self):
        self.requests = 0
        self.free_local = 0        # answered with zero remote tokens
        self.escalated = 0         # local died -> served by remote
        self.remote_used = 0       # actual remote tokens spent
        self.baseline_all_remote = 0  # what all requests would cost sent raw to remote

    def add(self, result, baseline_tokens: int) -> None:
        self.requests += 1
        self.remote_used += result.remote_tokens
        self.baseline_all_remote += baseline_tokens
        if result.remote_tokens == 0:
            self.free_local += 1
        if "escalated" in result.route:
            self.escalated += 1

    def line(self) -> str:
        saved = self.baseline_all_remote - self.remote_used
        pct = (100.0 * saved / self.baseline_all_remote) if self.baseline_all_remote else 0.0
        return (
            f"SESSION TOTALS: {self.requests} requests "
            f"| {self.free_local} served free locally "
            f"| {self.escalated} escalated "
            f"| remote tokens used: {self.remote_used} "
            f"| est. saved vs all-remote: {saved} tokens ({pct:.0f}%)"
        )


def _run_once(agent, prompt, armed, baseline_stub, totals) -> None:
    """Route one prompt with the failure trigger armed or disarmed, print the
    per-request view, and fold its numbers into the running session totals."""

    os.environ["FORCE_LOCAL_FAILURE"] = "1" if armed else ""
    banner = "FAILURE ARMED  (FORCE_LOCAL_FAILURE=1)" if armed else "NORMAL  (local healthy)"
    print("=" * 72)
    print(f"TokenSaver failover demo  |  {banner}")
    print("=" * 72)
    result = agent.route(prompt)
    _print_result(prompt, result)
    # All-remote baseline for THIS prompt (stub-derived; == baseline_delta.py).
    baseline_tokens = baseline_stub.generate(prompt).tokens
    totals.add(result, baseline_tokens)
    print("=" * 72)


def main() -> None:
    from app.models.stub_remote import StubRemoteModel

    # --session (or "both"): run NORMAL then FAILURE in ONE process and print the
    # combined recap. Preferred for a live demo — no fragile cross-process state.
    session_mode = "--session" in sys.argv or "both" in sys.argv

    prompt = os.getenv("DEMO_PROMPT", "What is the capital of France?")
    agent = build_demo_agent()
    baseline_stub = StubRemoteModel()  # keyless counterfactual "all-remote" meter
    totals = SessionTotals()

    if session_mode:
        _run_once(agent, prompt, False, baseline_stub, totals)  # NORMAL  -> local
        _run_once(agent, prompt, True, baseline_stub, totals)   # FAILURE -> escalate
    else:
        _run_once(agent, prompt, _failure_armed(), baseline_stub, totals)

    print(totals.line())


if __name__ == "__main__":
    main()
