"""Evaluation harness.

Runs the sample tasks through the agent and prints per-task routing plus a
summary with the headline metric: TOTAL REMOTE TOKENS (the only tokens that
count toward the hackathon score).

Two modes
---------
* ``--classify-only`` (default when no API key / GPU is available): exercises the
  routing *decisions* only — no model calls, no tokens spent. Great for a quick
  sanity check anywhere.
* full run: loads the local model and calls Fireworks, reporting real token
  counts. Requires the ML stack and ``FIREWORKS_API_KEY``.

Usage
-----
    python -m app.eval.run_eval                # full run (needs models + key)
    python -m app.eval.run_eval --classify-only
    python -m app.eval.run_eval --tasks path/to/tasks.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from app.config import load_config
from app.optimizer import PreflightOptimizer
from app.router.agent import Agent

SAMPLE_TASKS = Path(__file__).with_name("sample_tasks.json")

# A response counts as "degraded" if the optimized prompt's answer diverges
# from the raw prompt's answer below this cosine (or, without embeddings, this
# token-Jaccard). 1.0 == identical.
_DEGRADE_THRESHOLD = 0.98


def load_tasks(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_agent(classify_only: bool) -> Agent:
    cfg = load_config()
    if classify_only:
        # No models injected -> agent evaluates routing decisions only. The
        # LLM judge needs the local model, so this mode is always heuristic.
        return Agent(config=cfg)

    from app.models.local_model import LocalModel
    from app.models.remote_model import RemoteModel
    from app.router.classifier import LLMClassifier

    local = LocalModel(cfg.local_model, device=cfg.device, max_new_tokens=cfg.max_new_tokens)
    remote = RemoteModel(
        api_key=cfg.fireworks_api_key,
        model=cfg.remote_model,
        base_url=cfg.fireworks_base_url,
        max_new_tokens=cfg.max_new_tokens,
        timeout=cfg.remote_timeout,
        retries=cfg.remote_retries,
    )
    # CLASSIFIER=llm|heuristic: benchmark both against the same scored metric.
    # The judge shares the answering LocalModel — weights load once.
    classifier = LLMClassifier(local_model=local) if cfg.classifier == "llm" else None
    return Agent(config=cfg, classifier=classifier, local_model=local, remote_model=remote)


def run(tasks_path: Path, classify_only: bool) -> None:
    tasks = load_tasks(tasks_path)
    if not tasks:
        print(f"[eval] no tasks found in {tasks_path} — nothing to run.")
        return
    agent = build_agent(classify_only)

    mode = "CLASSIFY-ONLY (no tokens spent)" if classify_only else "FULL (models + Fireworks)"
    print("=" * 74)
    print(f"TokenSaver eval  |  mode: {mode}")
    print(f"local model : {agent.config.local_model}")
    print(f"remote model: {agent.config.remote_model}")
    print("=" * 74)

    total_remote = 0
    total_local = 0
    n_local = 0
    n_remote = 0
    n_correct_route = 0

    for task in tasks:
        query = task["query"]
        expected = task.get("expected_route")
        result = agent.route(query)

        total_remote += result.remote_tokens
        total_local += result.local_tokens

        # The classify step is no longer necessarily trace[0] (the cache runs
        # first, and a cache hit skips classification entirely).
        clf_step = next((s for s in result.trace if s.get("step") == "classify"), {})

        # "Primary" route for the local/remote split: any remote tokens => remote.
        primary = "remote" if result.remote_tokens > 0 else "local"
        if classify_only:
            # Without models, use the classifier's decision from the trace.
            primary = clf_step.get("route", "local")
        if primary == "remote":
            n_remote += 1
        else:
            n_local += 1

        route_ok = ""
        if expected is not None:
            hit = primary == expected
            n_correct_route += int(hit)
            route_ok = "OK " if hit else "MISS"

        difficulty = clf_step.get("difficulty", "-")
        print(
            f"[{task.get('id', '?'):>2}] {route_ok:<4} route={result.route:<34} "
            f"diff={difficulty:<5} remote_tok={result.remote_tokens:<5} "
            f"local_tok={result.local_tokens:<5}"
        )
        print(f"      q: {query[:78]}")

    n = len(tasks)
    print("-" * 74)
    print("SUMMARY")
    print(f"  tasks              : {n}")
    print(f"  routed local (free): {n_local}")
    print(f"  routed remote      : {n_remote}")
    if any(t.get("expected_route") for t in tasks):
        print(f"  routing match      : {n_correct_route}/{n}")
    print(f"  local tokens (0-cost): {total_local}")
    print("=" * 74)
    print(f"  TOTAL REMOTE TOKENS : {total_remote}   <-- scored metric")
    print("=" * 74)


# --------------------------------------------------------------------------- #
# Guardrail: does optimization change the answer?
# --------------------------------------------------------------------------- #
_WORD = re.compile(r"\w+")


def _load_embedder() -> Optional[Callable[[List[str]], object]]:
    """Batch-embed function (normalized vectors) or None if unavailable.

    Uses the **same** shared :class:`~app.embeddings.Embedder` as the semantic
    cache, so cache hits and validation cosine are scored with identical
    weights and the same backend selection.
    """

    try:
        from app.embeddings import get_embedder  # noqa: WPS433

        embedder = get_embedder()
        if embedder.warm():
            print(f"[validate] embedding backend: {embedder.active_backend}")
            return embedder.embed_batch
    except Exception:  # noqa: BLE001 — fall back to lexical similarity
        pass
    return None


def _jaccard(a: str, b: str) -> float:
    """Token-set Jaccard similarity — the no-embeddings fallback."""

    sa, sb = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0


def _build_answer_fn(classify_only: bool) -> Tuple[str, Optional[Callable[[str], str]]]:
    """A single-model answer function for A/B validation (bypasses routing).

    Prefers the remote model (needs FIREWORKS_API_KEY); otherwise the local
    model. Returns ``(name, fn)`` or ``(reason, None)`` when neither is usable.
    """

    cfg = load_config()
    if not classify_only and cfg.fireworks_api_key:
        from app.models.remote_model import RemoteModel

        rm = RemoteModel(
            api_key=cfg.fireworks_api_key,
            model=cfg.remote_model,
            base_url=cfg.fireworks_base_url,
            max_new_tokens=cfg.max_new_tokens,
            timeout=cfg.remote_timeout,
            retries=cfg.remote_retries,
        )
        return cfg.remote_model, lambda p: rm.generate(p).text

    if not classify_only:
        try:
            from app.models.local_model import LocalModel

            lm = LocalModel(cfg.local_model, device=cfg.device, max_new_tokens=cfg.max_new_tokens)
            return cfg.local_model, lambda p: lm.generate(p).text
        except Exception:  # noqa: BLE001
            pass
    return "no model available (set FIREWORKS_API_KEY, or install the local stack)", None


def validate(tasks_path: Path, classify_only: bool) -> None:
    """Run each prompt raw AND fully optimized, then diff the two answers.

    This is the proof that savings didn't cost quality: for every prompt we
    report exact-match and answer similarity (embedding cosine, or lexical
    Jaccard when embeddings are absent), and flag any prompt whose answer
    changed. Attribution: the pre-flight ``removed`` labels are printed for
    each offender, so a regression points at the stage (dedup / filler /
    whitespace) that caused it — every stage is independently toggleable.
    """

    tasks = load_tasks(tasks_path)
    if not tasks:
        print(f"[validate] no tasks found in {tasks_path} — nothing to validate.")
        return
    optimizer = PreflightOptimizer()
    model_name, answer = _build_answer_fn(classify_only)

    print("=" * 74)
    print("TokenSaver guardrail  |  raw vs optimized answer diff")
    print(f"answering model : {model_name}")
    print("=" * 74)

    if answer is None:
        print("[skip] Cannot compare answers without a model.")
        print("       Optimization-only preview (tokens saved, no answer diff):\n")
        for task in tasks:
            opt = optimizer.optimize(task["query"])
            print(f"[{task.get('id', '?'):>2}] -{opt.tokens_saved:>3} tok  "
                  f"removed={opt.removed or '[]'}")
        return

    embed = _load_embedder()
    sim_kind = "cosine" if embed else "jaccard"
    identical = 0
    degraded: List[dict] = []
    total_saved = 0

    for task in tasks:
        query = task["query"]
        opt = optimizer.optimize(query)
        total_saved += opt.tokens_saved

        raw_ans = (answer(query) or "").strip()
        opt_ans = (answer(opt.text) or "").strip()

        exact = raw_ans == opt_ans
        if embed is not None:
            vecs = embed([raw_ans, opt_ans])
            sim = float((vecs[0] * vecs[1]).sum())  # normalized -> dot == cosine
        else:
            sim = _jaccard(raw_ans, opt_ans)

        if exact:
            identical += 1
        flag = "IDENTICAL" if exact else ("OK " if sim >= _DEGRADE_THRESHOLD else "DEGRADED")
        if not exact and sim < _DEGRADE_THRESHOLD:
            degraded.append({"id": task.get("id", "?"), "sim": sim,
                             "removed": opt.removed, "query": query})

        print(f"[{task.get('id', '?'):>2}] {flag:<9} {sim_kind}={sim:.3f}  "
              f"saved={opt.tokens_saved:>3} tok  removed={opt.removed or '[]'}")

    n = len(tasks)
    print("-" * 74)
    print("GUARDRAIL SUMMARY")
    print(f"  prompts             : {n}")
    print(f"  identical answers   : {identical}/{n}  ({100 * identical / n:.0f}%)")
    print(f"  degraded (< {_DEGRADE_THRESHOLD:.2f} {sim_kind}): {len(degraded)}/{n}  "
          f"({100 * len(degraded) / n:.0f}%)")
    print(f"  total tokens saved  : {total_saved}")
    if degraded:
        print("  worst offenders     :")
        for d in sorted(degraded, key=lambda x: x["sim"])[:5]:
            print(f"    [{d['id']}] {sim_kind}={d['sim']:.3f}  attribute->{d['removed']}")
            print(f"        q: {d['query'][:70]}")
    else:
        print("  worst offenders     : none — optimization preserved every answer")
    print("=" * 74)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="TokenSaver eval harness")
    parser.add_argument(
        "--tasks",
        type=Path,
        default=SAMPLE_TASKS,
        help="Path to a tasks JSON file (default: bundled sample_tasks.json).",
    )
    parser.add_argument(
        "--classify-only",
        action="store_true",
        help="Only evaluate routing decisions; do not call any model.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Guardrail: run each prompt raw AND optimized, diff the answers.",
    )
    args = parser.parse_args(argv)

    if args.validate:
        validate(args.tasks, args.classify_only)
        return

    classify_only = args.classify_only
    # Auto-fallback: without an API key a full run can't reach the remote model,
    # so default to classify-only and tell the user.
    if not classify_only and not os.getenv("FIREWORKS_API_KEY"):
        print("[info] FIREWORKS_API_KEY not set -> falling back to --classify-only.\n")
        classify_only = True

    run(args.tasks, classify_only)


if __name__ == "__main__":
    main()
