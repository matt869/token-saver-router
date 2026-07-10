"""Routing + verification core.

This is the heart of TokenSaver. It ties together a *swappable* classifier and
a *swappable* verifier with the local and remote models. Both the classifier and
the verifier are injected, so a fine-tuned model can replace either one without
editing this file.

Routing flow
------------
1. Classify difficulty (cheap, local, zero-token).
2. Hard   -> go straight to the remote (scored) model.
3. Easy   -> answer locally (free), then run a cheap self-check.
             If confidence < threshold, escalate to the remote model.

Token accounting is tracked separately: ``remote_tokens`` (scored) vs
``local_tokens`` (scored as zero), plus a step-by-step ``trace``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, runtime_checkable

from app.cache import QueryCache, create_cache
from app.config import Config
from app.metrics import MetricsRegistry, RequestRecord
from app.optimizer import PreflightOptimizer
from app.router.classifier import Classification, Classifier, HeuristicClassifier
from app.router.executor import Executor
from app.router.ids import RegexIDS
from app.tokens import count_tokens

# One INFO line per routed request records the chosen tier and the concrete
# reason (classifier signal or IDS rule) so a real run is auditable. Handlers
# are the deployment's choice; main.py enables INFO at server startup.
logger = logging.getLogger("tokensaver.router")

# User-facing degraded messages returned when every tier fails. Non-empty on
# purpose so a caller never receives a silent 200 with answer="", and never a
# 500 — the request degrades cleanly instead.
_DEGRADED_BOTH = (
    "[degraded] Service temporarily unavailable: the remote model failed and the "
    "local fallback also failed to respond. Please retry shortly."
)
_DEGRADED_NO_LOCAL = (
    "[degraded] Service temporarily unavailable: the remote model failed and no "
    "local fallback is configured. Please retry shortly."
)


def _norm_answer(text: str) -> str:
    """Whitespace/case/trailing-punctuation-insensitive form for agreement."""
    return " ".join((text or "").split()).lower().rstrip(".!?")


def _agreement(answers: List[str]):
    """Return ``(trusted, majority_answer, ratio)`` for self-consistency samples.

    ``trusted`` requires a *strict majority* of the (normalized) samples to agree
    — 2 of 3, both of 2. The returned answer is the original (un-normalized) text
    of the majority so casing/formatting is preserved.
    """
    from collections import Counter

    norms = [_norm_answer(a) for a in answers]
    top_norm, top_count = Counter(norms).most_common(1)[0]
    ratio = top_count / len(answers) if answers else 0.0
    trusted = top_count > len(answers) / 2
    majority_original = answers[norms.index(top_norm)]
    return trusted, majority_original, ratio


# --------------------------------------------------------------------------- #
# Verifier: swappable self-check on a local answer.
# --------------------------------------------------------------------------- #
@dataclass
class Verification:
    """Confidence in a local answer and whether to escalate."""

    confidence: float  # 0.0 (no confidence) .. 1.0 (fully confident)
    escalate: bool
    reasons: List[str] = field(default_factory=list)


@runtime_checkable
class Verifier(Protocol):
    """Swappable verifier interface for the local-answer self-check."""

    def verify(self, query: str, answer: str, threshold: float) -> Verification:  # pragma: no cover
        ...


class HeuristicVerifier:
    """Cheap, dependency-free confidence check on a local answer.

    Confidence drops when the answer is empty/very short, when it contains
    hedging or refusal phrases, or when it looks truncated. Replace with a
    learned verifier without touching :class:`Agent`.
    """

    LOW_CONFIDENCE_PHRASES = (
        "i don't know", "i do not know", "i'm not sure", "i am not sure",
        "not sure", "cannot help", "can't help", "as an ai", "i cannot",
        "unable to", "no idea", "unclear", "it depends",
    )

    def verify(self, query: str, answer: str, threshold: float) -> Verification:
        reasons: List[str] = []
        text = (answer or "").strip()
        confidence = 1.0

        if not text:
            return Verification(confidence=0.0, escalate=True, reasons=["empty answer"])

        n_words = len(text.split())
        if n_words < 3:
            confidence -= 0.5
            reasons.append("answer very short")

        lowered = text.lower()
        for phrase in self.LOW_CONFIDENCE_PHRASES:
            if phrase in lowered:
                confidence -= 0.6
                reasons.append(f"hedging phrase: '{phrase}'")
                break

        # Looks truncated (ends mid-sentence without terminal punctuation).
        if not text.endswith((".", "!", "?", "```", ")", "\"")):
            confidence -= 0.15
            reasons.append("possibly truncated")

        confidence = max(0.0, min(1.0, confidence))
        escalate = confidence < threshold
        if escalate and not reasons:
            reasons.append("confidence below threshold")
        return Verification(confidence=round(confidence, 3), escalate=escalate, reasons=reasons)


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
@dataclass
class _RemoteOutcome:
    """Internal result of a remote attempt (possibly served via failover)."""

    text: str
    remote_tokens: int  # scored (total billed)
    local_tokens: int  # non-zero only when we failed over to the local model
    preflight_tokens_saved: int  # real-tokenizer prompt savings on this call
    prompt_tokens: int  # usage.prompt_tokens the provider actually billed
    failed_over: bool  # True if the remote errored and local served the answer
    ok: bool  # True if we produced an answer at all


@dataclass
class RouteResult:
    """Full result of routing one query.

    The old overloaded ``est_tokens_saved`` is split into two honest fields:
    ``preflight_tokens_saved`` (compression on a live remote call) and
    ``remote_tokens_avoided`` (spend a cache hit skipped). ``routing_tier`` and
    ``est_tokens_avoided_routing`` feed the metrics registry only and stay out
    of the API payload.
    """

    answer: str
    route: str  # "local", "local->remote (escalated)", "remote", or "cache"
    remote_tokens: int  # scored
    local_tokens: int  # scored as zero
    trace: List[Dict[str, object]] = field(default_factory=list)
    cached: bool = False  # served from the query cache (zero remote tokens)
    cache_hit_type: str = "none"  # none | exact | semantic
    preflight_tokens_saved: int = 0  # prompt tokens compression removed from a live call
    remote_tokens_avoided: int = 0  # remote tokens a cache hit avoided spending
    prompt_tokens_before: int = 0  # real-tokenizer count of the raw query
    prompt_tokens_after: int = 0  # prompt tokens actually billed remotely (0 = no remote call)
    model_used: str = ""  # concrete model that answered, or "cache"
    est_tokens_avoided_routing: int = 0  # metrics-only: cheap-tier answer's token proxy
    routing_tier: str = ""  # metrics-only: "local" | "cheap_remote"

    def as_dict(self) -> Dict[str, object]:
        return {
            "answer": self.answer,
            "route": self.route,
            "remote_tokens": self.remote_tokens,
            "local_tokens": self.local_tokens,
            "trace": self.trace,
            "cached": self.cached,
            "cache_hit_type": self.cache_hit_type,
            "preflight_tokens_saved": self.preflight_tokens_saved,
            "remote_tokens_avoided": self.remote_tokens_avoided,
            "prompt_tokens_before": self.prompt_tokens_before,
            "prompt_tokens_after": self.prompt_tokens_after,
            "model_used": self.model_used,
        }


class Agent:
    """Routing agent. Models are injected so it stays testable and swappable.

    ``local_model`` and ``remote_model`` are ``None``-able: this lets the eval
    harness exercise pure routing decisions (``classify`` only) with no GPU or
    API key. When a path is taken but its model is missing, the agent records a
    clear note in the trace instead of crashing.
    """

    def __init__(
        self,
        config: Config,
        classifier: Optional[Classifier] = None,
        verifier: Optional[Verifier] = None,
        local_model=None,
        remote_model=None,
        cheap_remote_model=None,
        cache: Optional[QueryCache] = None,
        optimizer: Optional[PreflightOptimizer] = None,
        ids: Optional[RegexIDS] = None,
        executor: Optional[Executor] = None,
        metrics: Optional[MetricsRegistry] = None,
    ):
        self.config = config
        self.classifier = classifier or HeuristicClassifier(
            length_cutoff=config.complexity_length_cutoff
        )
        self.verifier = verifier or HeuristicVerifier()
        self.local_model = local_model
        self.remote_model = remote_model
        # Cheap-remote tier: a smaller Fireworks model that answers "easy"
        # tasks when no free local model is available (e.g. no GPU). Billed,
        # but at a fraction of the main model's price.
        self.cheap_remote_model = cheap_remote_model

        # One record per request; snapshot served by GET /metrics.
        self.metrics = metrics if metrics is not None else MetricsRegistry(
            main_model=config.remote_model,
            cheap_model=getattr(config, "cheap_remote_model", ""),
            default_price_per_1m=getattr(config, "remote_price_per_1m_usd", 0.20),
        )

        # Token-saving layers. Each honours its config flag but can also be
        # injected explicitly (e.g. a shared cache across requests, or a
        # smarter compressor). Pass the flag off => attribute is None.
        if cache is not None:
            self.cache = cache
        else:
            # Semantic (MiniLM + faiss) or exact per CACHE_BACKEND; the
            # semantic backend degrades to exact when the stack is missing.
            self.cache = create_cache(config) if config.cache_enabled else None

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = PreflightOptimizer() if config.preflight_enabled else None

        if ids is not None:
            self.ids = ids
        else:
            self.ids = RegexIDS(config.ids_threshold) if config.ids_enabled else None

        # Executor: deterministic math (sympy) + verified code (execution),
        # answered at ZERO remote tokens ahead of any model call.
        if executor is not None:
            self.executor = executor
        else:
            self.executor = Executor() if getattr(config, "executor_enabled", True) else None

    # -- model calls (isolated so subclasses/mocks can override) ----------- #
    def _call_local(self, query: str):
        if self.local_model is None:
            return None
        return self.local_model.generate(query)

    def _self_consistency(self, query: str, first, trace: List[Dict[str, object]]):
        """Sample the local model N times; trust only on strict-majority agreement.

        Returns ``(trusted, answer, extra_local_tokens)``. Disabled (returns the
        first answer, trusted) when ``self_consistency_samples <= 1`` or the local
        model can't sample (``generate_samples`` absent) — so a shape-verified
        answer still passes, but a *confidently-wrong* one whose resamples diverge
        gets caught before it is trusted or cached.
        """

        n = int(getattr(self.config, "self_consistency_samples", 1) or 1)
        if n <= 1 or not hasattr(self.local_model, "generate_samples"):
            return True, first.text, 0
        try:
            extra = self.local_model.generate_samples(query, n - 1)
        except Exception as exc:  # noqa: BLE001 — probing must never break routing
            trace.append({"step": "self-consistency", "status": f"skipped: {exc}"})
            return True, first.text, 0

        answers = [first.text] + [getattr(r, "text", "") for r in extra]
        extra_tokens = sum(int(getattr(r, "tokens", 0) or 0) for r in extra)
        trusted, majority, ratio = _agreement(answers)
        trace.append({
            "step": "self-consistency",
            "samples": len(answers),
            "agreement": round(ratio, 2),
            "trusted": trusted,
            "note": "trust+cache only on majority agreement",
        })
        return trusted, majority, extra_tokens

    def _call_remote(
        self,
        prompt: str,
        trace: List[Dict[str, object]],
        reason: str,
        preflight_saved: int = 0,
        raw_query: str = "",
        model=None,
    ):
        """Call a scored remote model with failover.

        ``prompt`` is already normalized — pre-flight now runs once at the top
        of :meth:`route` (before the cache key), not here. ``preflight_saved``
        is attached to the outcome so savings are only claimed when a billed
        call actually happened. ``model`` overrides the main remote (used for
        the cheap tier); failover uses ``raw_query`` so the local model sees
        the user's original words.

        Returns a :class:`_RemoteOutcome`, or ``None`` when no remote model is
        configured (so callers can pick a sensible fallback route label).
        """

        remote_model = model if model is not None else self.remote_model
        if remote_model is None:
            trace.append({"step": "remote", "status": "unavailable (no remote model configured)"})
            return None
        model_name = getattr(remote_model, "model", "remote")

        # --- Call remote; fail over to the local model on any error. ------- #
        try:
            remote = remote_model.generate(prompt)
        except Exception as exc:  # noqa: BLE001 — failover is intentionally broad
            trace.append({"step": "remote", "model": model_name, "status": f"failed: {exc}"})
            if self.config.failover_enabled and self.local_model is not None:
                # Guard the failover itself: if the local model ALSO throws (GPU
                # OOM, driver error), degrade cleanly instead of letting the
                # exception 500 the request — the mirror of the easy-path guard.
                try:
                    local = self.local_model.generate(raw_query or prompt)
                except Exception as local_exc:  # noqa: BLE001 — double failure must not crash
                    trace.append({
                        "step": "failover-local",
                        "status": f"failed: {local_exc}",
                        "note": "remote AND local both failed; returning degraded result",
                    })
                    # failed_over=True keeps this degraded answer OUT of the cache.
                    return _RemoteOutcome(_DEGRADED_BOTH, 0, 0, 0, 0, True, False)
                trace.append({
                    "step": "failover-local",
                    "tokens": local.tokens,
                    "note": "remote failed; served from local model (free)",
                })
                return _RemoteOutcome(local.text, 0, local.tokens, 0, 0, True, True)
            # Remote failed and there is no local fallback: return an explicit
            # degraded result, never a silent empty answer.
            trace.append({
                "step": "failover-local",
                "status": "unavailable (no local model configured)",
                "note": "remote failed and no local fallback; returning degraded result",
            })
            return _RemoteOutcome(_DEGRADED_NO_LOCAL, 0, 0, 0, 0, True, False)

        trace.append({
            "step": "remote",
            "model": model_name,
            "tokens": remote.tokens,
            "prompt_tokens": getattr(remote, "prompt_tokens", 0),
            "completion_tokens": getattr(remote, "completion_tokens", 0),
            "reason": reason,
        })
        return _RemoteOutcome(
            remote.text,
            remote.tokens,
            0,
            preflight_saved,
            getattr(remote, "prompt_tokens", 0),
            False,
            True,
        )

    def _store(self, key_text: str, result: "RouteResult", cacheable: bool = True) -> "RouteResult":
        """Cache a *trusted* answer so a (near-)duplicate query later costs zero.

        ``key_text`` is the **normalized** prompt (post pre-flight), matching
        the key used at lookup time so a greeting-only variant hits later.

        ``cacheable=False`` keeps known-degraded answers out of the cache:
        failover answers (weak local model standing in for the remote on a hard
        query) and local answers the verifier flagged as low-confidence. Caching
        those would serve a bad answer forever — and the semantic backend would
        spread it to paraphrases too.
        """

        if cacheable and self.cache is not None and result.answer and not result.cached:
            self.cache.put(key_text, {"answer": result.answer, "remote_tokens": result.remote_tokens})
        return result

    @staticmethod
    def _failover_route(base: str, outcome: "_RemoteOutcome") -> str:
        return "remote->local (failover)" if outcome.failed_over else base

    def _remote_route_result(
        self,
        outcome: "_RemoteOutcome",
        base_route: str,
        prior_local: int,
        prior_remote: int,
        prompt_tokens_before: int,
        model_name: str,
        trace: List[Dict[str, object]],
    ) -> "RouteResult":
        """Build a RouteResult from a remote outcome, honouring failover.

        On failover the local model produced the answer, so ``model_used`` is
        the local model and no prompt tokens were billed remotely.

        A *degraded* outcome (``ok`` is False — every tier failed) must not imply
        that any model produced the answer: it is labelled ``route="degraded"``
        with an empty ``model_used`` instead of ``"…(failover)"``/<local model>.
        """

        if not outcome.ok:
            route = "degraded"
            model_used = ""
        else:
            route = self._failover_route(base_route, outcome)
            model_used = self.config.local_model if outcome.failed_over else model_name

        return RouteResult(
            answer=outcome.text,
            route=route,
            remote_tokens=prior_remote + outcome.remote_tokens,
            local_tokens=prior_local + outcome.local_tokens,
            trace=trace,
            preflight_tokens_saved=outcome.preflight_tokens_saved,
            prompt_tokens_before=prompt_tokens_before,
            prompt_tokens_after=outcome.prompt_tokens,
            model_used=model_used,
        )

    # -- main entry point --------------------------------------------------- #
    def route(self, query: str) -> RouteResult:
        """Route one query, then fold the outcome into the metrics registry."""

        result = self._route(query)
        if self.metrics is not None:
            self.metrics.record(RequestRecord(
                route=result.route,
                model_routed_to=result.model_used,
                cache_hit_type=result.cache_hit_type,
                prompt_tokens_before=result.prompt_tokens_before,
                prompt_tokens_after=result.prompt_tokens_after,
                tokens_saved_preflight=result.preflight_tokens_saved,
                tokens_avoided_cache=result.remote_tokens_avoided,
                est_tokens_avoided_routing=result.est_tokens_avoided_routing,
                routing_tier=result.routing_tier,
                remote_tokens_spent=result.remote_tokens,
            ))
        return result

    def _route(self, query: str) -> RouteResult:
        trace: List[Dict[str, object]] = []
        remote_tokens = 0
        local_tokens = 0
        # Prefer the concrete model's own name (e.g. the keyless stub reports
        # "stub/echo") so model_used and the trace agree; fall back to config.
        remote_name = getattr(self.remote_model, "model", None) or self.config.remote_model
        local_name = self.config.local_model

        # ---- Pre-flight FIRST: normalize before the cache key ------------- #
        # Runs ahead of the cache lookup so two prompts differing only by a
        # greeting or "please" collapse to one normalized text and share the
        # same exact-cache entry. Compression savings are still only *claimed*
        # (preflight_tokens_saved) when a billed remote call actually happens.
        prompt = query
        preflight_saved = 0
        if self.optimizer is not None:
            opt = self.optimizer.optimize(query)
            prompt = opt.text or query  # never let compression empty the prompt
            preflight_saved = opt.tokens_saved
            prompt_tokens_before = opt.original_tokens
            if opt.saved_chars > 0:
                trace.append({
                    "step": "preflight",
                    "saved_chars": opt.saved_chars,
                    "saved_pct": opt.saved_pct,
                    "tokens_saved": preflight_saved,
                    "removed": opt.removed,
                })
        else:
            prompt_tokens_before = count_tokens(query)

        # ---- Cache on the NORMALIZED prompt: duplicates cost zero ---------- #
        # Still ahead of the classifier: with CLASSIFIER=llm the judge is a
        # real local-model call (seconds), while a cache lookup is ~ms.
        if self.cache is not None:
            if hasattr(self.cache, "get_with_info"):
                hit, hit_type = self.cache.get_with_info(prompt)
            else:  # injected custom cache with the plain get/put interface
                hit = self.cache.get(prompt)
                hit_type = "exact" if hit is not None else "none"
            if hit is not None:
                logger.info(
                    "route: tier=cache reason=%s hit_type=%s hit_rate=%s tokens_avoided=%s",
                    "duplicate query served from cache",
                    hit_type,
                    self.cache.stats.hit_rate,
                    int(hit.get("remote_tokens", 0)),
                )
                trace.append({
                    "step": "cache",
                    "status": f"hit ({hit_type})",
                    "note": "served from cache; zero remote tokens",
                    "hit_rate": self.cache.stats.hit_rate,
                    "semantic_hits": getattr(self.cache.stats, "semantic_hits", 0),
                })
                return RouteResult(
                    answer=hit["answer"],
                    route="cache",
                    remote_tokens=0,
                    local_tokens=0,
                    trace=trace,
                    cached=True,
                    cache_hit_type=hit_type,
                    remote_tokens_avoided=int(hit.get("remote_tokens", 0)),
                    prompt_tokens_before=prompt_tokens_before,
                    model_used="cache",
                )
            trace.append({"step": "cache", "status": "miss"})

        # ---- Executor: deterministic math / verified code, ZERO remote ---- #
        # Runs ahead of the classifier so a math/code task never leaks to a
        # guessing local answer or an unnecessary remote call. Math is solved by
        # sympy; code is generated by the free local model then *executed* to
        # verify before we trust it. A declined attempt just falls through to
        # normal routing (recorded in the trace).
        if self.executor is not None:
            code_gen = (
                (lambda p: self.local_model.generate(p))
                if self.local_model is not None else None
            )
            ex = self.executor.try_solve(query, code_generate=code_gen)
            if ex.solved:
                logger.info("route: tier=executor(%s) reason=%s", ex.kind, ex.method)
                trace.append({
                    "step": "executor", "kind": ex.kind, "method": ex.method,
                    "note": "solved deterministically; zero remote tokens",
                })
                # Verified answer -> safe to cache (math is exact; code passed
                # execution checks), unlike a shape-only-verified local guess.
                return self._store(prompt, RouteResult(
                    answer=ex.answer,
                    route=f"executor({ex.kind})",
                    remote_tokens=0,
                    local_tokens=local_tokens + ex.local_tokens,
                    trace=trace,
                    prompt_tokens_before=prompt_tokens_before,
                    est_tokens_avoided_routing=max(ex.local_tokens, 1),
                    routing_tier="local",
                    model_used=f"executor:{ex.kind}",
                ))
            if ex.kind:  # detected as math/code but declined -> note why, keep routing
                trace.append({
                    "step": "executor", "kind": ex.kind,
                    "status": "declined", "method": ex.method,
                })

        # ---- Classify (judge tokens run locally: free, but tracked) ------- #
        classification: Classification = self.classifier.classify(query)
        judge_tokens = int(getattr(classification, "tokens", 0) or 0)
        local_tokens += judge_tokens
        classify_step: Dict[str, object] = {
            "step": "classify",
            "route": classification.route,
            "difficulty": classification.difficulty,
            "signals": classification.signals,
        }
        if judge_tokens:
            classify_step["judge_tokens"] = judge_tokens
        trace.append(classify_step)

        # ---- IDS: regex sniff for heavy-compute signatures ---------------- #
        # A hard flag forces the remote route even when the classifier said easy.
        route_decision = classification.route
        remote_reason = "classified hard"
        if self.ids is not None:
            verdict = self.ids.inspect(query)
            trace.append({"step": "ids", **verdict.as_dict()})
            if verdict.flagged and route_decision != "remote":
                route_decision = "remote"
                # Name the concrete rules that fired so the log is auditable.
                remote_reason = (
                    "IDS heavy-compute signature "
                    f"[{', '.join(verdict.matches)}] (severity={verdict.severity})"
                )

        # One auditable line per request: chosen pre-generation tier + the
        # concrete reason (classifier signals, or the IDS rule that overrode it).
        logger.info(
            "route: tier=%s reason=%s | classifier=%s(diff=%s) signals=%s",
            route_decision,
            remote_reason if route_decision == "remote" else "classifier easy -> local-first",
            classification.route,
            classification.difficulty,
            classification.signals,
        )

        # ---- Hard path: straight to remote -------------------------------- #
        if route_decision == "remote":
            outcome = self._call_remote(
                prompt, trace, remote_reason,
                preflight_saved=preflight_saved, raw_query=query,
            )
            if outcome is None:
                return self._store(prompt, RouteResult(
                    answer="", route="remote", remote_tokens=0, local_tokens=local_tokens,
                    trace=trace, prompt_tokens_before=prompt_tokens_before,
                ))
            return self._store(prompt, self._remote_route_result(
                outcome, "remote", local_tokens, 0, prompt_tokens_before, remote_name, trace,
            ), cacheable=not outcome.failed_over)  # a failover answer is degraded

        # ---- Easy path: local first, then verify -------------------------- #
        # A local generate() failure (missing/broken GPU stack, OOM, driver
        # error) must degrade to the remote path, never 500 the request — the
        # mirror image of remote->local failover. Treat any error as "local
        # unavailable" and let the block below escalate to the remote tier.
        local = None
        local_failed = False
        try:
            local = self._call_local(query)
        except Exception as exc:  # noqa: BLE001 — local failure escalates, never crashes
            trace.append({
                "step": "local",
                "status": f"failed: {exc}",
                "note": "local model errored; escalating to remote",
            })
            local_failed = True

        if local is None:
            if not local_failed:
                trace.append({"step": "local", "status": "unavailable (no local model configured)"})
            # Cheap-remote tier: prefer a smaller/cheaper Fireworks model over
            # the main one so the router still saves money without a GPU.
            if self.cheap_remote_model is not None:
                outcome = self._call_remote(
                    prompt, trace, "easy task -> cheap remote tier",
                    preflight_saved=preflight_saved, raw_query=query,
                    model=self.cheap_remote_model,
                )
                if outcome is not None and outcome.ok:
                    res = self._remote_route_result(
                        outcome, "cheap-remote", local_tokens, 0,
                        prompt_tokens_before,
                        getattr(self.cheap_remote_model, "model", "cheap-remote"), trace,
                    )
                    if not outcome.failed_over:  # answered cheap instead of main
                        res.est_tokens_avoided_routing = outcome.remote_tokens
                        res.routing_tier = "cheap_remote"
                    return self._store(prompt, res, cacheable=not outcome.failed_over)

            # No cheap tier (or it was unavailable) -> main remote.
            outcome = self._call_remote(
                prompt, trace, "local model unavailable",
                preflight_saved=preflight_saved, raw_query=query,
            )
            if outcome is None:
                return self._store(prompt, RouteResult(
                    answer="", route="local", remote_tokens=0, local_tokens=local_tokens,
                    trace=trace, prompt_tokens_before=prompt_tokens_before,
                ))
            return self._store(prompt, self._remote_route_result(
                outcome, "local->remote (escalated)", local_tokens, 0,
                prompt_tokens_before, remote_name, trace,
            ), cacheable=not outcome.failed_over)  # a failover answer is degraded

        local_tokens += local.tokens
        trace.append({"step": "local", "tokens": local.tokens, "note": "free / scored as zero"})

        verification = self.verifier.verify(query, local.text, self.config.confidence_threshold)
        trace.append({
            "step": "verify",
            "confidence": verification.confidence,
            "threshold": self.config.confidence_threshold,
            "escalate": verification.escalate,
            "reasons": verification.reasons,
        })

        # Trust the local answer only if the shape verifier passed AND the local
        # model is self-consistent across resamples. This stops a confidently-
        # wrong answer from clearing the shape-only check and then poisoning the
        # cache (and its paraphrases) — the resamples of an uncertain answer
        # diverge and force escalation instead.
        trusted_local = False
        answer_text = local.text
        if not verification.escalate:
            trusted_local, answer_text, sc_tokens = self._self_consistency(query, local, trace)
            local_tokens += sc_tokens

        if trusted_local:
            # Answered on the free local tier: routing saved the full main-model
            # price on a token count proxied by the local answer's own tokens.
            return self._store(prompt, RouteResult(
                answer=answer_text,
                route="local",
                remote_tokens=remote_tokens,
                local_tokens=local_tokens,
                trace=trace,
                prompt_tokens_before=prompt_tokens_before,
                est_tokens_avoided_routing=local.tokens,
                routing_tier="local",
                model_used=local_name,
            ))

        # Verifier flagged it, or self-consistency diverged -> escalate to remote.
        outcome = self._call_remote(
            prompt, trace, "low local confidence or self-consistency disagreement",
            preflight_saved=preflight_saved, raw_query=query,
        )
        if outcome is None:
            # Can't escalate; return the local answer but flag it in the trace.
            # NOT cacheable: the verifier just told us this answer is weak.
            return self._store(prompt, RouteResult(
                answer=local.text,
                route="local (wanted escalation, remote unavailable)",
                remote_tokens=remote_tokens,
                local_tokens=local_tokens,
                trace=trace,
                prompt_tokens_before=prompt_tokens_before,
                model_used=local_name,
            ), cacheable=False)
        return self._store(prompt, self._remote_route_result(
            outcome, "local->remote (escalated)", local_tokens, remote_tokens,
            prompt_tokens_before, remote_name, trace,
        ), cacheable=not outcome.failed_over)  # failover here = verifier-flagged local answer
