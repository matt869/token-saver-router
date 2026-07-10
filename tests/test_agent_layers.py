"""Integration tests for the token-saving layers on the Agent.

Covers the pre-flight optimizer, the exact-match cache, the regex IDS override,
and remote->local failover — all with lightweight fake models (no GPU, no key).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_config  # noqa: E402
from app.router.agent import Agent  # noqa: E402


class FakeLocal:
    def __init__(self, text="Local answer.", tokens=10):
        self.text = text
        self.tokens = tokens
        self.calls = []

    def generate(self, prompt, max_new_tokens=None):
        self.calls.append(prompt)
        return SimpleNamespace(text=self.text, tokens=self.tokens)


class FakeRemote:
    def __init__(self, text="Remote answer.", tokens=100):
        self.text = text
        self.tokens = tokens
        self.calls = []

    def generate(self, prompt, max_new_tokens=None):
        self.calls.append(prompt)
        return SimpleNamespace(text=self.text, tokens=self.tokens)


class ExplodingRemote:
    def generate(self, prompt, max_new_tokens=None):
        raise RuntimeError("fireworks is down")


class ExplodingLocal:
    """Local model whose generate() raises (e.g. GPU stack missing / OOM)."""

    def generate(self, prompt, max_new_tokens=None):
        raise RuntimeError("cuda is unavailable")


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
def test_duplicate_query_is_served_from_cache_for_zero_tokens():
    cfg = load_config()
    remote = FakeRemote(tokens=120)
    agent = Agent(config=cfg, local_model=FakeLocal(), remote_model=remote)

    q = "Prove that the square root of 2 is irrational, step by step."
    first = agent.route(q)
    assert first.remote_tokens == 120
    assert first.cached is False

    second = agent.route(q)
    assert second.cached is True
    assert second.route == "cache"
    assert second.remote_tokens == 0
    assert second.cache_hit_type == "exact"
    assert second.remote_tokens_avoided == 120
    # Remote was called exactly once across both routes.
    assert len(remote.calls) == 1


# --------------------------------------------------------------------------- #
# Pre-flight optimizer
# --------------------------------------------------------------------------- #
def test_preflight_strips_filler_from_the_remote_prompt():
    cfg = load_config()
    remote = FakeRemote()
    agent = Agent(config=cfg, local_model=FakeLocal(), remote_model=remote)

    # "prove"/"theorem" route this to remote; politeness must be stripped first.
    agent.route("Please prove this theorem step by step, thank you very much!")
    sent = remote.calls[0]
    assert "please" not in sent.lower()
    assert "thank you" not in sent.lower()


# --------------------------------------------------------------------------- #
# IDS override
# --------------------------------------------------------------------------- #
def test_ids_forces_remote_on_a_code_fence():
    cfg = load_config()
    remote = FakeRemote()
    local = FakeLocal()
    agent = Agent(config=cfg, local_model=local, remote_model=remote)

    # Short fenced code: the classifier alone scores this "local", but the IDS
    # heavy-compute flag forces the remote route.
    result = agent.route("```\nx=1\n```")
    ids_step = next(s for s in result.trace if s.get("step") == "ids")
    assert ids_step["flagged"] is True
    assert result.route.startswith("remote")
    assert len(remote.calls) == 1


# --------------------------------------------------------------------------- #
# Failover
# --------------------------------------------------------------------------- #
def test_remote_failure_fails_over_to_local():
    cfg = load_config()
    local = FakeLocal(text="Served locally.", tokens=15)
    agent = Agent(config=cfg, local_model=local, remote_model=ExplodingRemote())

    result = agent.route("Prove this theorem step by step.")
    assert result.route == "remote->local (failover)"
    assert result.answer == "Served locally."
    assert result.remote_tokens == 0
    assert result.local_tokens == 15
    assert any(s.get("step") == "failover-local" for s in result.trace)


# --------------------------------------------------------------------------- #
# Cache hygiene: degraded answers must never be cached
# --------------------------------------------------------------------------- #
def test_local_generate_failure_escalates_to_remote_instead_of_crashing():
    # An "easy" query would normally be answered locally; if the local model
    # errors at generate() time (broken/missing GPU stack), the request must
    # degrade to the remote path, not raise.
    cfg = load_config()
    remote = FakeRemote(text="Remote saved the day.", tokens=42)
    agent = Agent(config=cfg, local_model=ExplodingLocal(), remote_model=remote)

    result = agent.route("What is the capital of France?")
    assert result.answer == "Remote saved the day."
    assert result.remote_tokens == 42
    # The failure is recorded, and the remote actually served the answer.
    assert any(
        s.get("step") == "local" and str(s.get("status", "")).startswith("failed")
        for s in result.trace
    )
    assert len(remote.calls) == 1


def test_local_generate_failure_with_no_remote_is_handled():
    # Same failure but nothing to escalate to: must return a clean (empty)
    # result, never propagate the exception.
    cfg = load_config()
    agent = Agent(config=cfg, local_model=ExplodingLocal(), remote_model=None)

    result = agent.route("What is the capital of France?")
    assert result.route == "local"
    assert result.remote_tokens == 0
    assert any(
        s.get("step") == "local" and str(s.get("status", "")).startswith("failed")
        for s in result.trace
    )


def test_double_failure_degrades_cleanly_without_raising():
    # Remote raises AND the local failover ALSO raises: the request must return
    # a clean degraded result (non-empty, clearly labeled), never propagate a 500.
    cfg = load_config()
    agent = Agent(
        config=cfg,
        local_model=ExplodingLocal(),    # the failover target also throws
        remote_model=ExplodingRemote(),  # remote throws first
    )
    q = "Prove this theorem step by step."  # routes remote (hard)

    result = agent.route(q)  # must NOT raise
    assert result.answer  # non-empty
    assert "degraded" in result.answer.lower()

    # Labels must not imply a model served the answer (both tiers failed).
    assert result.route == "degraded"
    assert result.model_used == ""

    # The trace shows BOTH failures, in order.
    assert any(
        s.get("step") == "remote" and str(s.get("status", "")).startswith("failed")
        for s in result.trace
    )
    assert any(
        s.get("step") == "failover-local" and str(s.get("status", "")).startswith("failed")
        for s in result.trace
    )

    # A degraded answer must never be cached (a later retry may recover).
    second = agent.route(q)
    assert second.cached is False


def test_failover_answer_is_not_cached():
    cfg = load_config()
    agent = Agent(
        config=cfg,
        local_model=FakeLocal(text="Weak local stand-in answer here.", tokens=15),
        remote_model=ExplodingRemote(),
    )
    q = "Prove this theorem step by step."
    first = agent.route(q)
    assert first.route == "remote->local (failover)"

    # Same query again: must NOT be a cache hit — the failover answer is
    # degraded and remote may have recovered.
    second = agent.route(q)
    assert second.cached is False
    assert second.route == "remote->local (failover)"


def test_low_confidence_local_answer_is_not_cached():
    cfg = load_config()
    # Hedging answer -> verifier escalates; remote unavailable -> flagged result.
    agent = Agent(
        config=cfg,
        local_model=FakeLocal(text="I'm not sure, it depends.", tokens=9),
        remote_model=None,
    )
    q = "What is the capital of France?"
    first = agent.route(q)
    assert first.route == "local (wanted escalation, remote unavailable)"

    second = agent.route(q)
    assert second.cached is False  # the flagged answer was never stored


def test_cache_hit_skips_the_classifier_entirely():
    cfg = load_config()

    class CountingClassifier:
        calls = 0

        def classify(self, query):
            CountingClassifier.calls += 1
            from app.router.classifier import Classification
            return Classification(route="remote", difficulty=1.0, signals={})

    remote = FakeRemote(tokens=50)
    agent = Agent(
        config=cfg, classifier=CountingClassifier(),
        local_model=FakeLocal(), remote_model=remote,
    )
    q = "Prove this theorem."
    agent.route(q)
    assert CountingClassifier.calls == 1
    hit = agent.route(q)
    assert hit.cached is True
    # The expensive classify step never ran for the duplicate.
    assert CountingClassifier.calls == 1
