"""Self-consistency: trust/cache a local answer only when resamples agree.

A confidently-wrong answer that passes the shape-only verifier must NOT be
trusted or cached if the local model's resamples diverge — it escalates instead.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_config  # noqa: E402
from app.router.agent import Agent  # noqa: E402
from app.router.classifier import HeuristicClassifier  # noqa: E402

_Q = "What is the capital of France?"  # easy path -> local first


class SamplingLocal:
    """Local model whose first answer passes the shape check; resamples are
    scripted so we can force agreement or divergence."""

    def __init__(self, first, samples):
        self.first = first
        self.samples = samples
        self.sample_calls = 0

    def generate(self, prompt, max_new_tokens=None):
        return SimpleNamespace(text=self.first, tokens=10)

    def generate_samples(self, prompt, n, max_new_tokens=None):
        self.sample_calls += 1
        return [SimpleNamespace(text=s, tokens=8) for s in self.samples[:n]]


class FakeRemote:
    model = "fake/remote"

    def __init__(self):
        self.calls = []

    def generate(self, prompt, max_new_tokens=None):
        self.calls.append(prompt)
        return SimpleNamespace(text="Paris (remote-verified).", tokens=100,
                               prompt_tokens=20, completion_tokens=80)


def _agent(local, remote=None):
    a = Agent(config=load_config(), classifier=HeuristicClassifier(24),
              local_model=local, remote_model=remote or FakeRemote())
    return a


def test_agreement_trusts_and_caches():
    local = SamplingLocal("The capital of France is Paris.",
                          ["The capital of France is Paris.", "The capital of France is Paris."])
    remote = FakeRemote()
    agent = _agent(local, remote)
    r = agent.route(_Q)
    assert r.route == "local"
    assert r.answer == "The capital of France is Paris."
    assert local.sample_calls == 1  # self-consistency actually ran
    assert len(remote.calls) == 0   # never escalated
    sc = next(s for s in r.trace if s.get("step") == "self-consistency")
    assert sc["trusted"] is True and sc["samples"] == 3
    # trusted answer is cached -> repeat is a cache hit
    assert agent.route(_Q).cached is True


def test_majority_2_of_3_trusts():
    local = SamplingLocal("The capital of France is Paris.",
                          ["The capital of France is Paris.", "The capital of France is Lyon."])
    agent = _agent(local)
    r = agent.route(_Q)
    assert r.route == "local"
    assert r.answer == "The capital of France is Paris."  # majority


def test_divergence_escalates_and_does_not_cache_local():
    local = SamplingLocal("The capital of France is Paris.",
                          ["The capital of France is London.", "The capital of France is Berlin."])
    remote = FakeRemote()
    agent = _agent(local, remote)
    r = agent.route(_Q)
    assert r.route == "local->remote (escalated)"
    assert r.answer == "Paris (remote-verified)."   # remote answer, not the local guess
    assert r.remote_tokens == 100
    assert len(remote.calls) == 1
    sc = next(s for s in r.trace if s.get("step") == "self-consistency")
    assert sc["trusted"] is False
    # the wrong local answer must NOT have been cached
    second = agent.route(_Q)
    assert second.cached is True  # the REMOTE (correct) answer is cached...
    assert "remote" in second.answer.lower()  # ...not the local guess


def test_unsupported_local_model_skips_self_consistency():
    # A plain fake with no generate_samples() behaves exactly as before (trusts).
    class PlainLocal:
        def generate(self, prompt, max_new_tokens=None):
            return SimpleNamespace(text="The capital of France is Paris.", tokens=10)

    agent = _agent(PlainLocal())
    r = agent.route(_Q)
    assert r.route == "local"
    assert not any(s.get("step") == "self-consistency" for s in r.trace)
