"""Unit tests for the classifier and routing logic.

These tests run WITHOUT a GPU and WITHOUT any API key: they exercise only the
heuristic classifier and the pure routing decisions (with no models injected).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when pytest is run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_config  # noqa: E402
from app.router.agent import Agent, HeuristicVerifier  # noqa: E402
from app.router.classifier import Classification, Classifier, HeuristicClassifier  # noqa: E402


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #
def test_easy_factual_query_routes_local():
    clf = HeuristicClassifier(length_cutoff=24)
    result = clf.classify("What is the capital of France?")
    assert result.route == "local"
    assert 0.0 <= result.difficulty < 0.5


def test_greeting_routes_local():
    clf = HeuristicClassifier(length_cutoff=24)
    result = clf.classify("Hello there!")
    assert result.route == "local"


def test_hard_proof_query_routes_remote():
    clf = HeuristicClassifier(length_cutoff=24)
    result = clf.classify(
        "Prove that the square root of 2 is irrational and explain each step."
    )
    assert result.route == "remote"
    assert result.difficulty >= 0.5


def test_system_design_query_routes_remote():
    clf = HeuristicClassifier(length_cutoff=24)
    result = clf.classify(
        "Design a distributed rate limiter and analyze its trade-offs."
    )
    assert result.route == "remote"


def test_long_query_exceeding_cutoff_routes_remote():
    clf = HeuristicClassifier(length_cutoff=8)
    long_query = "please tell me about the history and culture and food and music of this region"
    result = clf.classify(long_query)
    assert result.signals["n_words"] > result.signals["length_cutoff"]
    assert result.route == "remote"


def test_difficulty_is_always_normalised():
    clf = HeuristicClassifier(length_cutoff=24)
    for q in ["hi", "What is 2+2?", "Prove and derive and analyze and optimize and refactor this algorithm step by step"]:
        result = clf.classify(q)
        assert 0.0 <= result.difficulty <= 1.0


def test_empty_query_is_safe():
    clf = HeuristicClassifier(length_cutoff=24)
    result = clf.classify("")
    assert result.route == "local"
    assert result.difficulty == 0.0


def test_length_cutoff_is_respected():
    strict = HeuristicClassifier(length_cutoff=3)
    lenient = HeuristicClassifier(length_cutoff=50)
    query = "tell me a little bit about cats and dogs please"
    # A tight cutoff makes the same query look harder than a lenient one.
    assert strict.classify(query).difficulty >= lenient.classify(query).difficulty


def test_heuristic_classifier_satisfies_protocol():
    clf = HeuristicClassifier()
    assert isinstance(clf, Classifier)


def test_classification_returns_expected_shape():
    result = HeuristicClassifier().classify("What is Python?")
    assert isinstance(result, Classification)
    assert isinstance(result.signals, dict)
    assert "hard_hits" in result.signals


# --------------------------------------------------------------------------- #
# Swappable classifier (fine-tuned model stand-in)
# --------------------------------------------------------------------------- #
class AlwaysRemoteClassifier:
    """A stand-in for a swapped-in classifier — no keyword logic at all."""

    def classify(self, query: str) -> Classification:
        return Classification(route="remote", difficulty=1.0, signals={"stub": True})


def test_agent_accepts_a_swapped_classifier():
    cfg = load_config()
    agent = Agent(config=cfg, classifier=AlwaysRemoteClassifier())
    # No models injected: remote path is "unavailable" but must not crash.
    result = agent.route("What is the capital of France?")
    clf_step = next(s for s in result.trace if s.get("step") == "classify")
    assert clf_step["route"] == "remote"
    assert result.route == "remote"


# --------------------------------------------------------------------------- #
# Verifier
# --------------------------------------------------------------------------- #
def test_verifier_escalates_on_empty_answer():
    v = HeuristicVerifier()
    result = v.verify("q", "", threshold=0.6)
    assert result.escalate is True
    assert result.confidence == 0.0


def test_verifier_escalates_on_hedging():
    v = HeuristicVerifier()
    result = v.verify("q", "I'm not sure, it depends.", threshold=0.6)
    assert result.escalate is True


def test_verifier_accepts_confident_answer():
    v = HeuristicVerifier()
    result = v.verify("What is 2+2?", "The answer is 4.", threshold=0.6)
    assert result.escalate is False
    assert result.confidence >= 0.6


# --------------------------------------------------------------------------- #
# Routing decisions with no models (GPU/key-free)
# --------------------------------------------------------------------------- #
def test_agent_routes_without_models_and_tracks_zero_tokens():
    cfg = load_config()
    agent = Agent(config=cfg)  # no local/remote models
    result = agent.route("What is the capital of France?")
    # Easy query -> local path chosen, but no local model -> falls back with 0 tokens.
    assert result.remote_tokens == 0
    assert result.local_tokens == 0
    # Cache runs first now; the classify step must still be present.
    assert any(s.get("step") == "classify" for s in result.trace)
