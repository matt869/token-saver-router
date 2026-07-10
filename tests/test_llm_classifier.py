"""Tests for the LLM-as-a-Judge classifier.

The parsing/bias logic is tested with a fake local model so it runs anywhere
(no GPU, no download). The real-model integration test is guarded and skips
when no CUDA device is present.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.router.classifier import (  # noqa: E402
    Classifier,
    Decision,
    HeuristicClassifier,
    LLMClassifier,
)


class FakeJudge:
    """Stands in for LocalModel; records the generate() call."""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = []

    def generate(self, prompt, max_new_tokens=None):
        self.calls.append({"prompt": prompt, "max_new_tokens": max_new_tokens})
        return SimpleNamespace(text=self.reply, tokens=5)


class ExplodingJudge:
    def generate(self, prompt, max_new_tokens=None):
        raise RuntimeError("OOM")


# A query in the heuristic's gray zone (0.2 < difficulty < 0.8): the judge IS
# consulted for it. Confidently-easy or -hard queries skip the judge entirely.
GRAY_QUERY = "Prove the Riemann hypothesis."


def test_remote_verdict_routes_remote():
    clf = LLMClassifier(local_model=FakeJudge("REMOTE"))
    d = clf.classify(GRAY_QUERY)
    assert d.route == "remote"
    assert d.difficulty > 0.5  # duck-type compat with Classification


def test_local_verdict_routes_local():
    judge = FakeJudge("LOCAL")
    clf = LLMClassifier(local_model=judge)
    d = clf.classify(GRAY_QUERY)
    assert d.route == "local"
    assert len(judge.calls) == 1  # gray zone -> judge consulted


def test_verdict_parsing_tolerates_punctuation_and_case():
    clf = LLMClassifier(local_model=FakeJudge("remote."))
    assert clf.classify(GRAY_QUERY).route == "remote"


def test_ambiguous_reply_biases_local():
    # Anything that doesn't cleanly parse as REMOTE defaults to LOCAL: the
    # downstream verifier still escalates a bad local answer.
    clf = LLMClassifier(local_model=FakeJudge("Well, it depends on the query"))
    d = clf.classify(GRAY_QUERY)
    assert d.route == "local"
    assert "defaulting LOCAL" in d.reason


def test_empty_reply_biases_local():
    clf = LLMClassifier(local_model=FakeJudge(""))
    assert clf.classify(GRAY_QUERY).route == "local"


def test_judge_output_is_capped_at_5_tokens():
    judge = FakeJudge("LOCAL")
    LLMClassifier(local_model=judge).classify(GRAY_QUERY)
    assert judge.calls[0]["max_new_tokens"] == 5


def test_judge_reports_its_local_token_cost():
    d = LLMClassifier(local_model=FakeJudge("LOCAL")).classify(GRAY_QUERY)
    assert d.tokens == 5  # FakeJudge reports 5 tokens; free, but tracked


def test_judge_failure_falls_back_to_heuristic():
    clf = LLMClassifier(local_model=ExplodingJudge())
    d = clf.classify(GRAY_QUERY)
    assert d.route in ("local", "remote")  # heuristic decided, no crash
    assert "heuristic fallback" in d.reason


# --------------------------------------------------------------------------- #
# Gray-zone gating: obvious queries never pay for a judge call
# --------------------------------------------------------------------------- #
def test_confidently_easy_query_skips_judge():
    judge = FakeJudge("REMOTE")  # would say REMOTE — but must not be asked
    clf = LLMClassifier(local_model=judge)
    d = clf.classify("What is the capital of France?")
    assert d.route == "local"
    assert judge.calls == []
    assert "judge skipped" in d.reason


def test_confidently_hard_query_skips_judge():
    judge = FakeJudge("LOCAL")  # would say LOCAL — but must not be asked
    clf = LLMClassifier(local_model=judge)
    d = clf.classify(
        "Prove and derive and analyze and optimize and refactor this algorithm step by step"
    )
    assert d.route == "remote"
    assert judge.calls == []
    assert "judge skipped" in d.reason


def test_no_local_model_falls_back_to_heuristic():
    clf = LLMClassifier(local_model=None)
    result = clf.classify("What is the capital of France?")
    assert result.route == "local"  # HeuristicClassifier's decision


def test_llm_classifier_satisfies_protocol():
    assert isinstance(LLMClassifier(local_model=FakeJudge("LOCAL")), Classifier)


def test_decision_exposes_classification_shape():
    d = Decision(route="local", score=0.2, reason="test")
    assert d.difficulty == 0.2
    assert d.signals["judge"] == "llm"


# --------------------------------------------------------------------------- #
# Real-model integration (GPU only)
# --------------------------------------------------------------------------- #
def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.mark.skipif(not _cuda_available(), reason="needs a CUDA/ROCm GPU")
def test_real_judge_end_to_end():
    from app.config import load_config
    from app.models.local_model import LocalModel

    cfg = load_config()
    local = LocalModel(cfg.local_model, device=cfg.device, max_new_tokens=cfg.max_new_tokens)
    clf = LLMClassifier(local_model=local)
    easy = clf.classify("What is the capital of France?")
    assert easy.route in ("local", "remote")  # sane verdict, no crash
