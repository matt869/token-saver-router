"""Tests for the --validate guardrail: raw vs optimized answer diffing."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.eval import run_eval  # noqa: E402


def _run_validate(monkeypatch, capsys, answer_fn, tasks, embed=None):
    monkeypatch.setattr(run_eval, "_build_answer_fn", lambda co: ("stub-model", answer_fn))
    monkeypatch.setattr(run_eval, "_load_embedder", lambda: embed)
    monkeypatch.setattr(run_eval, "load_tasks", lambda p: tasks)
    run_eval.validate(Path("ignored.json"), classify_only=False)
    return capsys.readouterr().out


def test_validate_reports_all_identical_when_model_ignores_prompt(monkeypatch, capsys):
    # A model that returns a constant answer -> optimization never changes output.
    tasks = [{"id": 1, "query": "Please summarize this. Thanks!"},
             {"id": 2, "query": "Hi there, what is 2+2?"}]
    out = _run_validate(monkeypatch, capsys, lambda p: "constant", tasks)
    assert "identical answers   : 2/2" in out
    assert "degraded" in out
    assert "0/2" in out  # zero degraded
    assert "preserved every answer" in out


def test_validate_flags_and_attributes_a_degradation(monkeypatch, capsys):
    # A pathological model that echoes the prompt -> optimized != raw, so the
    # guardrail must flag it and attribute it to the removed labels.
    tasks = [{"id": 7, "query": "Please kindly summarize this long report for me. Thank you very much!"}]
    out = _run_validate(monkeypatch, capsys, lambda p: p, tasks)  # jaccard sim path
    assert "degraded" in out
    assert "worst offenders" in out
    assert "attribute->" in out  # attribution line printed
    assert "please" in out.lower()  # the removed label surfaced for the offender


def test_validate_uses_cosine_when_embedder_present(monkeypatch, capsys):
    import numpy as np

    # Fake embedder: identical strings -> identical unit vectors (cosine 1.0).
    def fake_embed(texts):
        return np.array([[1.0, 0.0] for _ in texts], dtype="float32")

    tasks = [{"id": 1, "query": "Please help me."}]
    out = _run_validate(monkeypatch, capsys, lambda p: "same", tasks, embed=fake_embed)
    assert "cosine=1.000" in out
