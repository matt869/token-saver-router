"""Tests for the regex IDS (heavy-compute signature inspector)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.router.ids import RegexIDS  # noqa: E402


def test_flags_fenced_code():
    ids = RegexIDS()
    verdict = ids.inspect("Fix this:\n```python\nprint(1)\n```")
    assert verdict.flagged is True
    assert "fenced_code" in verdict.matches


def test_flags_sql():
    ids = RegexIDS()
    verdict = ids.inspect("Optimize: SELECT id FROM users WHERE active = 1")
    assert verdict.flagged is True
    assert "sql" in verdict.matches


def test_does_not_flag_simple_factual_query():
    ids = RegexIDS()
    verdict = ids.inspect("What is the capital of France?")
    assert verdict.flagged is False
    assert verdict.severity < 0.5


def test_severity_is_clamped():
    ids = RegexIDS()
    verdict = ids.inspect(
        "```code``` prove this theorem step by step with O(n log n) using dynamic programming"
    )
    assert 0.0 <= verdict.severity <= 1.0


def test_threshold_is_configurable():
    strict = RegexIDS(threshold=0.99)
    verdict = strict.inspect("Explain the algorithm complexity here.")
    # A single moderate signal should not clear a near-1.0 threshold.
    assert verdict.flagged is False
