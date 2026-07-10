"""Tests for the pre-flight token-saving optimizer."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.optimizer import PreflightOptimizer  # noqa: E402


def test_strips_please_and_thanks():
    opt = PreflightOptimizer()
    result = opt.optimize("Please summarize this. Thank you very much!")
    assert "please" not in result.text.lower()
    assert "thank you" not in result.text.lower()
    assert result.saved_chars > 0


def test_collapses_whitespace_and_blank_lines():
    opt = PreflightOptimizer()
    result = opt.optimize("hello    world\n\n\n\nfoo   bar")
    assert "    " not in result.text
    assert "\n\n\n" not in result.text


def test_strips_leading_greeting():
    opt = PreflightOptimizer()
    result = opt.optimize("Hi there, what is the capital of France?")
    assert result.text.lower().startswith("what is")


def test_never_touches_code_blocks():
    opt = PreflightOptimizer()
    code = "Please fix this:\n```python\ndef  f( x ):\n    return    x\n```\nThanks!"
    result = opt.optimize(code)
    # The exact 4-space indentation and double spaces inside the fence survive.
    assert "def  f( x ):" in result.text
    assert "    return    x" in result.text
    # But the surrounding politeness is gone.
    assert "please" not in result.text.lower()
    assert "thanks" not in result.text.lower()


def test_never_touches_quoted_strings():
    # A quoted phrase is often the task's subject — it must survive verbatim.
    opt = PreflightOptimizer()
    result = opt.optimize('Please translate "please help me kindly" into French. Thanks!')
    assert '"please help me kindly"' in result.text
    # ... while the surrounding politeness still goes.
    assert result.text.lower().startswith("translate")


def test_no_orphan_punctuation_after_filler_removal():
    opt = PreflightOptimizer()
    result = opt.optimize("Can you summarize this for me? Thanks a lot!")
    assert "?!" not in result.text
    result2 = opt.optimize("Thanks! Now summarize the report.")
    assert not result2.text.startswith("!")


def test_reports_savings_and_token_estimate():
    opt = PreflightOptimizer()
    result = opt.optimize("Please    please    kindly help.   ")
    assert result.optimized_chars < result.original_chars
    assert 0.0 < result.saved_pct <= 100.0
    # tokens_saved comes from a real tokenizer measured before/after.
    assert result.tokens_saved > 0
    assert result.optimized_tokens <= result.original_tokens


def test_idempotent_on_clean_prompt():
    opt = PreflightOptimizer()
    clean = "Summarize the attached report in three bullet points."
    result = opt.optimize(clean)
    assert result.text == clean
    assert result.saved_chars == 0


def test_empty_prompt_is_safe():
    opt = PreflightOptimizer()
    result = opt.optimize("")
    assert result.text == ""
    assert result.saved_pct == 0.0
    assert result.tokens_saved == 0
