"""Tests for the second-pass preflight upgrades: sentence dedup, context-aware
filler removal, and real-tokenizer counts."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.optimizer import PreflightOptimizer  # noqa: E402
from app.tokens import count_tokens  # noqa: E402


def test_deduplicates_repeated_instruction_sentences():
    opt = PreflightOptimizer()
    boilerplate = "Always respond in valid JSON with no extra commentary. "
    prompt = boilerplate + "Summarize the report. " + boilerplate
    result = opt.optimize(prompt)
    # The repeated boilerplate sentence appears once, not twice.
    assert result.text.lower().count("always respond in valid json") == 1
    assert "dedup-sentence" in result.removed
    assert result.tokens_saved > 0


def test_dedup_keeps_short_sentences_untouched():
    # Short repeats ("Yes.") are structural, never deduped.
    opt = PreflightOptimizer()
    result = opt.optimize("Yes. Do it. Yes. Do it.")
    assert result.text.count("Yes.") == 2


def test_just_survives_as_an_adjective():
    # "just" is only filler after an auxiliary/pronoun, never as an adjective.
    opt = PreflightOptimizer()
    result = opt.optimize("Explain the concept of a just war in ethics.")
    assert "just war" in result.text
    # ... but "could just check" loses the filler "just".
    result2 = opt.optimize("Could you just check the spelling here.")
    assert "just" not in result2.text.lower()


def test_single_quoted_subject_survives():
    opt = PreflightOptimizer()
    result = opt.optimize("Please translate 'please help me' into German.")
    assert "'please help me'" in result.text
    assert result.text.lower().startswith("translate")


def test_apostrophe_in_word_does_not_open_a_quote():
    # "don't" must not be treated as an opening single quote that protects the
    # rest of the prompt from compression.
    opt = PreflightOptimizer()
    result = opt.optimize("Please don't    add   extra spaces here.")
    assert "  " not in result.text  # whitespace still collapsed
    assert "please" not in result.text.lower()


def test_token_counts_are_populated():
    opt = PreflightOptimizer()
    result = opt.optimize("Please   summarize   this.")
    assert result.original_tokens == count_tokens("Please   summarize   this.")
    assert result.optimized_tokens == count_tokens(result.text)
