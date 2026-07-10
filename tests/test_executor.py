"""Unit tests for the deterministic executor (math via sympy, code via execution).

Confirms answers are actually CORRECT, not just non-empty, and that unverifiable
code declines (so the router escalates) instead of returning a guess.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.router.executor import Executor  # noqa: E402


# --------------------------------------------------------------------------- #
# Math
# --------------------------------------------------------------------------- #
def test_solves_linear_equation_the_named_leak():
    r = Executor().try_solve("Solve for x: 7x - 3 = 4x + 18")
    assert r.solved and r.kind == "math"
    assert r.answer == "x = 7"  # correct, not just non-empty


def test_percent_of():
    r = Executor().try_solve("What is 15% of 200?")
    assert r.solved and r.answer == "30"


def test_plain_arithmetic():
    r = Executor().try_solve("Compute 12 * 47 + 5")
    assert r.solved and r.answer == "569"


def test_evaluates_equality_without_variable():
    r = Executor().try_solve("Evaluate 2**10")
    assert r.solved and r.answer == "1024"


def test_quadratic_returns_both_roots():
    r = Executor().try_solve("Solve x^2 - 5x + 6 = 0")
    assert r.solved and "2" in r.answer and "3" in r.answer


def test_non_math_fact_is_not_grabbed():
    # No operator/expression -> executor must decline (it's a lookup, not compute).
    assert Executor().try_solve("What is the capital of France?").solved is False
    assert Executor().try_solve("How many days are there in a leap year?").solved is False


# --------------------------------------------------------------------------- #
# Code
# --------------------------------------------------------------------------- #
_GOOD_PALINDROME = SimpleNamespace(
    text="```python\ndef is_palindrome(s):\n    return s == s[::-1]\n```", tokens=20,
)
_WRONG_PALINDROME = SimpleNamespace(
    text="def is_palindrome(s):\n    return True\n", tokens=12,
)

_PALINDROME_Q = (
    "Write a function is_palindrome(s) that returns True for a palindrome. "
    "For example is_palindrome('racecar') == True and is_palindrome('hello') == False."
)


def test_code_verified_by_execution_passes():
    r = Executor().try_solve(_PALINDROME_Q, code_generate=lambda q: _GOOD_PALINDROME)
    assert r.solved and r.kind == "code"
    assert "def is_palindrome" in r.answer
    assert r.local_tokens == 20  # generator tokens tracked (free)


def test_wrong_code_declines_so_router_escalates():
    r = Executor().try_solve(_PALINDROME_Q, code_generate=lambda q: _WRONG_PALINDROME)
    assert r.solved is False
    assert r.kind == "code"  # detected as code, but failed verification


def test_code_without_checks_declines():
    r = Executor().try_solve(
        "Write a function foo(x) that does something clever.",
        code_generate=lambda q: _GOOD_PALINDROME,
    )
    assert r.solved is False


def test_code_without_generator_declines():
    r = Executor().try_solve(_PALINDROME_Q, code_generate=None)
    assert r.solved is False


def test_disabled_executor_is_noop():
    assert Executor(enabled=False).try_solve("Solve for x: 7x - 3 = 4x + 18").solved is False
