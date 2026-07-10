"""Deterministic executor path — solve or verify at ZERO remote tokens.

For tasks a small model would only *guess* at, we compute the answer directly
instead of trusting a generation:

* **math** — parse the equation/arithmetic with ``sympy`` and return the solved,
  symbolically-checked result. Fully deterministic, no model, no tokens.
* **code** — ask a supplied code generator (the free local model) for a function,
  then *actually execute* it against checks parsed from the prompt in an isolated
  subprocess. We return the code only if it runs and passes; otherwise we decline
  so the router escalates rather than returning unverified code.

The executor is wired into the router BEFORE the classifier, so math/code tasks
never leak to a guessing local answer or an unnecessary remote call. Every entry
point is guarded: a parse/exec failure returns ``ExecResult(solved=False)`` so
the request degrades to normal routing, never crashes.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple


@dataclass
class ExecResult:
    """Outcome of an executor attempt."""

    solved: bool
    answer: str = ""
    kind: str = ""       # "math" | "code"
    method: str = ""     # how it was solved, or why it declined (for the trace)
    local_tokens: int = 0  # tokens a code generator spent (free / scored zero)


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
# A math task shows an equation, an explicit "solve/compute/evaluate", a percent
# expression, or a bare arithmetic expression (digits + operators). Word facts
# ("how many days in a leap year") have no operator and are NOT grabbed.
_MATH_VERB = re.compile(r"\b(solve|evaluate|compute|calculate|simplify|what\s+is|how\s+much\s+is)\b", re.I)
_HAS_OPERATOR = re.compile(r"[-+*/^=]|%|\b(?:sqrt|sin|cos|tan|log|ln|factorial)\b", re.I)
_HAS_DIGIT = re.compile(r"\d")
_EQ_SPLIT = re.compile(r"(?<![<>=!])=(?![=])")  # a single '=' (not ==, <=, >=, !=)

_CODE_TASK = re.compile(
    r"\b(write|implement|create|define|complete)\b[^.]*\b(function|method|def|program|code|routine)\b",
    re.I,
)
_DEF_HINT = re.compile(r"\bdef\s+\w+\s*\(|\bfunction\s+\w+\s*\(", re.I)


class Executor:
    """Math solver + verified-code runner. Injected into :class:`Agent`."""

    def __init__(self, enabled: bool = True, code_timeout: float = 5.0):
        self.enabled = enabled
        self.code_timeout = code_timeout

    # -- public entry ------------------------------------------------------- #
    def try_solve(
        self,
        query: str,
        code_generate: Optional[Callable[[str], object]] = None,
    ) -> ExecResult:
        """Attempt math, then code. ``code_generate`` is a prompt->result callable
        (the local model); if absent, code tasks decline (router escalates)."""

        if not self.enabled or not query:
            return ExecResult(False)
        try:
            m = self._solve_math(query)
            if m.solved:
                return m
        except Exception:  # noqa: BLE001 — a parser hiccup must never crash routing
            pass
        try:
            c = self._solve_code(query, code_generate)
            if c.solved:
                return c
            if c.kind == "code":  # detected as code but couldn't verify -> report why
                return c
        except Exception:  # noqa: BLE001
            pass
        return ExecResult(False)

    # -- math --------------------------------------------------------------- #
    def is_math(self, query: str) -> bool:
        if not (_HAS_DIGIT.search(query) and _HAS_OPERATOR.search(query)):
            return False
        return bool(_MATH_VERB.search(query) or "=" in query or "%" in query or
                    re.search(r"\d\s*[-+*/^]\s*\d", query))

    def _solve_math(self, query: str) -> ExecResult:
        if not self.is_math(query):
            return ExecResult(False)

        from sympy import Eq, solve, Symbol
        from sympy.parsing.sympy_parser import (
            parse_expr, standard_transformations,
            implicit_multiplication_application, convert_xor,
        )
        transforms = standard_transformations + (
            implicit_multiplication_application, convert_xor,
        )

        expr_text = self._extract_math_text(query)
        if not expr_text:
            return ExecResult(False)
        expr_text = self._normalize_percent(expr_text)

        # Equation -> solve for its variable.
        parts = _EQ_SPLIT.split(expr_text)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            lhs = parse_expr(parts[0], transformations=transforms)
            rhs = parse_expr(parts[1], transformations=transforms)
            eq = Eq(lhs, rhs)
            syms = sorted(eq.free_symbols, key=lambda s: s.name)
            if not syms:
                # No variable: it's an assertion like "2+2 = 4" — evaluate LHS.
                val = lhs.evalf() if lhs.free_symbols == set() else lhs
                return ExecResult(True, str(_pretty(lhs)), "math", "evaluated (no variable)")
            var = syms[0]
            sols = solve(eq, var, dict=False)
            if not sols:
                return ExecResult(False)
            shown = ", ".join(f"{var} = {_pretty(s)}" for s in sols)
            return ExecResult(True, shown, "math", f"sympy solve for {var}")

        # Pure arithmetic expression -> evaluate.
        val = parse_expr(expr_text, transformations=transforms)
        if val.free_symbols:
            # Still has variables but no '=' — a simplify request.
            return ExecResult(True, str(_pretty(val)), "math", "sympy simplify")
        return ExecResult(True, str(_pretty(val)), "math", "sympy evaluate")

    @staticmethod
    def _extract_math_text(query: str) -> str:
        """Pull the math out of the sentence: prefer text after 'solve...:' or a
        clause containing '=', else strip a leading 'what is/compute' verb."""
        q = query.strip().rstrip("?.! ")
        if ":" in q:
            after = q.split(":", 1)[1].strip()
            if _HAS_OPERATOR.search(after) or _HAS_DIGIT.search(after):
                q = after
        # When the prompt has commas, keep the clause that holds the equation
        # ("If 3n + 7 = 25, what is n" -> "If 3n + 7 = 25").
        if "," in q:
            eq_clause = next((c for c in q.split(",") if "=" in c), None)
            if eq_clause:
                q = eq_clause
        # drop a leading connective / imperative verb phrase
        q = re.sub(
            r"^(please\s+)?(if|given|suppose|let|solve|evaluate|compute|calculate|simplify|find)\b[:,]?\s*",
            "", q, flags=re.I,
        )
        q = re.sub(r"^(what\s+is|how\s+much\s+is)\s+", "", q, flags=re.I)
        q = re.sub(r"\bfor\s+[a-zA-Z]\b", "", q, flags=re.I)  # "solve for x" leftover
        q = re.sub(r"=\s*\?*\s*$", "", q)  # "X = ?" or trailing bare "=" -> evaluate X
        return q.strip()

    @staticmethod
    def _normalize_percent(text: str) -> str:
        # "15% of 200" -> "(15/100)*(200)" ; standalone "15%" -> "(15/100)"
        text = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*of\s*", r"(\1/100)*", text, flags=re.I)
        text = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", text)
        return text

    # -- code --------------------------------------------------------------- #
    def is_code(self, query: str) -> bool:
        return bool(_CODE_TASK.search(query) or _DEF_HINT.search(query))

    def _solve_code(self, query, code_generate) -> ExecResult:
        if not self.is_code(query):
            return ExecResult(False)

        func = self._target_function(query)
        checks = self._parse_checks(query, func)
        if not checks:
            # Detected code but no verifiable checks in the prompt — decline so
            # the router escalates instead of returning unverified code.
            return ExecResult(False, "", "code", "code task, no inline checks to verify -> escalate")
        if code_generate is None:
            return ExecResult(False, "", "code", "code task, no local generator available -> escalate")

        gen = code_generate(query)
        code = _extract_code(getattr(gen, "text", gen))
        tokens = int(getattr(gen, "tokens", 0) or 0)
        if not code:
            return ExecResult(False, "", "code", "generator produced no code -> escalate", tokens)

        ok, detail = self._run_checks(code, func, checks)
        if ok:
            return ExecResult(True, code, "code", f"verified: {len(checks)} checks passed", tokens)
        return ExecResult(False, "", "code", f"generated code failed checks ({detail}) -> escalate", tokens)

    @staticmethod
    def _target_function(query: str) -> Optional[str]:
        m = re.search(r"\b(?:function|def)\s+([a-zA-Z_]\w*)\s*\(", query)
        if m:
            return m.group(1)
        m = re.search(r"\b([a-zA-Z_]\w*)\s*\([^)]*\)\s*(?:->|==|returns?)", query)
        return m.group(1) if m else None

    @staticmethod
    def _parse_checks(query: str, func: Optional[str]) -> List[Tuple[str, str]]:
        """Parse inline examples: ``func(args) == val`` / ``-> val`` / ``returns val``.
        Returns a list of (call_expr, expected_literal_repr)."""
        if not func:
            return []
        checks: List[Tuple[str, str]] = []
        pat = re.compile(
            re.escape(func) + r"\s*\(([^()]*)\)\s*(?:==|->|returns?)\s*"
            r"(True|False|None|-?\d+(?:\.\d+)?|'[^']*'|\"[^\"]*\"|\[[^\]]*\])",
            re.I,
        )
        for args, expected in pat.findall(query):
            checks.append((f"{func}({args})", expected))
        return checks

    def _run_checks(self, code: str, func: str, checks: List[Tuple[str, str]]) -> Tuple[bool, str]:
        """Execute the generated code + checks in an isolated subprocess.

        NOTE: this runs model-generated code. It is sandboxed only by process
        isolation (``python -I``) and a wall-clock timeout — adequate for a local
        eval on trusted prompts, not a hostile-input server. No network/file
        restrictions are applied.
        """
        # Insert CALL and EXPECTED as *code*, not string literals, so quotes in
        # the call (e.g. is_palindrome('racecar')) can't break the harness.
        lines = [code, "", "_fails = []"]
        for call, expected in checks:
            lines += [
                "try:",
                f"    _r = {call}",
                f"    _ok = (_r == ({expected}))",
                f"    _fails.append(None) if _ok else _fails.append(repr(_r))",
                "except Exception as _e:",
                "    _fails.append('exc:' + repr(_e))",
            ]
        lines.append("_bad = [f for f in _fails if f is not None]")
        lines.append("print('OK' if not _bad else 'FAIL:' + ' | '.join(_bad))")
        harness = "\n".join(lines) + "\n"

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "candidate.py"
            p.write_text(harness, encoding="utf-8")
            try:
                out = subprocess.run(
                    [sys.executable, "-I", str(p)],
                    capture_output=True, text=True, timeout=self.code_timeout,
                    cwd=td,
                )
            except subprocess.TimeoutExpired:
                return False, "timeout"
        stdout = (out.stdout or "").strip()
        if stdout == "OK":
            return True, "all checks passed"
        if out.returncode != 0:
            return False, (out.stderr or "").strip().splitlines()[-1:] and (out.stderr or "").strip().splitlines()[-1] or "error"
        return False, stdout


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _pretty(sym) -> str:
    """Render a sympy result as a clean int when it is one."""
    try:
        from sympy import Integer, Rational, Float
        if sym == int(sym):
            return str(int(sym))
    except (TypeError, ValueError):
        pass
    return str(sym)


def _extract_code(text) -> str:
    """Pull a python code block out of a model reply (fenced or bare def)."""
    if not isinstance(text, str):
        return ""
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, re.S | re.I)
    if fence:
        return fence.group(1).strip()
    # else: from the first 'def '/'import' to the end
    m = re.search(r"(^|\n)(import |from |def |class )", text)
    return text[m.start():].strip() if m else ""
