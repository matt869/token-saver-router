"""Regex IDS — an intrusion-detection-style inspector for prompts.

A network IDS sniffs packets for attack signatures; this one sniffs *prompts*
for **heavy-compute signatures**: the tell-tale marks of a query that a small
local model will botch and that should go straight to the strong remote model
(code fences, SQL, proofs, big-O analysis, dense math). It is deliberately built
from precompiled regexes with word boundaries — far more precise than substring
keyword matching, and effectively free (no tokens, no model).

It complements the :class:`~app.router.classifier.HeuristicClassifier`: the
classifier scores general difficulty, while the IDS raises a hard flag on the
specific patterns that most need the remote model. When it fires above
``threshold`` the agent forces the remote route regardless of the classifier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# (rule name, pattern, weight). Weights accumulate into a 0..1 severity.
_RULES: List[Tuple[str, str, float]] = [
    ("fenced_code", r"```", 0.6),
    ("code_def", r"\b(?:def|class|function|public|private|void|struct)\s+\w", 0.4),
    ("import_stmt", r"^\s*(?:import|from|#include|using)\s+\w", 0.3),
    ("sql", r"\bselect\b[\s\S]+?\bfrom\b", 0.4),
    ("shell", r"(?:^|\s)(?:sudo|grep|awk|sed|curl|docker|kubectl)\s", 0.3),
    ("math_symbol", r"[∫∑∂√π≤≥≠≈∇∞⊕⊗]", 0.4),
    ("big_o", r"\bO\(\s*n(?:\s*\^|\s+log|\s*\))", 0.4),
    ("proof", r"\b(?:prove|proof|derive|derivation|theorem|lemma|q\.?e\.?d)\b", 0.4),
    ("algorithmic", r"\b(?:np-?(?:hard|complete)|dynamic\s+programming|"
                    r"time\s+complexity|asymptotic|recurrence)\b", 0.4),
    ("multi_step", r"\bstep[-\s]by[-\s]step\b", 0.3),
    ("regex_task", r"\b(?:regex|regular\s+expression)\b", 0.3),
    ("equation", r"[A-Za-z0-9)]\s*=\s*[^=]", 0.2),
]

_COMPILED: List[Tuple[str, "re.Pattern[str]", float]] = [
    (name, re.compile(pat, re.IGNORECASE | re.MULTILINE), weight)
    for name, pat, weight in _RULES
]


@dataclass
class IDSVerdict:
    """Result of inspecting one prompt."""

    flagged: bool
    severity: float  # 0.0 (clean) .. 1.0 (definitely heavy compute)
    matches: List[str] = field(default_factory=list)  # rule names that fired

    def as_dict(self) -> Dict[str, object]:
        return {"flagged": self.flagged, "severity": self.severity, "matches": self.matches}


class RegexIDS:
    """Precompiled regex inspector for heavy-compute prompt signatures."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def inspect(self, query: str) -> IDSVerdict:
        text = query or ""
        matches: List[str] = []
        severity = 0.0
        for name, pattern, weight in _COMPILED:
            if pattern.search(text):
                matches.append(name)
                severity += weight

        severity = round(min(1.0, severity), 3)
        return IDSVerdict(
            flagged=severity >= self.threshold,
            severity=severity,
            matches=matches,
        )
