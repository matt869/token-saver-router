"""Query difficulty classifiers.

Two swappable implementations behind the same :class:`Classifier` protocol,
selected via the ``CLASSIFIER`` env var (``llm`` | ``heuristic``, default
``llm``) so both can be benchmarked against the same scored-token metric:

* :class:`HeuristicClassifier` — regex/keyword/length rules. Dependency-free,
  deterministic, runs anywhere (no GPU, no key). Kept as the fallback.
* :class:`LLMClassifier` — LLM-as-a-Judge: asks the *local* model (free, zero
  scored tokens) whether it can answer the query well, capped at ~5 output
  tokens so judging stays cheap and fast. Biased toward LOCAL on ambiguity —
  the downstream verify-and-escalate step in ``agent.py`` still catches a bad
  local answer, whereas a needless REMOTE verdict spends scored tokens with no
  safety net.

Anything that implements :class:`Classifier.classify` can replace either —
for example a fine-tuned distilled model — without touching ``agent.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Protocol, runtime_checkable

# Signals that a query is likely to need heavy reasoning / a stronger model.
HARD_KEYWORDS: List[str] = [
    "prove", "derive", "proof", "theorem", "integral", "derivative",
    "differential", "algorithm", "complexity", "optimize", "optimise",
    "analyze", "analyse", "explain why", "step by step", "step-by-step",
    "reasoning", "trade-off", "tradeoff", "architecture", "design a",
    "design an", "implement", "refactor", "debug", "compare and contrast",
    "evaluate", "critique", "multi-step", "distributed", "concurrency",
    "asymptotic", "np-hard", "recursion", "dynamic programming",
    "why does", "how would you", "pros and cons", "edge case",
]

# Signals that a query is a simple lookup / factual / short-form answer.
EASY_KEYWORDS: List[str] = [
    "what is", "what's", "who is", "who was", "where is", "when did",
    "when was", "define", "definition of", "capital of", "translate",
    "spell", "how many", "how do you say", "meaning of", "synonym",
    "antonym", "hello", "hi ", "hey ", "thanks", "thank you",
]

# Substrings that hint at code or math content (bias toward "hard").
CODE_MATH_MARKERS: List[str] = [
    "```", "def ", "class ", "import ", "=>", "->", "::", "\\", "∫",
    "∑", "∂", "√", "lambda ", "select ", "for(", "for (",
]


@dataclass
class Classification:
    """Result of classifying a single query."""

    route: str  # "local" (easy) or "remote" (hard)
    difficulty: float  # normalised 0.0 (trivial) .. 1.0 (very hard)
    signals: Dict[str, object] = field(default_factory=dict)


@runtime_checkable
class Classifier(Protocol):
    """Swappable classifier interface.

    Implementations must return a :class:`Classification`. Keep them cheap and
    side-effect free so routing decisions cost no scored tokens.
    """

    def classify(self, query: str) -> Classification:  # pragma: no cover
        ...


class HeuristicClassifier:
    """Keyword + length heuristic. Fast, deterministic, GPU-free.

    ``difficulty`` combines four signals: hard-keyword hits, easy-keyword hits
    (which pull the score down), query length relative to the configured
    ``length_cutoff``, and code/math markers. A query is routed ``"remote"``
    when its difficulty meets ``decision_threshold``.
    """

    def __init__(self, length_cutoff: int = 24, decision_threshold: float = 0.5):
        self.length_cutoff = max(1, length_cutoff)
        self.decision_threshold = decision_threshold

    def classify(self, query: str) -> Classification:
        text = (query or "").strip()
        lower = text.lower()
        words = lower.split()
        n_words = len(words)

        hard_hits = [k for k in HARD_KEYWORDS if k in lower]
        easy_hits = [k for k in EASY_KEYWORDS if k in lower]
        has_code_math = any(m in text for m in CODE_MATH_MARKERS)
        multi_question = lower.count("?") > 1

        # Length contribution: ramps up to 1.0 at the cutoff, with an extra
        # bump for queries that clearly exceed it.
        length_factor = min(n_words / self.length_cutoff, 1.0)

        score = 0.0
        # 0.30 per hard keyword: a single formal-reasoning verb ("prove",
        # "derive", "design a") on a medium-length query should clear the 0.5
        # threshold on its own — under-routing those wastes a local round-trip.
        score += 0.30 * len(hard_hits)
        score -= 0.15 * len(easy_hits)
        score += 0.40 * length_factor
        if n_words > self.length_cutoff:
            score += 0.30
        if has_code_math:
            score += 0.20
        if multi_question:
            score += 0.15

        difficulty = _clamp(score, 0.0, 1.0)
        route = "remote" if difficulty >= self.decision_threshold else "local"

        signals = {
            "n_words": n_words,
            "hard_hits": hard_hits,
            "easy_hits": easy_hits,
            "has_code_math": has_code_math,
            "multi_question": multi_question,
            "length_factor": round(length_factor, 3),
            "length_cutoff": self.length_cutoff,
            "decision_threshold": self.decision_threshold,
        }
        return Classification(route=route, difficulty=round(difficulty, 3), signals=signals)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# --------------------------------------------------------------------------- #
# LLM-as-a-Judge classifier
# --------------------------------------------------------------------------- #
@dataclass
class Decision:
    """LLM judge verdict, duck-type compatible with :class:`Classification`.

    Exposes ``route``/``score``/``reason`` as requested, plus ``difficulty``
    and ``signals`` properties so ``agent.py`` consumes it unchanged.
    ``tokens`` is what the judge spent on the *local* model (free in scoring,
    but reported so the eval's local/remote split stays honest).
    """

    route: str  # "local" or "remote"
    score: float  # 0.0 (trivial) .. 1.0 (needs the remote model)
    reason: str
    tokens: int = 0  # local tokens the judge consumed (scored as zero)

    @property
    def difficulty(self) -> float:
        return self.score

    @property
    def signals(self) -> Dict[str, object]:
        return {"judge": "llm", "reason": self.reason}


# The judge asks the local model to self-assess. Single-word contract keeps
# parsing trivial and the output cap tight.
_JUDGE_PROMPT = (
    "You are a routing judge. A 7B local language model will try to answer the "
    "user query below. Decide if the 7B model can answer it well on its own.\n"
    "Reply with exactly one word: LOCAL if the 7B model can answer it well, or "
    "REMOTE if it needs a stronger model.\n\n"
    "Query: {query}\n\n"
    "One word (LOCAL or REMOTE):"
)

# Cap the judge's completion: one word is all we asked for. Judge tokens run on
# the local model, so they are free in scoring — but still keep them tiny for
# latency.
_JUDGE_MAX_NEW_TOKENS = 5


class LLMClassifier:
    """Judges difficulty by asking the local model itself (zero scored tokens).

    Shares the agent's :class:`~app.models.local_model.LocalModel` instance, so
    the weights load once and are reused for both judging and answering. If the
    judge is unavailable or errors, falls back to the heuristic classifier —
    routing must never crash the request.

    **Gray-zone judging:** the free heuristic runs first; when it is already
    confident (difficulty <= ``easy_below`` or >= ``hard_above``) the LLM judge
    is skipped entirely. That saves a real model call (seconds) on obvious
    queries and contains small-judge miscalibration — the 1.5B judge, for
    example, over-calls REMOTE even on trivia. The judge only breaks ties in
    the ambiguous middle band, which is exactly where it adds signal.
    """

    def __init__(
        self,
        local_model=None,
        fallback: Classifier | None = None,
        easy_below: float = 0.2,
        hard_above: float = 0.8,
    ):
        self.local_model = local_model
        self.fallback = fallback or HeuristicClassifier()
        self.easy_below = easy_below
        self.hard_above = hard_above

    def classify(self, query: str) -> "Decision | Classification":
        if self.local_model is None:
            return self.fallback.classify(query)

        # Heuristic pre-pass: skip the judge outside the gray zone.
        heuristic = self.fallback.classify(query)
        if heuristic.difficulty <= self.easy_below or heuristic.difficulty >= self.hard_above:
            return Decision(
                route=heuristic.route,
                score=heuristic.difficulty,
                reason=(
                    f"judge skipped: heuristic confident "
                    f"(difficulty={heuristic.difficulty})"
                ),
            )

        try:
            result = self.local_model.generate(
                _JUDGE_PROMPT.format(query=(query or "").strip()),
                max_new_tokens=_JUDGE_MAX_NEW_TOKENS,
            )
            reply = (result.text or "").strip()
            judge_tokens = int(getattr(result, "tokens", 0) or 0)
        except Exception as exc:  # noqa: BLE001 — judge failure must not break routing
            return Decision(
                route=heuristic.route,
                score=heuristic.difficulty,
                reason=f"judge failed ({exc}); heuristic fallback",
            )

        verdict = reply.split()[0].upper().strip(".,:;!") if reply.split() else ""
        if verdict == "REMOTE":
            return Decision(
                route="remote", score=0.9,
                reason=f"judge said {reply!r}", tokens=judge_tokens,
            )

        # Bias toward LOCAL on anything that doesn't cleanly parse as REMOTE:
        # the verifier downstream still escalates a bad local answer, so an
        # over-eager REMOTE here would spend scored tokens for nothing.
        reason = (
            f"judge said {reply!r}"
            if verdict == "LOCAL"
            else f"unparseable judge reply {reply!r}; defaulting LOCAL"
        )
        return Decision(route="local", score=0.2, reason=reason, tokens=judge_tokens)
