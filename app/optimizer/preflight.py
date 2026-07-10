"""Pre-flight prompt optimizer — token-saving compression before the scored call.

Fireworks (and every remote provider) bills you for *prompt* tokens, not just
the completion. A large share of a typical prompt is signal-free: trailing
whitespace, blank-line padding, and social filler ("please", "thank you so
much", "I was wondering if you could ..."). Stripping it is lossless for the
task and shaves a reliable few percent off every remote call — think of it as
gzip for the wire before the model ever sees the bytes.

Design goals
------------
* **Deterministic & dependency-free** so it costs zero scored tokens and runs
  anywhere (CPU, no ML stack).
* **Safe:** never touch the inside of a fenced ``` code block ```, `inline
  code`, or a "double-quoted string" — code needs its whitespace and a quoted
  phrase is often the task's subject. Only prose segments are compressed.
* **Swappable:** :class:`PreflightOptimizer` is a plain object the agent injects,
  so a smarter compressor can drop in without touching ``agent.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

from app.tokens import count_tokens

# Social / filler phrases that carry no task signal. Order matters: longer,
# more specific phrases are removed before the short words they contain.
# Each entry is (label, compiled case-insensitive pattern, replacement).
# The replacement is usually a space; context-aware rules keep part of the
# match (e.g. "just" is only filler after an auxiliary/pronoun — "could just
# check" — never as an adjective, so "a just war" must survive).
_FILLER_PATTERNS: List[Tuple[str, "re.Pattern[str]", str]] = [
    (label, re.compile(pat, re.IGNORECASE), repl)
    for label, pat, repl in (
        ("courtesy-preamble",
         r"\bi(?:'m| am| was)\s+wondering\s+if\s+you\s+(?:could|would|can|might)\b", " "),
        ("courtesy-preamble",
         r"\bi\s+would\s+(?:really\s+)?appreciate\s+it\s+if\s+you\s+(?:could|would)\b", " "),
        ("courtesy-preamble", r"\bif\s+you\s+(?:could|would|don't\s+mind|do\s+not\s+mind)\b", " "),
        ("courtesy-preamble", r"\bwould\s+you\s+be\s+(?:so\s+)?kind\s+enough\s+to\b", " "),
        ("thanks", r"\bthank\s+you\s+(?:very\s+much|so\s+much|in\s+advance)\b", " "),
        ("thanks", r"\bthanks\s+(?:a\s+lot|so\s+much|in\s+advance|again)\b", " "),
        ("thanks", r"\bthank\s+you\b", " "),
        ("thanks", r"\bthanks\b", " "),
        ("please", r"\bplease\b", " "),
        ("please", r"\bkindly\b", " "),
        ("filler", r"\b((?:could|can|would|will|you|i|we|please)\s+)just\s+(?=\w)", r"\1"),
        ("filler", r"\bif\s+you\s+don't\s+mind\b", " "),
    )
]

# Sentences shorter than this (normalized) are never deduped: dropping a
# repeated "Yes." or "Step 2:" is more likely to break structure than save
# anything meaningful.
_DEDUP_MIN_CHARS = 30

# Sentence boundary for dedup: end punctuation + whitespace, or a newline
# (templated prompts repeat whole lines at least as often as sentences).
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

_WS = re.compile(r"\s+")

# A greeting that opens the prompt ("Hi,", "Hello there —") is pure preamble.
_LEADING_GREETING = re.compile(
    r"^(?:hi|hello|hey|greetings|good\s+(?:morning|afternoon|evening))"
    r"(?:\s+there)?\b[\s,!.:—-]*",
    re.IGNORECASE,
)


@dataclass
class OptimizeResult:
    """Outcome of one pre-flight pass.

    Token counts come from a real tokenizer (see ``app/tokens.py``), measured
    on the same vocabulary before and after so ``tokens_saved`` is honest —
    not a chars/4 guess.
    """

    text: str
    original_chars: int
    optimized_chars: int
    original_tokens: int = 0
    optimized_tokens: int = 0
    removed: List[str] = field(default_factory=list)  # labels of applied rules

    @property
    def saved_chars(self) -> int:
        return max(0, self.original_chars - self.optimized_chars)

    @property
    def saved_pct(self) -> float:
        if self.original_chars == 0:
            return 0.0
        return round(100.0 * self.saved_chars / self.original_chars, 1)

    @property
    def tokens_saved(self) -> int:
        """Real tokenizer delta between the original and optimized prompt."""
        return max(0, self.original_tokens - self.optimized_tokens)


class PreflightOptimizer:
    """Losslessly compresses a prompt before it hits the scored remote path."""

    def __init__(
        self,
        strip_filler: bool = True,
        collapse_whitespace: bool = True,
        dedup_sentences: bool = True,
        token_counter: Callable[[str], int] = count_tokens,
    ):
        self.strip_filler = strip_filler
        self.collapse_whitespace = collapse_whitespace
        self.dedup_sentences = dedup_sentences
        self.token_counter = token_counter

    def optimize(self, prompt: str) -> OptimizeResult:
        original = prompt or ""
        removed: List[str] = []

        # Split on protected spans we must never disturb: fenced ``` blocks,
        # `inline code`, "double-quoted strings", and 'single-quoted strings'
        # (a quoted phrase is often the *subject* of the task — e.g.
        # translate/rewrite requests — so deleting "please" inside it would
        # corrupt the task itself; the lookarounds keep apostrophes in words
        # like "don't" from opening a quote). Odd indices are the protected
        # spans; even indices are compressible prose.
        segments = re.split(
            r'(```.*?```|`[^`\n]+`|"[^"\n]+"|(?<!\w)\'[^\'\n]+\'(?!\w))',
            original,
            flags=re.DOTALL,
        )
        for i, seg in enumerate(segments):
            if i % 2 == 1:  # protected span — leave untouched
                continue
            segments[i] = self._compress_prose(seg, removed)

        # Exact-match sentence dedup across the whole prompt (templated prompts
        # repeat boilerplate instructions — that's where the real savings are).
        # Protected spans never count as sentences and are never dropped.
        if self.dedup_sentences:
            seen: set = set()
            for i, seg in enumerate(segments):
                if i % 2 == 0:
                    segments[i] = self._dedup_sentences(seg, seen, removed)

        text = "".join(segments)

        # Whole-prompt tidy-up (safe outside code, applied to the joined result
        # only at the very ends so interior code is never affected).
        if self.collapse_whitespace:
            text = text.strip()
            # Filler removal can orphan its punctuation ("... Thanks!" -> "!").
            text = re.sub(r"^[,.;:!?\s]+", "", text)

        return OptimizeResult(
            text=text,
            original_chars=len(original),
            optimized_chars=len(text),
            original_tokens=self.token_counter(original),
            optimized_tokens=self.token_counter(text),
            removed=_dedupe_preserving_order(removed),
        )

    # ------------------------------------------------------------------ #
    def _compress_prose(self, seg: str, removed: List[str]) -> str:
        if not seg:
            return seg

        if self.strip_filler:
            stripped = _LEADING_GREETING.sub("", seg, count=1)
            if stripped != seg:
                removed.append("greeting")
                seg = stripped

            for label, pattern, repl in _FILLER_PATTERNS:
                new_seg, n = pattern.subn(repl, seg)
                if n:
                    removed.append(label)
                    seg = new_seg

        if self.collapse_whitespace:
            # Collapse runs of spaces/tabs, tidy space-before-punctuation, and
            # cap consecutive blank lines at one — all lossless for prose.
            seg = re.sub(r"[ \t]+", " ", seg)
            seg = re.sub(r" *\n[ \t]*", "\n", seg)
            seg = re.sub(r"\n{3,}", "\n\n", seg)
            seg = re.sub(r"\s+([,.;:!?])", r"\1", seg)
            # "for me? Thanks!" -> "for me?!" — drop the orphaned mark left
            # behind by filler removal (keeps "..." ellipses intact).
            seg = re.sub(r"([.!?])[!?]+", r"\1", seg)

        return seg

    # ------------------------------------------------------------------ #
    def _dedup_sentences(self, seg: str, seen: set, removed: List[str]) -> str:
        """Drop later exact repeats of sentences already emitted in this prompt.

        ``seen`` is shared across all prose segments of one prompt so a
        boilerplate sentence repeated around a code block is still caught.
        Comparison is on a casefolded, whitespace-collapsed form; sentences
        under ``_DEDUP_MIN_CHARS`` are never touched (dropping a repeated
        "Yes." breaks structure for near-zero savings).
        """

        if not seg:
            return seg

        # Capturing split keeps separators at odd indices so the surviving
        # sentences rejoin with their original spacing.
        parts = _SENTENCE_SPLIT.split(seg)
        seps = _SENTENCE_SPLIT.findall(seg)
        out: List[str] = []
        dropped = False
        for idx, sent in enumerate(parts):
            sep = seps[idx] if idx < len(seps) else ""
            norm = _WS.sub(" ", sent).strip().casefold()
            if len(norm) >= _DEDUP_MIN_CHARS and norm in seen:
                dropped = True  # skip the sentence and its separator
                continue
            if len(norm) >= _DEDUP_MIN_CHARS:
                seen.add(norm)
            out.append(sent + sep)

        if dropped:
            removed.append("dedup-sentence")
        return "".join(out)


def _dedupe_preserving_order(items: List[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out
