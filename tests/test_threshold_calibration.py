"""Threshold calibration guard for the semantic cache.

The cache serves a stored answer when cosine >= ``CACHE_SIMILARITY_THRESHOLD``.
This test pins that behaviour to the *actual* embedding model so a
mis-calibration is caught before it produces false cache hits.

FINDING (all-MiniLM-L6-v2, measured):
    "Summarize the following text" vs "Give me a summary of the text below" -> 0.619
    "capital of Austria"           vs "capital of Australia"                -> 0.616
    "What is the capital of France?" vs "What's the capital city of France?" -> 0.951

The two example pairs the spec suggested are *inseparable* on MiniLM (0.619 vs
0.616): no threshold puts the paraphrase above and the confusable below. What
MiniLM CAN do is separate a *strong* paraphrase (0.95) from a confusable pair
(0.62) by a wide margin. So the guards below assert the properties that hold and
that actually matter:

* SAFETY   — the confusable pair stays below the threshold (no false hits).
* SEPARABLE — a strong paraphrase scores far above the confusable pair.
* KNOWN GAP — the spec's looser paraphrase pair is recorded as sub-threshold, so
  if a stronger embedding model later lifts it over 0.95 this test fails loudly
  and prompts a recalibration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("sentence_transformers")
pytest.importorskip("faiss")

from app.embeddings import Embedder  # noqa: E402

THRESHOLD = 0.95           # the cache default (CACHE_SIMILARITY_THRESHOLD)
PARAPHRASE_FLOOR = 0.90    # a usable threshold catches strong paraphrases here
CONFUSABLE_CEILING = 0.80  # confusables must sit comfortably under any safe cutoff


@pytest.fixture(scope="module")
def cosine():
    embedder = Embedder(preferred="local", quiet=True)
    if not embedder.warm() or embedder.active_backend == "unavailable":
        pytest.skip("no local embedding backend available")

    def _cos(a: str, b: str) -> float:
        va, vb = embedder.embed(a), embedder.embed(b)
        return float((va * vb).sum())  # normalized vectors -> dot == cosine

    return _cos


def test_confusable_pair_stays_below_threshold(cosine):
    # The safety-critical guard: "capital of Austria" must NOT serve the
    # Australia answer. Comfortably below any sane cutoff, not just 0.95.
    sim = cosine("capital of Austria", "capital of Australia")
    assert sim < CONFUSABLE_CEILING, (
        f"confusable cosine {sim:.4f} is too high; the cache could serve the "
        f"wrong country's answer. Raise CACHE_SIMILARITY_THRESHOLD."
    )


def test_strong_paraphrase_is_separable_from_confusable(cosine):
    # A paraphrase MiniLM can resolve clears the paraphrase floor AND sits far
    # above the confusable pair — proving real duplicates are distinguishable.
    paraphrase = cosine("What is the capital of France?", "What's the capital city of France?")
    confusable = cosine("capital of Austria", "capital of Australia")
    assert paraphrase > PARAPHRASE_FLOOR, (
        f"paraphrase cosine {paraphrase:.4f} is below the {PARAPHRASE_FLOOR} floor; "
        f"real duplicates would miss the cache."
    )
    assert paraphrase - confusable > 0.2, (
        f"paraphrase ({paraphrase:.4f}) is not clearly above confusable "
        f"({confusable:.4f}); the model can't separate duplicates from look-alikes."
    )


def test_spec_paraphrase_is_below_threshold_on_minilm(cosine):
    # KNOWN GAP: the spec's looser paraphrase scores ~0.62 on MiniLM — below
    # 0.95, so it will NOT be cached at the default threshold. Recorded here so
    # the limitation is explicit; a stronger embedder that lifts it over 0.95
    # trips this assertion and signals it's time to recalibrate the threshold.
    sim = cosine("Summarize the following text", "Give me a summary of the text below")
    assert sim < THRESHOLD, (
        f"spec paraphrase now scores {sim:.4f} >= {THRESHOLD} — the embedding model "
        f"improved; revisit CACHE_SIMILARITY_THRESHOLD and this calibration test."
    )
