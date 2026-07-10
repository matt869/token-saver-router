"""Tests for the shared embedding backend + fallback logic + the proportional
stub remote (no live API needed)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.embeddings import Embedder, EmbeddingUnavailable  # noqa: E402
from app.models.stub_remote import StubRemoteModel  # noqa: E402
from app.tokens import count_tokens  # noqa: E402


class _FakeBackend:
    """Stand-in embedder backend that needs no model download."""

    def __init__(self, kind, ok=True, dim=4):
        self.kind = kind
        self._ok = ok
        self.dim = dim
        self.label = f"{kind}:fake"
        self.warmed = 0

    def warm(self):
        self.warmed += 1
        return self._ok

    def embed_batch(self, texts):
        import numpy as np

        # Deterministic unit vectors keyed on text length.
        rows = [[float(len(t) % 7 + 1), 1.0, 0.0, 0.0] for t in texts]
        arr = np.asarray(rows, dtype="float32")
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def _embedder_with(local_ok, remote_ok, preferred="local"):
    emb = Embedder(preferred=preferred, quiet=True)
    emb._local = _FakeBackend("local", ok=local_ok)
    emb._remote = _FakeBackend("remote", ok=remote_ok)
    return emb


def test_prefers_local_then_falls_back_to_remote():
    emb = _embedder_with(local_ok=False, remote_ok=True, preferred="local")
    assert emb.warm() is True
    assert emb.active_backend == "remote:fake"  # local failed -> remote used


def test_respects_remote_preference():
    emb = _embedder_with(local_ok=True, remote_ok=True, preferred="remote")
    assert emb.warm() is True
    assert emb.active_backend == "remote:fake"


def test_unavailable_when_both_fail_and_raises():
    emb = _embedder_with(local_ok=False, remote_ok=False)
    assert emb.warm() is False
    assert emb.active_backend == "unavailable"
    try:
        emb.embed("x")
        assert False, "expected EmbeddingUnavailable"
    except EmbeddingUnavailable:
        pass


def test_warm_is_idempotent():
    emb = _embedder_with(local_ok=True, remote_ok=True)
    emb.warm()
    emb.warm()
    assert emb._local.warmed == 1  # resolved once, not per call


def test_embed_returns_normalized_vector():
    import numpy as np

    emb = _embedder_with(local_ok=True, remote_ok=True)
    emb.warm()
    v = emb.embed("hello")
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


# --------------------------------------------------------------------------- #
# Stub remote: token count moves in the correct direction
# --------------------------------------------------------------------------- #
def test_stub_remote_prompt_tokens_track_the_compressed_prompt():
    stub = StubRemoteModel()
    short = stub.generate("summarize this")
    long = stub.generate("please kindly summarize this very long report for me thanks")
    # Proportional, not constant: a longer prompt bills more prompt tokens.
    assert long.prompt_tokens > short.prompt_tokens
    assert short.prompt_tokens == count_tokens("summarize this")
    assert short.tokens == short.prompt_tokens + short.completion_tokens


def test_stub_after_never_exceeds_before_when_compression_helps():
    # The bug this guards: a constant stub could report after > before. Here the
    # stub echoes the compressed prompt, so after <= before whenever preflight
    # actually removed tokens.
    from app.optimizer import PreflightOptimizer

    raw = "Hi there! Please kindly summarize    this   report for me. Thank you very much!"
    opt = PreflightOptimizer().optimize(raw)
    before = count_tokens(raw)
    after = StubRemoteModel().generate(opt.text).prompt_tokens
    assert after <= before
