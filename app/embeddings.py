"""Shared embedding backend.

One embedding code path, used by **both** the semantic cache (`app/cache.py`)
and the ``--validate`` cosine scoring (`app/eval/run_eval.py`), so calibration
is consistent everywhere. Two interchangeable backends behind one
:class:`Embedder` facade:

* **local** (default) — ``all-MiniLM-L6-v2`` via ``sentence-transformers``,
  loaded **once** in :meth:`Embedder.warm` (never per request).
* **remote** — the Fireworks embeddings API, used when
  ``sentence-transformers`` isn't installed or the local weights can't load
  (e.g. no HuggingFace access on the demo box).

``EMBED_BACKEND=local|remote`` picks the *preferred* backend; whichever is
chosen, the other is tried as a fallback so a lookup never hard-fails. The
resolved backend is logged once at warm-up so the demo machine's mode is
obvious. All vectors are L2-normalized, so an inner product equals cosine
similarity (what faiss ``IndexFlatIP`` and the validator both expect).
"""

from __future__ import annotations

import threading
from typing import List, Optional

LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REMOTE_MODEL = "nomic-ai/nomic-embed-text-v1.5"


def _l2_normalize(mat):
    import numpy as np  # noqa: WPS433

    arr = np.asarray(mat, dtype="float32")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid divide-by-zero on a zero vector
    return arr / norms


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class LocalEmbedder:
    """MiniLM via sentence-transformers, materialised once."""

    kind = "local"

    def __init__(self, model_name: str = LOCAL_MODEL):
        self.model_name = model_name
        self._model = None
        self._dim = 0

    def warm(self) -> bool:
        if self._model is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer  # noqa: WPS433

            self._model = SentenceTransformer(self.model_name, device="cpu")
            # sentence-transformers >=5.6 renamed get_sentence_embedding_dimension.
            get_dim = getattr(self._model, "get_embedding_dimension", None) or (
                self._model.get_sentence_embedding_dimension
            )
            self._dim = int(get_dim())
            return True
        except Exception:  # noqa: BLE001 — missing dep or no HF access -> caller falls back
            self._model = None
            return False

    def embed_batch(self, texts: List[str]):
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return _l2_normalize(vecs)  # already unit-norm; guards odd dtypes

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def label(self) -> str:
        return f"local:{self.model_name.split('/')[-1]}"


class RemoteEmbedder:
    """Fireworks embeddings API (OpenAI-compatible ``/embeddings``)."""

    kind = "remote"

    def __init__(
        self,
        model: str = REMOTE_MODEL,
        api_key: str = "",
        base_url: str = "https://api.fireworks.ai/inference/v1",
        timeout: int = 30,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._dim = 0

    def warm(self) -> bool:
        if not self.api_key:
            return False
        try:
            vec = self._raw_embed(["warmup"])
            self._dim = len(vec[0])
            return self._dim > 0
        except Exception:  # noqa: BLE001 — no network / bad key -> caller falls back
            return False

    def _raw_embed(self, texts: List[str]) -> List[List[float]]:
        import requests  # noqa: WPS433

        resp = requests.post(
            f"{self.base_url}/embeddings",
            json={"model": self.model, "input": texts},
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        # Preserve request order regardless of how the API returns them.
        ordered = sorted(data, key=lambda d: d.get("index", 0))
        return [d["embedding"] for d in ordered]

    def embed_batch(self, texts: List[str]):
        return _l2_normalize(self._raw_embed(texts))

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def label(self) -> str:
        return f"remote:{self.model.split('/')[-1]}"


# --------------------------------------------------------------------------- #
# Facade
# --------------------------------------------------------------------------- #
class Embedder:
    """Preferred backend with automatic fallback to the other one.

    ``warm()`` resolves the active backend exactly once (thread-safe). After a
    successful warm, :meth:`embed` / :meth:`embed_batch` return normalized
    ``float32`` arrays; before it (or when both backends are unavailable) they
    raise :class:`EmbeddingUnavailable` so callers can degrade cleanly.
    """

    def __init__(
        self,
        preferred: str = "local",
        local_model: str = LOCAL_MODEL,
        remote_model: str = REMOTE_MODEL,
        api_key: str = "",
        base_url: str = "https://api.fireworks.ai/inference/v1",
        timeout: int = 30,
        quiet: bool = False,
    ):
        self.preferred = preferred if preferred in ("local", "remote") else "local"
        self._local = LocalEmbedder(local_model)
        self._remote = RemoteEmbedder(remote_model, api_key, base_url, timeout)
        self._quiet = quiet
        self._impl = None  # resolved backend, or _UNAVAILABLE
        self._lock = threading.Lock()

    def warm(self) -> bool:
        with self._lock:
            if self._impl is not None:
                return self._impl is not _UNAVAILABLE

            order = (self._local, self._remote)
            if self.preferred == "remote":
                order = (self._remote, self._local)

            for backend in order:
                if backend.warm():
                    self._impl = backend
                    self._log(f"[embeddings] active backend: {backend.label} (dim={backend.dim})")
                    return True

            self._impl = _UNAVAILABLE
            self._log("[embeddings] no backend available -> semantic cache degrades to exact-only")
            return False

    def embed_batch(self, texts: List[str]):
        if self._impl is None:
            self.warm()
        if self._impl is _UNAVAILABLE:
            raise EmbeddingUnavailable("no embedding backend available")
        return self._impl.embed_batch(list(texts))

    def embed(self, text: str):
        return self.embed_batch([text])[0]

    @property
    def dim(self) -> int:
        return 0 if self._impl in (None, _UNAVAILABLE) else self._impl.dim

    @property
    def active_backend(self) -> str:
        if self._impl is None:
            return "unwarmed"
        if self._impl is _UNAVAILABLE:
            return "unavailable"
        return self._impl.label

    def _log(self, msg: str) -> None:
        if not self._quiet:
            print(msg)


class EmbeddingUnavailable(RuntimeError):
    """Raised when neither embedding backend can produce a vector."""


_UNAVAILABLE = object()  # sentinel: warm() ran, nothing usable


# --------------------------------------------------------------------------- #
# Process-wide singleton
# --------------------------------------------------------------------------- #
_EMBEDDER: Optional[Embedder] = None
_EMBEDDER_LOCK = threading.Lock()


def get_embedder(config=None) -> Embedder:
    """The shared process-wide :class:`Embedder`, built from config once.

    Both the semantic cache and the ``--validate`` scorer call this, so they
    embed through the exact same weights and backend selection.
    """

    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    with _EMBEDDER_LOCK:
        if _EMBEDDER is None:
            if config is None:
                from app.config import load_config  # noqa: WPS433

                config = load_config()
            _EMBEDDER = Embedder(
                preferred=getattr(config, "embed_backend", "local"),
                local_model=getattr(config, "embed_model", LOCAL_MODEL),
                remote_model=getattr(config, "embed_remote_model", REMOTE_MODEL),
                api_key=getattr(config, "fireworks_api_key", ""),
                base_url=getattr(config, "fireworks_base_url", ""),
                timeout=getattr(config, "remote_timeout", 30),
            )
    return _EMBEDDER


def reset_embedder() -> None:
    """Drop the singleton (tests that swap config/backends)."""

    global _EMBEDDER
    with _EMBEDDER_LOCK:
        _EMBEDDER = None
