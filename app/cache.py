"""Query caches — duplicate (and near-duplicate) work costs zero scored tokens.

Two backends behind one ``get(query) / put(query, value)`` interface:

* :class:`QueryCache` — exact matching via SHA-256 of the normalized query.
  Zero dependencies; the safe fallback.
* :class:`SemanticCache` — semantic matching via ``sentence-transformers``
  (all-MiniLM-L6-v2) embeddings in an in-memory ``faiss-cpu`` index. A cached
  answer is served only when cosine similarity clears
  ``CACHE_SIMILARITY_THRESHOLD`` (default **0.95** — deliberately conservative:
  a loose threshold trades answer accuracy for token savings, and serving the
  Paris answer to "capital of Germany?" is far worse than paying for one more
  call). Near-duplicate-but-*different* questions must NOT collide.

The embedding model runs locally on CPU, so cache lookups are **free in
scoring** — no remote tokens are ever spent deciding a hit. A semantic hit on
a remote-answered query therefore turns a scored call into zero remote tokens.

Both backends hold their working set in memory; set ``CACHE_PERSIST_PATH`` to
also write-through entries to a small sqlite file (:class:`SqliteStore`) so
savings compound across restarts. Use :func:`create_cache` to build the
backend selected by config, degrading gracefully to exact matching when the
embedding stack isn't installed.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, List, Optional, Tuple, TypeVar

_WS = re.compile(r"\s+")

V = TypeVar("V")

_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# The embedding model itself now lives in app/embeddings.py (one shared
# process-wide instance). FastAPI runs sync endpoints in a threadpool, so
# concurrent requests are real: all cache mutation here stays lock-guarded.


def _normalize(query: str) -> str:
    return _WS.sub(" ", (query or "").strip().lower())


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    semantic_hits: int = 0  # subset of hits found by embedding similarity

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return round(self.hits / self.total, 3) if self.total else 0.0


# --------------------------------------------------------------------------- #
# Exact backend
# --------------------------------------------------------------------------- #
class QueryCache(Generic[V]):
    """Small thread-safe (lock-guarded) LRU cache keyed by a normalized-query hash."""

    def __init__(self, max_entries: int = 512):
        self.max_entries = max(1, max_entries)
        self._store: "OrderedDict[str, V]" = OrderedDict()
        self.stats = CacheStats()
        self._lock = threading.Lock()
        self.persist: Optional["SqliteStore"] = None  # write-through store, attached by create_cache

    @staticmethod
    def key(query: str) -> str:
        """Stable cache key: SHA-256 of the normalized query."""
        return hashlib.sha256(_normalize(query).encode("utf-8")).hexdigest()

    def warm(self) -> bool:
        """Preload heavy resources. Nothing to warm for the exact backend."""
        return False

    def get(self, query: str) -> Optional[V]:
        return self.get_with_info(query)[0]

    def get_with_info(self, query: str) -> Tuple[Optional[V], str]:
        """Like :meth:`get`, but also reports the hit type for metrics:
        ``"exact"``, ``"semantic"`` (never for this backend), or ``"none"``."""

        k = self.key(query)
        with self._lock:
            if k in self._store:
                self._store.move_to_end(k)  # mark most-recently used
                self.stats.hits += 1
                return self._store[k], "exact"
            self.stats.misses += 1
            return None, "none"

    def put(self, query: str, value: V) -> None:
        k = self.key(query)
        with self._lock:
            self._store[k] = value
            self._store.move_to_end(k)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)  # evict least-recently used
        if self.persist is not None:
            self.persist.save(k, _normalize(query), value)

    def __len__(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self.stats = CacheStats()


# --------------------------------------------------------------------------- #
# Semantic backend
# --------------------------------------------------------------------------- #
class SemanticCache(Generic[V]):
    """LRU cache with exact fast-path plus faiss cosine-similarity lookup.

    Lookup order: exact hash first (free, no embedding), then top-1 nearest
    neighbour in the faiss index; a semantic hit requires cosine similarity
    ``>= similarity_threshold``. Everything runs locally — zero scored tokens.

    Embedding is delegated to the shared :class:`app.embeddings.Embedder`
    (local MiniLM, or the Fireworks fallback) so the cache and the ``--validate``
    scorer share one code path. faiss/numpy are still imported lazily here; if
    the embedding stack is missing this degrades to exact-only matching so the
    service and tests keep working without it.
    """

    def __init__(
        self,
        max_entries: int = 512,
        similarity_threshold: float = 0.95,
        model_name: str = _EMBED_MODEL,
        embedder=None,
    ):
        self.max_entries = max(1, max_entries)
        self.similarity_threshold = similarity_threshold
        self.model_name = model_name
        self._embedder = embedder  # shared Embedder; resolved lazily if None
        self.stats = CacheStats()
        self.persist: Optional["SqliteStore"] = None  # write-through store, attached by create_cache

        # hash-key -> (normalized query, embedding | None, value), LRU-ordered.
        self._entries: "OrderedDict[str, Tuple[str, object, V]]" = OrderedDict()
        self._index = None  # faiss index, rebuilt on eviction
        self._row_keys: List[str] = []  # faiss row -> hash-key mapping
        self._backend = None  # None=untried, True=available, False=degraded
        # Guards entries + index + row_keys together: a torn update between
        # the index and _row_keys would return the WRONG cached answer.
        self._lock = threading.Lock()

    # -- embedding stack (lazy) ------------------------------------------- #
    def _ensure_backend(self) -> bool:
        if self._backend is not None:
            return self._backend
        try:
            import faiss  # noqa: WPS433
            import numpy as np  # noqa: WPS433
        except ImportError:
            self._backend = False  # no vector index -> exact-only degradation
            return self._backend

        if self._embedder is None:
            from app.embeddings import get_embedder  # noqa: WPS433

            self._embedder = get_embedder()
        # Loads the model once (MiniLM, or the remote fallback). If neither
        # backend is usable, degrade to exact-only.
        if not self._embedder.warm() or self._embedder.dim <= 0:
            self._backend = False
            return self._backend

        self._faiss, self._np = faiss, np
        self._index = faiss.IndexFlatIP(self._embedder.dim)
        self._backend = True
        return self._backend

    def _embed(self, text: str):
        # Embedder returns L2-normalized vectors => inner product == cosine.
        vec = self._embedder.embed(_normalize(text))
        return self._np.asarray(vec, dtype="float32").reshape(1, -1)

    def _rebuild_index(self) -> None:
        self._index = self._faiss.IndexFlatIP(self._index.d)
        self._row_keys = []
        for k, (_, emb, _) in self._entries.items():
            if emb is not None:
                self._index.add(emb)
                self._row_keys.append(k)

    # -- public interface --------------------------------------------------- #
    @staticmethod
    def key(query: str) -> str:
        return QueryCache.key(query)

    def warm(self) -> bool:
        """Preload the embedding stack (MiniLM + faiss) so request #1 is fast.

        Returns True when the semantic backend is available, False when it
        degraded to exact-only matching. Safe to call from any thread — the
        lock is held so a concurrent get/put can't observe a half-initialised
        index (``_index`` set before ``_faiss``/``_np``), which would otherwise
        crash or return the wrong cached answer.
        """
        with self._lock:
            return self._ensure_backend()

    def get(self, query: str) -> Optional[V]:
        return self.get_with_info(query)[0]

    def get_with_info(self, query: str) -> Tuple[Optional[V], str]:
        """Like :meth:`get`, but also reports the hit type for metrics:
        ``"exact"``, ``"semantic"``, or ``"none"``."""

        k = self.key(query)
        with self._lock:
            # 1) Exact fast-path: no embedding needed.
            if k in self._entries:
                self._entries.move_to_end(k)
                self.stats.hits += 1
                return self._entries[k][2], "exact"

            # 2) Semantic path: nearest neighbour above the cosine threshold.
            if self._ensure_backend() and self._index is not None and self._index.ntotal:
                scores, rows = self._index.search(self._embed(query), 1)
                score, row = float(scores[0][0]), int(rows[0][0])
                if row >= 0 and score >= self.similarity_threshold:
                    hit_key = self._row_keys[row]
                    self._entries.move_to_end(hit_key)
                    self.stats.hits += 1
                    self.stats.semantic_hits += 1
                    return self._entries[hit_key][2], "semantic"

            self.stats.misses += 1
            return None, "none"

    def put(self, query: str, value: V) -> None:
        k = self.key(query)
        with self._lock:
            if k in self._entries:  # update in place, keep the existing embedding
                norm, emb, _ = self._entries[k]
                self._entries[k] = (norm, emb, value)
                self._entries.move_to_end(k)
            else:
                emb = self._embed(query) if self._ensure_backend() else None
                self._entries[k] = (_normalize(query), emb, value)
                if emb is not None:
                    self._index.add(emb)
                    self._row_keys.append(k)

                evicted = False
                while len(self._entries) > self.max_entries:
                    self._entries.popitem(last=False)  # evict least-recently used
                    evicted = True
                if evicted and self._backend:
                    self._rebuild_index()
        if self.persist is not None:
            self.persist.save(k, _normalize(query), value)

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._row_keys = []
            if self._backend:
                self._rebuild_index()
            self.stats = CacheStats()


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
class SqliteStore:
    """Write-through sqlite store behind the cache ``get/put`` interface.

    Rows are (key, normalized query, JSON value, created-at). On startup
    :func:`create_cache` replays the newest ``max_entries`` rows through
    ``cache.put()`` — the semantic backend re-embeds them then, so the faiss
    index never needs to be serialised. Every write is best-effort: a broken
    disk must degrade to in-memory-only, never fail a request.
    """

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()  # sqlite conns aren't thread-safe by default
        self._conn = sqlite3.connect(path, check_same_thread=False)
        with self._lock, self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS cache_entries ("
                "  key TEXT PRIMARY KEY,"
                "  query TEXT NOT NULL,"
                "  value TEXT NOT NULL,"
                "  created REAL NOT NULL"
                ")"
            )

    def save(self, key: str, norm_query: str, value) -> None:
        try:
            payload = json.dumps(value, ensure_ascii=False, default=str)
            with self._lock, self._conn:
                self._conn.execute(
                    "INSERT OR REPLACE INTO cache_entries (key, query, value, created) "
                    "VALUES (?, ?, ?, ?)",
                    (key, norm_query, payload, time.time()),
                )
        except Exception:  # noqa: BLE001 — persistence is best-effort
            pass

    def load(self, limit: int) -> List[Tuple[str, object]]:
        """Newest ``limit`` rows as (query, value), oldest first.

        Oldest-first so replaying through ``put()`` leaves the newest entries
        most-recently-used in the LRU. Also prunes anything beyond ``limit``
        so the file can't grow without bound across restarts.
        """

        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT query, value FROM cache_entries ORDER BY created DESC LIMIT ?",
                    (max(1, limit),),
                ).fetchall()
                with self._conn:
                    self._conn.execute(
                        "DELETE FROM cache_entries WHERE key NOT IN ("
                        "  SELECT key FROM cache_entries ORDER BY created DESC LIMIT ?"
                        ")",
                        (max(1, limit),),
                    )
            return [(query, json.loads(value)) for query, value in reversed(rows)]
        except Exception:  # noqa: BLE001 — a corrupt file means an empty cache, not a crash
            return []


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def create_cache(config) -> "QueryCache | SemanticCache":
    """Build the cache backend selected by ``CACHE_BACKEND``.

    ``semantic`` (default) returns a :class:`SemanticCache`, which itself
    degrades to exact-only matching if the embedding stack isn't installed —
    so this never fails, it only gets less clever.

    When ``CACHE_PERSIST_PATH`` is set, previously saved entries are replayed
    into the fresh cache and the store is attached for write-through — the
    attach happens *after* the replay so loading doesn't re-write every row.
    """

    if getattr(config, "cache_backend", "semantic") == "semantic":
        from app.embeddings import get_embedder  # noqa: WPS433

        cache: "QueryCache | SemanticCache" = SemanticCache(
            max_entries=config.cache_max_entries,
            similarity_threshold=config.cache_similarity_threshold,
            embedder=get_embedder(config),
        )
    else:
        cache = QueryCache(config.cache_max_entries)

    persist_path = getattr(config, "cache_persist_path", "")
    if persist_path:
        try:
            store = SqliteStore(persist_path)
        except Exception:  # noqa: BLE001 — unwritable path -> in-memory only
            return cache
        for query, value in store.load(config.cache_max_entries):
            cache.put(query, value)
        cache.persist = store

    return cache
