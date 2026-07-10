"""Pre-fetch the local embedding weights so first-request latency is off the
demo's critical path.

Downloads (and caches, via HuggingFace's on-disk cache) the MiniLM weights and
runs one embed to force the full load. Run this once during setup:

    python -m app.prefetch_embeddings

It's a no-op cost after the first run (weights are cached), and it prints the
resolved backend so you can confirm the demo box will use local embeddings and
not silently fall back to the Fireworks API mid-demo.
"""

from __future__ import annotations

import sys

from app.config import load_config
from app.embeddings import Embedder


def main() -> int:
    cfg = load_config()
    # Force local so this actually fetches weights (not the remote API path).
    embedder = Embedder(
        preferred="local",
        local_model=cfg.embed_model,
        remote_model=cfg.embed_remote_model,
        api_key=cfg.fireworks_api_key,
        base_url=cfg.fireworks_base_url,
    )
    if not embedder.warm():
        print("[prefetch] FAILED: local embedding weights could not be loaded.")
        print("           Install sentence-transformers, or run with EMBED_BACKEND=remote.")
        return 1

    _ = embedder.embed("warm the encoder end to end")
    print(f"[prefetch] ready: {embedder.active_backend} (dim={embedder.dim})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
