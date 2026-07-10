# TokenSaver

A hybrid, token-efficient routing agent. Only the **remote** (Fireworks AI) path
is scored, so TokenSaver stacks several free layers in front of it and sends the
strong remote model as little text, as rarely, as possible.

## Routing flow

```
query
  │
  ├─ pre-flight optimize (free)            collapse whitespace, strip greetings/
  │                                        politeness/filler, dedup repeated sentences
  │                                        → normalized prompt (measured with a real tokenizer)
  ├─ cache    (semantic, free, ~ms)        exact or ≥0.95-cosine duplicate of the
  │                                        NORMALIZED prompt? → answer, 0 tokens, skip below
  ├─ classify (free)                       CLASSIFIER=llm → heuristic pre-pass, local 7B
  │                                        judges only the ambiguous gray zone (≤5 tokens)
  │                                        CLASSIFIER=heuristic → regex/length rules only
  ├─ IDS      (regex, free)                heavy-compute signature? → force remote
  │
  ├─ local model (free, bf16)  ─ verify ─ confident? → done (cached)
  │                                       └ not confident → escalate
  ├─ cheap-remote tier (opt-in)            no GPU? easy tasks → smaller Fireworks model
  └─ remote model (SCORED)
        ├─ transient error → retry once, then failover to local
        └─ only trusted answers are cached (never failover / flagged ones)

Every routed request is folded into a MetricsRegistry (GET /metrics, /dashboard):
prompt tokens before→after, tokens saved per layer, exact vs semantic hit rate,
model routed to, and estimated $ saved from a per-model price table.
```

## Normalize-before-cache

Pre-flight now runs **before** the cache key is computed, so two prompts that
differ only by a greeting or a "please" collapse to the same normalized text
and share one exact-cache entry — raising the exact hit rate for free. Filler
removal is quote- and context-aware: quoted subjects (`"…"` / `'…'`) and code
survive verbatim, `"a just war"` keeps its adjective, and repeated boilerplate
instruction sentences are de-duplicated (where templated prompts hide real
savings). Savings are measured with a real tokenizer (tiktoken `cl100k_base`,
falling back to a chars/4 estimate) — not guessed.

## Token-saving layers

| Layer | File | What it saves |
| --- | --- | --- |
| **LLM-as-a-Judge classifier** | [app/router/classifier.py](app/router/classifier.py) | The local model judges its own ability (≤5 output tokens, free in scoring) and only routes REMOTE when it genuinely needs the stronger model. Biased LOCAL on ambiguity — the verifier still escalates bad local answers. `CLASSIFIER=heuristic` keeps the regex/length rules for benchmarking. |
| **Pre-flight optimizer** | [app/optimizer/preflight.py](app/optimizer/preflight.py) | Strips signal-free whitespace, greetings, and politeness ("please", "thank you so much") from every remote prompt — a lossless few-percent discount on prompt tokens. Never touches fenced code blocks. |
| **Regex IDS** | [app/router/ids.py](app/router/ids.py) | Precompiled regexes sniff prompts for heavy-compute signatures (code fences, SQL, proofs, big-O, dense math) and force the remote route when the cheap classifier would have under-called it. |
| **Semantic cache** | [app/cache.py](app/cache.py) | faiss over embeddings from the shared [app/embeddings.py](app/embeddings.py) backend: exact *and* near-duplicate queries (cosine ≥ `CACHE_SIMILARITY_THRESHOLD`, default 0.95) are served from memory for **zero** remote tokens. Degrades to exact hashing if the stack is missing. Set `CACHE_PERSIST_PATH` to write-through to sqlite so savings compound across restarts. |
| **Embedding backend** | [app/embeddings.py](app/embeddings.py) | One embedder shared by the semantic cache *and* the `--validate` scorer. `EMBED_BACKEND=local` (default) loads `all-MiniLM-L6-v2` once at `warm()`; `EMBED_BACKEND=remote` calls the Fireworks embeddings API. Whichever is preferred, the other is the automatic fallback, and the resolved backend is logged at startup. Pre-fetch local weights with `python -m app.prefetch_embeddings` so first-request latency stays off the demo path. |
| **Cheap-remote tier** | [app/router/agent.py](app/router/agent.py) | Set `CHEAP_REMOTE_MODEL` to route "easy" tasks to a smaller/cheaper Fireworks model when no local GPU is available — the router still saves the price delta vs. the main model. |
| **Failover** | [app/router/agent.py](app/router/agent.py) | If the remote call errors, the local model serves the answer instead of failing the request. |

## Metrics & guardrail

```bash
# Live savings dashboard (polls /metrics every 2s)
uvicorn app.main:app --port 8000    # then open http://localhost:8000/dashboard

# GET /metrics — cumulative totals + per-layer breakdown (compression vs exact
# vs semantic cache vs routing), hit rates (sourced from CacheStats), and $ saved.

# Guardrail: run every sample prompt BOTH raw and optimized, diff the answers.
# Prints % identical, % degraded, and worst offenders with per-stage attribution.
python -m app.eval.run_eval --validate
```

`/route` responses now split the old `est_tokens_saved` into two honest fields:
`preflight_tokens_saved` (compression on a live call) and `remote_tokens_avoided`
(spend a cache hit skipped), plus `prompt_tokens_before`/`prompt_tokens_after`
from real provider usage data and the concrete `model_used`.

### Threshold calibration

[tests/test_threshold_calibration.py](tests/test_threshold_calibration.py) pins
the 0.95 cutoff to the actual embedding model and fails loudly if it drifts.
Measured on `all-MiniLM-L6-v2`: the confusable pair *"capital of Austria"* vs
*"capital of Australia"* scores **0.62** — safely rejected — while a strong
paraphrase (*"capital of France"* vs *"capital city of France"*) scores **0.95**.
Note the model's ceiling: looser paraphrases (*"summarize the text"* vs *"give me
a summary"*) also land near **0.62**, so at 0.95 the semantic cache only catches
near-exact restatements. If you want broader paraphrase caching, either lower
`CACHE_SIMILARITY_THRESHOLD` toward ~0.90 (still far above the 0.62 confusable
floor) or switch to a stronger embedder via `EMBED_BACKEND=remote`.

Local model: `Qwen/Qwen2.5-7B-Instruct` in **bf16** (~15-16 GB VRAM), loaded
once and shared by the judge and the answering path. Tight on VRAM? Set
`LOCAL_MODEL=Qwen/Qwen2.5-3B-Instruct` (or a 1.5B/2B model).

Every layer is togglable via env vars (see [.env.example](.env.example)) and
injectable into `Agent(...)`, so a fine-tuned model can replace any of them.

## Run

```bash
pip install -r requirements.txt

# Pre-fetch the local embedding weights (keeps first-request latency off the demo path)
python -m app.prefetch_embeddings

# API  (keyless demo? add REMOTE_STUB=true so the dashboard shows real savings)
uvicorn app.main:app --port 8000
#   GET  /health
#   POST /route   {"query": "..."} → {answer, route, remote_tokens, local_tokens,
#                  cached, cache_hit_type, preflight_tokens_saved, remote_tokens_avoided,
#                  prompt_tokens_before, prompt_tokens_after, model_used, trace}
#   GET  /metrics     cumulative savings + per-layer breakdown
#   GET  /dashboard   live HTML view of /metrics

# Eval harness (no key / GPU needed in classify-only mode)
python -m app.eval.run_eval --classify-only

# Guardrail: prove optimization didn't change the answers
python -m app.eval.run_eval --validate

# Reproduce the benchmark: prints per-task routing + the baseline-vs-TokenSaver
# remote-token delta (%). Keyless (stub remote); set LOCAL_MODEL to a small model
# for a quick CPU run, e.g. LOCAL_MODEL=Qwen/Qwen2.5-0.5B-Instruct
python baseline_delta.py

# Tests
pytest
```

The response `trace` records every layer's decision (preflight → cache → classify
→ ids → local → verify → remote/failover) with real per-call token counts, so you
can see exactly where tokens were spent or avoided.

## Benchmark

Measured on a **36-task mixed workload** (~60% easy lookups / ~40% hard
reasoning — not a curated 50/50) via `baseline_delta.py`. The routing split is
compared against a naive all-remote baseline.

> **Source labelling (read this before quoting a number).** These figures come
> from a **0.5B model on CPU** with the **keyless proportional stub** standing in
> for the remote provider, using the **heuristic** classifier for determinism.
> They are **stub-derived estimates**, not the scored config (7B + real
> Fireworks). Treat them as directional, not billing truth.

| Metric | Result |
| --- | --- |
| Tasks routed to the free local tier | 26 / 36 |
| Remote tokens — baseline (all-remote) vs TokenSaver | 988 → 329 |
| **Raw remote-token reduction** | **66.7%** |
| **Quality-adjusted reduction** | **~54%** |
| Cache hit rate (stream w/ repeats) | 41.7% (exact 4, semantic 1) |

**Why quality-adjusted < raw.** The heuristic under-routed 4 genuinely hard
tasks (a SQL query, a Bloom-filter derivation, an IPv4 regex, an ML gradient) to
the local tier — some with a lone heavy cue, some with signals split across the
classifier and the regex IDS, one phrased with no trigger word at all, none
clearing the routing threshold — inflating the raw savings at the cost of answer
quality. Counting those as
remote (where they belong) gives the honest **~54%**. The production default is
the LLM judge, which re-evaluates exactly that gray zone; the heuristic
benchmarked here therefore **understates routing accuracy**.

**Cache.** Exact repeats hit reliably (4/4); at the default `0.95` cosine
threshold the semantic layer barely fires (1 of 4 near-duplicates) — in practice
it behaves close to an exact cache. The 41.7% hit rate reflects a stream
deliberately seeded with repeats; real hit rate tracks your workload's
duplication.

## Docker

The default image is **CPU-portable** — the API, router, classifier, semantic
cache, and every token-saving layer build and run without a GPU (the heavy local
model is imported lazily).

```bash
docker build -t tokensaver .
docker run --rm -p 8000:8000 \
  -e FIREWORKS_API_KEY=your_key_here \
  tokensaver
# then: http://localhost:8000/health  and  /dashboard
```

Keyless demo (no Fireworks key): add `-e REMOTE_STUB=true` so the scored remote
is swapped for a proportional stub and the dashboard still shows real savings.

### Running on an AMD GPU (ROCm)

To run the free local model on an AMD GPU, switch the base image in the
[Dockerfile](Dockerfile) from `python:3.11-slim` to the official ROCm PyTorch
image and enable the local ML stack:

```dockerfile
FROM rocm/pytorch:latest
# torch/transformers already present; ROCm exposes the GPU to PyTorch as "cuda",
# which is exactly what LOCAL_DEVICE defaults to — no code change needed.
```

Then run with device passthrough:

```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  -e FIREWORKS_API_KEY=your_key_here -p 8000:8000 tokensaver
```

`LOCAL_DEVICE=cuda` is the default precisely because ROCm presents AMD GPUs as
the `cuda` device in PyTorch. To force CPU, set `LOCAL_DEVICE=cpu`.

## 4-bit quantization (memory footprint)

The local model can load in **4-bit NF4** via `bitsandbytes`, which drops
`Qwen2.5-7B-Instruct` from ~15–16 GB (bf16) to ~5–6 GB of VRAM — the difference
between fitting and OOM/paging-file crashes on a 16 GB card. Compute still runs
in bf16 for accuracy (double-quant also compresses the quant constants).

`load()` in [app/models/local_model.py](app/models/local_model.py) is
**environment-aware** so the *same code* runs on both machines:

| Environment | `ENV` | 4-bit default | Why |
| --- | --- | --- | --- |
| **Windows dev** | unset / `development` | **off** (bf16/fp32) | Stock `bitsandbytes` is CUDA-only and can crash on import; keep local testing smooth. |
| **Linux/ROCm prod** | `production` | **on** (4-bit NF4) | Locks into the optimized config automatically on the deploy box. |

Precedence: an explicit `load_in_4bit=...` arg wins, then the `LOAD_IN_4BIT` env
var, then the `ENV`-based auto-default. Even when 4-bit is requested, the
`bitsandbytes` import is **guarded** — if it's missing or its kernels won't load,
the loader logs an install hint and falls back to bf16/fp32 instead of crashing.

### Installing bitsandbytes (dev vs. deploy differ!)

⚠️ **`bitsandbytes` is not one package across platforms** — install the build
that matches each machine:

**Windows dev laptop** (only needed if you set `LOAD_IN_4BIT=true` locally; you
generally *don't* need it since dev defaults to bf16/fp32):

```powershell
pip install bitsandbytes accelerate
```

The stock PyPI wheel targets **CUDA/NVIDIA**. It only does real 4-bit work on an
NVIDIA GPU; on an AMD/CPU Windows box it may fail to import — which is exactly
why dev defaults to *off* and the import is guarded.

**Linux/ROCm production server** (Proxmox + AMD RX 9060 XT): do **not** use the
stock wheel. Install the **ROCm multi-backend build** of bitsandbytes:

```bash
pip install accelerate
# ROCm build of bitsandbytes (multi-backend) — see the official guide:
#   https://huggingface.co/docs/bitsandbytes/main/en/installation#amd-gpu
# Typically built against your ROCm version, e.g.:
pip install "bitsandbytes>=0.43" --index-url https://download.pytorch.org/whl/rocm6.1
```

ROCm exposes the GPU to PyTorch as `cuda`, so no code changes are needed — set
`ENV=production` (and `LOCAL_DEVICE=cuda`, the default) and 4-bit engages
automatically. Verify with the `[local_model] loaded '…' in 4-bit NF4` log line
printed at model load.

## Project structure

```
tokensaver/
├── app/
│   ├── main.py              # FastAPI server (/health, /route, /metrics, /dashboard)
│   ├── config.py            # env-driven config
│   ├── router/
│   │   ├── classifier.py    # heuristic + LLM-as-a-Judge classifiers (swappable)
│   │   ├── agent.py         # routing + verification core (+ failover, cheap-remote)
│   │   └── ids.py           # regex heavy-compute detector
│   ├── optimizer/preflight.py  # lossless prompt compression
│   ├── cache.py             # exact + semantic (faiss) cache
│   ├── embeddings.py        # shared embedding backend (local MiniLM / Fireworks)
│   ├── metrics.py, pricing.py, tokens.py
│   ├── models/
│   │   ├── local_model.py   # local model on AMD GPU / ROCm (lazy)
│   │   ├── remote_model.py  # Fireworks AI client (returns text + total tokens)
│   │   └── stub_remote.py   # keyless demo stub
│   └── eval/
│       ├── run_eval.py      # eval harness (--classify-only / --validate)
│       └── sample_tasks.json
├── tests/                   # run WITHOUT a GPU or API key
├── Dockerfile · requirements.txt · .env.example · .gitignore · .dockerignore
└── LICENSE (MIT) · README.md
```

## Configuration

All config is environment-driven with safe defaults (see
[.env.example](.env.example) and [app/config.py](app/config.py)); the service
runs out of the box and every layer is togglable.

| Variable | Default | Purpose |
| --- | --- | --- |
| `FIREWORKS_API_KEY` | `""` | Fireworks AI key for the scored remote path. |
| `FIREWORKS_BASE_URL` | `https://api.fireworks.ai/inference/v1` | Remote API base URL. |
| `REMOTE_MODEL` | `accounts/fireworks/models/gemma-2-9b-it` | Scored remote model. |
| `CHEAP_REMOTE_MODEL` | `""` | Optional smaller Fireworks model for easy tasks (no GPU). |
| `LOCAL_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Free local model (also the judge). |
| `LOCAL_DEVICE` | `cuda` | PyTorch device; ROCm exposes AMD GPUs as `cuda`. |
| `ENV` | `development` | `production` auto-enables 4-bit; also `APP_ENV`/`ENVIRONMENT`. |
| `LOAD_IN_4BIT` | *(auto)* | Force 4-bit on/off. Unset → on in production, off in dev. |
| `CLASSIFIER` | `llm` | `llm` (judge via local model) or `heuristic` (regex/length). |
| `CONFIDENCE_THRESHOLD` | `0.6` | Verifier self-check cutoff for escalation. |
| `COMPLEXITY_LENGTH_CUTOFF` | `24` | Word count that biases a query toward "hard". |
| `MAX_NEW_TOKENS` | `256` | Generation cap for local + remote. |
| `REMOTE_TIMEOUT` | `60` | Remote request timeout (s). |
| `REMOTE_RETRIES` | `1` | Extra attempts on transient 429/5xx/network errors. |
| `WARMUP_ENABLED` | `true` | Preload local model + cache encoder at boot. |
| `REMOTE_STUB` | `false` | Keyless demo: swap the scored remote for a stub. |
| `PREFLIGHT_ENABLED` | `true` | Strip whitespace/politeness/filler before remote. |
| `CACHE_ENABLED` | `true` | Serve (near-)duplicate queries for zero remote tokens. |
| `CACHE_BACKEND` | `semantic` | `semantic` (MiniLM + faiss) or `exact` (hash only). |
| `CACHE_SIMILARITY_THRESHOLD` | `0.95` | Cosine cutoff for a semantic cache hit. |
| `CACHE_MAX_ENTRIES` | `512` | LRU cap on the query cache. |
| `CACHE_PERSIST_PATH` | `""` | sqlite file to persist the cache across restarts. |
| `EMBED_BACKEND` | `local` | `local` (MiniLM) or `remote` (Fireworks embeddings). |
| `IDS_ENABLED` | `true` | Regex inspection for heavy-compute signatures. |
| `IDS_THRESHOLD` | `0.5` | Severity at which IDS forces the remote route. |
| `FAILOVER_ENABLED` | `true` | On remote failure, serve from the local model. |

## Tests

Unit tests run with **no GPU and no API key** — they exercise the classifier,
verifier, routing decisions, cache, preflight, IDS, embeddings, and metrics with
models mocked or absent.

```bash
pytest
```

## Limitations

Honest scope of what the numbers above do and don't prove:

- **Not the scored config.** The benchmark ran on a **0.5B model on CPU** with a
  **keyless stub** remote, not the 7B + real Fireworks path. Token counts,
  routing quality, and the delta will differ on the scored config.
- **No answer-quality validation.** The savings assume the locally-answered
  tasks are acceptable; that isn't verified. So "saves ~54% **at equal quality**"
  is *not* yet supported — quality is the next thing to measure.
- **Routing has a known gap.** Hard tasks whose heavy signals each fall below the
  threshold (or that use no trigger word at all) can be under-routed to local.
  The LLM judge targets this gray zone but is biased toward local by design and
  is unmeasured on these cases at 7B.
- **Semantic cache is conservative.** At `0.95` it catches near-exact
  restatements only; broader paraphrase caching needs a lower threshold or a
  stronger embedder. Hit rate is workload-dependent.
- **No latency benchmark.** Local-vs-remote latency needs the GPU deploy box; it
  hasn't been measured.
- **Token counts are approximate.** Counting uses tiktoken `cl100k`, not the
  provider's tokenizer — consistent before/after, but not exact billing.

**Next step to firm up the headline:** run the same benchmark on the 7B with a
real key and add an answer-quality check, to confirm the ~54% holds *at quality*.

## License

MIT — see [LICENSE](LICENSE).
