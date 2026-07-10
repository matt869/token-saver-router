# TokenSaver — containerized service.
#
# Default base is CPU-portable so the API, router, classifier, semantic cache,
# and all token-saving layers build and run anywhere (CI, laptops, the demo box)
# without a GPU. The heavy local-model path (torch/transformers) is imported
# lazily, so it's fine that this image ships without them.
#
# ── Switch to AMD GPU / ROCm ────────────────────────────────────────────────
# To run the free local model on an AMD GPU, swap the base image for the
# official ROCm PyTorch image and drop the CPU torch install:
#
#     FROM rocm/pytorch:latest
#     # torch/transformers already present; ROCm exposes the GPU as "cuda".
#     # Then `pip install -r requirements.txt` (torch line stays commented) and
#     # uncomment torch/transformers in requirements.txt OR install them here.
#
# Run with device passthrough:
#     docker run --rm -it \
#       --device=/dev/kfd --device=/dev/dri \
#       --group-add video --ipc=host \
#       -e FIREWORKS_API_KEY=... -p 8000:8000 tokensaver
# ────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first for better layer caching.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# App source.
COPY app ./app

# --- Track 1 batch submission defaults --------------------------------------
# The scoring harness runs this as a batch job: read /input/tasks.json, route
# every task, write /output/results.json, exit 0.
ENV INPUT_PATH=/input/tasks.json \
    OUTPUT_PATH=/output/results.json \
    RESULT_FORMAT=objects
# RESULT_FORMAT=objects -> results.json is [{"id": ..., "answer": ...}], matching
# the Track 1 baseline. Override to "strings" only if a harness wants bare strings.
# Pin a SMALL local model. The 7B OOM'd on a 14 GB box (fp32 ~28 GB); a 1.5B is
# ~3 GB in bf16 / ~6 GB fp32 and fits any GPU pod with headroom -> no OOM risk.
# Override at run time if the harness pins a different local model.
ENV LOCAL_MODEL=Qwen/Qwen2.5-1.5B-Instruct
# REMOTE_MODEL stays overridable so the harness can pin a specific Fireworks
# model; FIREWORKS_API_KEY / FIREWORKS_BASE_URL are read from the environment.

# --- Bake weights so the graded run needs NO Hugging Face download -----------
# Pin cache dirs so the files fetched here are exactly what runs at inference
# time (any user). `snapshot_download` only fetches files — no GPU, no model
# instantiation — so this step is identical on the CPU base and the
# rocm/pytorch GPU base. Build host needs egress; the runtime container does not.
ENV HF_HOME=/opt/hf \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken \
    HF_HUB_DISABLE_TELEMETRY=1
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${LOCAL_MODEL}')" \
 && python -m app.prefetch_embeddings \
 && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"
# All weights are baked, so you may enforce fully-offline inference at run time:
#   docker run -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 ...

EXPOSE 8000

# Default entry point = the batch job (exits 0 after writing results.json).
CMD ["python", "-m", "app.run_batch"]

# The HTTP server is still available for local/interactive use — run it with:
#   docker run --entrypoint uvicorn <img> app.main:app --host 0.0.0.0 --port 8000
