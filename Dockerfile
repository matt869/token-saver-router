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

EXPOSE 8000

# Start the FastAPI server. Override env vars at `docker run` time.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
