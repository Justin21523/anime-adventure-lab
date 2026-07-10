# Dockerfile for Anime-Adventure-Lab (CPU-only / API container)
#
# GPU workloads (T2I, VLM, training) are NOT run inside this container.
# On AMD ROCm hosts, run Celery workers natively in the host conda env
# (see scripts/run-worker.sh).
#
# This image builds the FastAPI API server + Redis-dependent services.

FROM python:3.10-slim

WORKDIR /app

ARG REQUIREMENTS_FILE=requirements-docker.txt

# System deps (no GPU drivers needed — API is CPU-only)
RUN apt-get update && apt-get install -y \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY ${REQUIREMENTS_FILE} ./requirements.txt

ENV PIP_NO_CACHE_DIR=1 PIP_MAX_WORKERS=1
# Install without torch/xformers — those live on the host for workers
RUN pip install -r requirements.txt && pip cache purge 2>/dev/null; true

COPY . .

RUN mkdir -p /warehouse/ai_cache /warehouse/ai_models /warehouse/ai_output /app/logs

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    AI_CACHE_ROOT=/warehouse/ai_cache \
    AI_MODELS_ROOT=/warehouse/ai_models \
    AI_OUTPUT_ROOT=/warehouse/ai_output \
    HF_HOME=/warehouse/ai_cache/huggingface \
    TRANSFORMERS_CACHE=/warehouse/ai_cache/huggingface \
    TORCH_HOME=/warehouse/ai_cache/torch \
    XDG_CACHE_HOME=/warehouse/ai_cache

# Default LLM backend: external llama.cpp server (HTTP)
ENV LLM_BACKEND=llamacpp \
    LLAMA_CPP_SERVER=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
