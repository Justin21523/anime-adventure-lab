#!/usr/bin/env bash
# Run a Celery worker natively on the ROCm host.
#
# Usage:
#   bash scripts/run-worker.sh [QUEUE_NAME]
#
# Queues: generation (default), postprocess, training, all
#
# Requires: conda env rocm-pytorch-r9700 activated beforehand.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
QUEUE="${1:-generation}"

# Load project .env if present
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.env"
    set +a
fi

# Ensure critical env vars are set for ROCm
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# AI_WAREHOUSE roots (use .env values or defaults)
export AI_CACHE_ROOT="${AI_CACHE_ROOT:-/mnt/c/ai_cache}"
export AI_MODELS_ROOT="${AI_MODELS_ROOT:-/mnt/c/ai_models}"
export AI_OUTPUT_ROOT="${AI_OUTPUT_ROOT:-/mnt/c/ai_output/anime-adventure-lab}"
export HF_HOME="${HF_HOME:-/mnt/c/ai_cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/mnt/c/ai_cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/mnt/c/ai_cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/c/ai_cache}"

# Disable mocks when running on host with GPU
export T2I_MOCK=0
export VLM_MOCK=0
export LLM_MOCK=0

# LLM backend
export LLM_BACKEND="${LLM_BACKEND:-llamacpp}"
export LLAMA_CPP_SERVER="${LLAMA_CPP_SERVER:-1}"
export LLAMA_SERVER_URL="${LLAMA_SERVER_URL:-http://localhost:8080}"

echo "================================================"
echo "  Anime-Adventure-Lab Celery Worker (ROCm Host)"
echo "================================================"
echo "  Queue     : $QUEUE"
echo "  Project   : $PROJECT_DIR"
echo "  GPU       : $(rocm-smi --showproductName 2>/dev/null | tail -1 || echo 'unknown')"
echo "  Python    : $(python --version 2>&1)"
echo "  Torch     : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed')"
echo "  CUDA/ROCm : $(python -c 'import torch; print(torch.version.hip or torch.version.cuda or "none")' 2>/dev/null || echo 'unknown')"
echo "================================================"

cd "$PROJECT_DIR"

case "$QUEUE" in
    generation)
        celery -A workers.celery_app:celery_app worker \
            -l INFO \
            --queues=generation \
            --concurrency=2 \
            --logfile=logs/worker-generation.log \
            --pidfile=logs/worker-generation.pid
        ;;
    postprocess)
        celery -A workers.celery_app:celery_app worker \
            -l INFO \
            --queues=postprocess \
            --concurrency=1 \
            --logfile=logs/worker-postprocess.log \
            --pidfile=logs/worker-postprocess.pid
        ;;
    training)
        celery -A workers.celery_app:celery_app worker \
            -l INFO \
            --queues=training \
            --concurrency=1 \
            --logfile=logs/worker-training.log \
            --pidfile=logs/worker-training.pid
        ;;
    all)
        celery -A workers.celery_app:celery_app worker \
            -l INFO \
            --queues=generation,postprocess,training \
            --concurrency=2 \
            --logfile=logs/worker-all.log \
            --pidfile=logs/worker-all.pid
        ;;
    *)
        echo "Unknown queue: $QUEUE"
        echo "Available: generation, postprocess, training, all"
        exit 1
        ;;
esac
