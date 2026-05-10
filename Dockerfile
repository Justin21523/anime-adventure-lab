# Dockerfile for Anime-Adventure-Lab

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Allow choosing a requirements file at build time (e.g. requirements-docker.txt)
ARG REQUIREMENTS_FILE=requirements.txt

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY ${REQUIREMENTS_FILE} ./requirements.txt

# Install Python dependencies (limit parallelism to avoid OOM)
ENV PIP_NO_CACHE_DIR=1 PIP_MAX_WORKERS=1
RUN pip install -r requirements.txt && pip cache purge 2>/dev/null; true

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /warehouse/ai_cache /warehouse/ai_models /warehouse/ai_output /app/logs

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    AI_CACHE_ROOT=/warehouse/ai_cache \
    AI_MODELS_ROOT=/warehouse/ai_models \
    AI_OUTPUT_ROOT=/warehouse/ai_output \
    HF_HOME=/warehouse/ai_cache/huggingface \
    TRANSFORMERS_CACHE=/warehouse/ai_cache/huggingface \
    TORCH_HOME=/warehouse/ai_cache/torch \
    XDG_CACHE_HOME=/warehouse/ai_cache

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Default command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
