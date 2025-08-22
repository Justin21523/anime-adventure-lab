# api/dependencies.py
import os, pathlib, torch
from typing import Iterator

AI_CACHE_ROOT = os.getenv("AI_CACHE_ROOT", "../ai_warehouse/cache")
for k, v in {
    "HF_HOME": f"{AI_CACHE_ROOT}/hf",
    "TRANSFORMERS_CACHE": f"{AI_CACHE_ROOT}/hf/transformers",
    "HF_DATASETS_CACHE": f"{AI_CACHE_ROOT}/hf/datasets",
    "HUGGINGFACE_HUB_CACHE": f"{AI_CACHE_ROOT}/hf/hub",
    "TORCH_HOME": f"{AI_CACHE_ROOT}/torch",
}.items():
    os.environ[k] = v
    pathlib.Path(v).mkdir(parents=True, exist_ok=True)

APP_DIRS = {
    "MODELS_SD": f"{AI_CACHE_ROOT}/models/sd",
    "MODELS_SDXL": f"{AI_CACHE_ROOT}/models/sdxl",
    "MODELS_CONTROLNET": f"{AI_CACHE_ROOT}/models/controlnet",
    "MODELS_LORA": f"{AI_CACHE_ROOT}/models/lora",
    "MODELS_IPADAPTER": f"{AI_CACHE_ROOT}/models/ipadapter",
    "DATASETS_META": f"{AI_CACHE_ROOT}/datasets/metadata",
    "OUTPUT_DIR": f"{AI_CACHE_ROOT}/outputs/saga-forge",
    # RAG
    "RAG_INDEX": f"{AI_CACHE_ROOT}/rag/indexes",
    "RAG_DOCS": f"{AI_CACHE_ROOT}/rag/documents",
    "RAG_EMBEDDINGS": f"{AI_CACHE_ROOT}/rag/embeddings",
    "WORLDPACKS": f"{AI_CACHE_ROOT}/worldpacks",
}
for p in APP_DIRS.values():
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)


def get_cache_root() -> str:
    return AI_CACHE_ROOT
