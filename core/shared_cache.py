# core/shared_cache.py
"""
Shared Model/Data Warehouse Bootstrap
Ensures consistent cache directory structure across all modules
"""

import os
import pathlib
import torch
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SharedCache:
    """Manages shared model and data warehouse directories"""

    def __init__(self, cache_root: Optional[str] = None):
        self.cache_root = cache_root or os.getenv(
            "AI_CACHE_ROOT", "../ai_warehouse/cache"
        )
        self.app_dirs = {}
        self._setup_environment()
        self._create_directories()

    def _setup_environment(self) -> None:
        """Setup HuggingFace and PyTorch cache directories"""
        cache_mappings = {
            "HF_HOME": f"{self.cache_root}/hf",
            "TRANSFORMERS_CACHE": f"{self.cache_root}/hf/transformers",
            "HF_DATASETS_CACHE": f"{self.cache_root}/hf/datasets",
            "HUGGINGFACE_HUB_CACHE": f"{self.cache_root}/hf/hub",
            "TORCH_HOME": f"{self.cache_root}/torch",
        }

        for env_key, cache_path in cache_mappings.items():
            os.environ[env_key] = cache_path
            pathlib.Path(cache_path).mkdir(parents=True, exist_ok=True)

    def _create_directories(self) -> None:
        """Create application-specific directories"""
        self.app_dirs = {
            # Model Registry file
            "TRAIN_REGISTRY": f"{self.cache_root}/train/model_registry.json",
            # Models
            "MODELS_SD": f"{self.cache_root}/models/sd",
            "MODELS_SDXL": f"{self.cache_root}/models/sdxl",
            "MODELS_CONTROLNET": f"{self.cache_root}/models/controlnet",
            "MODELS_LORA": f"{self.cache_root}/models/lora",
            "MODELS_IPADAPTER": f"{self.cache_root}/models/ipadapter",
            "MODELS_LLM": f"{self.cache_root}/models/llm",
            "MODELS_VLM": f"{self.cache_root}/models/vlm",
            "MODELS_EMBEDDING": f"{self.cache_root}/models/embedding",
            "MODELS_CLIP": f"{self.cache_root}/models/clip",
            # Datasets
            "DATASETS_RAW": f"{self.cache_root}/datasets/raw",
            "DATASETS_PROCESSED": f"{self.cache_root}/datasets/processed",
            "DATASETS_METADATA": f"{self.cache_root}/datasets/metadata",
            # Outputs
            "OUTPUT_DIR": f"{self.cache_root}/outputs/saga-forge",
            "OUTPUT_BATCH": f"{self.cache_root}/outputs/batch",
            "OUTPUT_LORA": f"{self.cache_root}/outputs/lora",
            "OUTPUT_RAG": f"{self.cache_root}/outputs/rag",
            # RAG
            "RAG_INDEX": f"{self.cache_root}/rag/indexes",
            "RAG_DOCS": f"{self.cache_root}/rag/documents",
            "RAG_EMBEDDINGS": f"{self.cache_root}/rag/embeddings",
            "WORLDPACKS": f"{self.cache_root}/worldpacks",
        }

        for dir_path in self.app_dirs.values():
            pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)

    def get_path(self, key: str) -> str:
        """Get directory path by key"""
        if key not in self.app_dirs:
            raise KeyError(f"Unknown cache directory: {key}")
        return self.app_dirs[key]

    def get_gpu_info(self) -> Dict:
        """Get GPU availability and memory info"""
        gpu_info = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": 0,
            "current_device": None,
            "memory_info": {},
        }

        if torch.cuda.is_available():
            gpu_info["device_count"] = torch.cuda.device_count()
            gpu_info["current_device"] = torch.cuda.current_device()

            try:
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                memory_total = (
                    torch.cuda.get_device_properties(0).total_memory / 1024**3
                )  # GB

                gpu_info["memory_info"] = {
                    "allocated_gb": round(memory_allocated, 2),
                    "reserved_gb": round(memory_reserved, 2),
                    "total_gb": round(memory_total, 2),
                    "free_gb": round(memory_total - memory_reserved, 2),
                }
            except Exception as e:
                logger.warning(f"Could not get GPU memory info: {e}")

        return gpu_info

    def get_summary(self) -> Dict:
        """Get cache summary information"""
        return {
            "cache_root": self.cache_root,
            "directories": self.app_dirs,
            "gpu_info": self.get_gpu_info(),
            "env_vars": {
                "HF_HOME": os.environ.get("HF_HOME"),
                "TORCH_HOME": os.environ.get("TORCH_HOME"),
                "CUDA_VISIBLE_DEVICES": os.environ.get(
                    "CUDA_VISIBLE_DEVICES", "not_set"
                ),
            },
        }


# Global instance for easy import
_shared_cache = None


def get_shared_cache(cache_root: Optional[str] = None) -> SharedCache:
    """Get or create shared cache instance"""
    global _shared_cache
    if _shared_cache is None:
        _shared_cache = SharedCache(cache_root)
    return _shared_cache


def bootstrap_cache(cache_root: Optional[str] = None) -> SharedCache:
    """Bootstrap shared cache and print summary"""
    cache = get_shared_cache(cache_root)
    gpu_info = cache.get_gpu_info()

    print(f"[🏪 SharedCache] Root: {cache.cache_root}")
    print(
        f"[🖥️  GPU] Available: {gpu_info['cuda_available']} | Devices: {gpu_info['device_count']}"
    )

    if gpu_info["memory_info"]:
        mem = gpu_info["memory_info"]
        print(
            f"[💾 VRAM] {mem['allocated_gb']:.1f}GB used / {mem['total_gb']:.1f}GB total"
        )

    return cache


if __name__ == "__main__":
    # Test bootstrap
    cache = bootstrap_cache()
    import json

    print("\n[📋 Cache Summary]")
    print(json.dumps(cache.get_summary(), indent=2))
