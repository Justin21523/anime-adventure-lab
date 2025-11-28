# core/shared_cache.py
"""
Shared Model/Data Warehouse Bootstrap
Ensures consistent cache directory structure across all modules
"""

import json
import os
import pathlib
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

logger = logging.getLogger(__name__)


class SharedCache:
    """Manages shared model and data warehouse directories"""

    def __init__(self, cache_root: Optional[str] = None):
        raw_root = cache_root or os.getenv(
            "AI_CACHE_ROOT", "/mnt/c/AI_LLM_projects/ai_warehouse"
        )
        root_path = Path(raw_root).expanduser()

        # Backward compatibility: if someone passes the old .../cache path, lift one level
        if root_path.name == "cache":
            root_path = root_path.parent

        self.root = root_path
        self.root.mkdir(parents=True, exist_ok=True)

        self.cache_root = Path(self.root)  # keep legacy attribute name as Path
        self.cache_dir = self.cache_root / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._memory_cache: Dict[str, Any] = {}
        self.app_dirs = {}
        self._setup_environment()
        self._create_directories()

    def _setup_environment(self) -> None:
        """Setup HuggingFace and PyTorch cache directories"""
        hf_root = self.cache_dir / "hf"
        cache_mappings = {
            "AI_CACHE_ROOT": str(self.cache_root),  # expose normalized root
            "HF_HOME": str(hf_root),
            "TRANSFORMERS_CACHE": str(hf_root / "transformers"),
            "HF_DATASETS_CACHE": str(hf_root / "datasets"),
            "HUGGINGFACE_HUB_CACHE": str(hf_root / "hub"),
            "TORCH_HOME": str(self.cache_dir / "torch"),
        }

        for env_key, cache_path in cache_mappings.items():
            os.environ[env_key] = cache_path
            pathlib.Path(cache_path).mkdir(parents=True, exist_ok=True)

    def _create_directories(self) -> None:
        """Create application-specific directories"""
        models_root = self.root / "models"
        datasets_root = self.root / "datasets"
        outputs_root = self.root / "outputs"
        rag_root = self.root / "rag"
        worldpacks_root = self.root / "worldpacks"

        self.app_dirs = {
            # Model Registry file
            "TRAIN_REGISTRY": str(self.root / "train" / "model_registry.json"),
            # Models
            "MODELS_SD": str(models_root / "stable-diffusion"),
            "MODELS_SDXL": str(models_root / "stable-diffusion" / "xl"),
            "MODELS_CONTROLNET": str(models_root / "controlnet"),
            "MODELS_LORA": str(models_root / "stable-diffusion" / "lora"),
            "MODELS_IPADAPTER": str(models_root / "ipadapter"),
            "MODELS_LLM": str(models_root / "llm"),
            "MODELS_VLM": str(models_root / "vlm"),
            "MODELS_TEXT2IMAGE": str(models_root / "stable-diffusion"),
            "MODELS_TEXT2VIDEO": str(models_root / "video"),
            "MODELS_TTS": str(models_root / "audio"),
            "MODELS_ENHANCEMENT": str(models_root / "enhancement"),
            "MODELS_TAGGING": str(models_root / "tagging"),
            "MODELS_EMBEDDING": str(models_root / "embeddings"),
            "MODELS_SAFETY": str(models_root / "safety"),
            "MODELS_AUDIO": str(models_root / "audio"),
            "MODELS_CLIP": str(models_root / "clip"),
            # Datasets
            "DATASETS_RAW": str(datasets_root / "raw"),
            "DATASETS_PROCESSED": str(datasets_root / "processed"),
            "DATASETS_METADATA": str(datasets_root / "metadata"),
            # Outputs
            "OUTPUT_DIR": str(outputs_root / "saga-forge"),
            "OUTPUT_BATCH": str(outputs_root / "batch"),
            "OUTPUT_TRAINING": str(outputs_root / "training"),
            "OUTPUT_GAMES": str(outputs_root / "games"),
            "OUTPUT_LORA": str(outputs_root / "lora"),
            "OUTPUT_RAG": str(outputs_root / "rag"),
            # RAG
            "RAG_INDEX": str(rag_root / "indexes"),
            "RAG_DOCS": str(rag_root / "documents"),
            "RAG_EMBEDDINGS": str(rag_root / "embeddings"),
            "RAG_VECTOR_STORE": str(rag_root / "knowledge_base" / "vector_store"),
            "WORLDPACKS": str(worldpacks_root),
            # Cache root (HF/torch cache lives under root/cache)
            "CACHE_DIR": str(self.cache_dir),
        }

        for dir_path in self.app_dirs.values():
            path_obj = pathlib.Path(dir_path)
            target_dir = path_obj.parent if path_obj.suffix else path_obj
            target_dir.mkdir(parents=True, exist_ok=True)

    def get_path(self, key: str) -> str:
        """Get directory path by key"""
        if key not in self.app_dirs:
            raise KeyError(f"Unknown cache directory: {key}")
        return self.app_dirs[key]

    def get_model_path(self, model_type: str, model_name: str) -> Path:
        """Get standardized model path"""
        return Path(self.cache_root) / "models" / model_type / model_name

    def get_dataset_path(self, dataset_type: str, dataset_name: str) -> Path:
        """Get standardized dataset path"""
        return Path(self.cache_root) / "datasets" / dataset_type / dataset_name

    def get_output_path(self, output_type: str = "multi-modal-lab") -> Path:
        """Get output directory path"""
        path = Path(self.cache_root) / "outputs" / output_type
        path.mkdir(parents=True, exist_ok=True)
        return path

    def cache_model_info(self, model_key: str, model_info: Dict[str, Any]) -> None:
        """Cache model information with timestamp"""
        cache_info = {
            **model_info,
            "cached_at": datetime.now().isoformat(),
            "cache_key": model_key,
        }

        info_file = Path(self.cache_root) / "models" / f"{model_key}_info.json"
        info_file.parent.mkdir(parents=True, exist_ok=True)

        with open(info_file, "w", encoding="utf-8") as f:
            json.dump(cache_info, f, indent=2, ensure_ascii=False)

    def get_model_info(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached model information"""
        info_file = Path(self.cache_root) / "models" / f"{model_key}_info.json"

        if not info_file.exists():
            return None

        try:
            with open(info_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load model info for {model_key}: {e}")
            return None

    def set_memory_cache(self, key: str, value: Any, ttl_minutes: int = 60) -> None:
        """Set value in memory cache with TTL"""
        expires_at = datetime.now() + timedelta(minutes=ttl_minutes)
        self._memory_cache[key] = {"value": value, "expires_at": expires_at}

    def get_memory_cache(self, key: str) -> Optional[Any]:
        """Get value from memory cache"""
        if key not in self._memory_cache:
            return None

        cache_item = self._memory_cache[key]
        if datetime.now() > cache_item["expires_at"]:
            del self._memory_cache[key]
            return None

        return cache_item["value"]

    def clear_memory_cache(self) -> None:
        """Clear all memory cache"""
        self._memory_cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        import psutil
        import torch

        stats = {
            "cache_root": str(self.cache_root),
            "disk_usage": {},
            "memory_cache_size": len(self._memory_cache),
            "gpu_available": torch.cuda.is_available(),
        }

        # Calculate disk usage
        for subdir in ["models", "datasets", "outputs"]:
            path = Path(self.cache_root) / subdir
            if path.exists():
                total_size = sum(
                    f.stat().st_size for f in path.rglob("*") if f.is_file()
                )
                stats["disk_usage"][subdir] = {
                    "size_gb": round(total_size / (1024**3), 2),
                    "files_count": len(list(path.rglob("*"))),
                }

        # GPU memory if available
        if torch.cuda.is_available():
            stats["gpu_memory"] = {
                "total_gb": round(
                    torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
                ),
                "allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 2),
                "cached_gb": round(torch.cuda.memory_reserved() / (1024**3), 2),
            }

        return stats

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

    def get_device_config(self, device) -> Dict[str, Any]:
        """Get device configuration for model loading"""
        try:
            config = {
                "device": (
                    device
                    if device != "auto"
                    else ("cuda" if torch.cuda.is_available() else "cpu")
                ),
                "torch_dtype": "float16" if torch.cuda.is_available() else "float32",
            }

            # VRAM optimization for low-end GPUs
            if torch.cuda.is_available():
                try:
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (
                        1024**3
                    )  # GB
                    if gpu_memory < 8:
                        config.update(
                            {
                                "enable_attention_slicing": True,
                                "enable_vae_slicing": True,
                                "enable_cpu_offload": True,
                            }
                        )
                        logger.info("Enabled VRAM optimization for low-memory GPU")
                except:
                    pass

            return config

        except ImportError:
            return {"device": "cpu", "torch_dtype": "float32"}

    def get_summary(self) -> Dict:
        """Get cache summary information"""
        return {
            "cache_root": self.cache_root,
            "directories": self.app_dirs,
            "gpu_info": self.get_gpu_info(),
            "env_vars": {
                "AI_CACHE_ROOT": os.environ.get("AI_CACHE_ROOT"),
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


# Legacy alias for tests
def setup_shared_cache(cache_root: Optional[str] = None) -> SharedCache:
    """Initialize shared cache (alias for bootstrap_cache)."""
    return bootstrap_cache(cache_root)


if __name__ == "__main__":
    # Test bootstrap
    cache = bootstrap_cache()
    import json

    print("\n[📋 Cache Summary]")
    print(json.dumps(cache.get_summary(), indent=2))
