# core/shared_cache.py
"""
Shared filesystem bootstrap (AI_WAREHOUSE 3.0 friendly).

This project used to rely on a single "warehouse" root (AI_CACHE_ROOT) that
contained models/datasets/outputs and also HF/torch cache under a nested
`cache/` folder.

We now split storage into:
- AI_CACHE_ROOT   -> /mnt/c/ai_cache          (HF/torch/XDG cache)
- AI_MODELS_ROOT  -> /mnt/c/ai_models         (managed weights / LoRAs / checkpoints)
- AI_OUTPUT_ROOT  -> /mnt/c/ai_output/...     (runs, training outputs, generated media)

The class keeps best-effort backward compatibility when pointing AI_CACHE_ROOT
to the legacy "ai_warehouse" layout.
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
        def _looks_like_legacy_warehouse(path: Path) -> bool:
            name = path.name.lower()
            if "warehouse" in name:
                return True
            try:
                # Heuristic: old layout typically has {cache,models,outputs} at top level.
                return (path / "cache").exists() and (
                    (path / "models").exists()
                    or (path / "outputs").exists()
                    or (path / "datasets").exists()
                )
            except Exception:
                return False

        # -------- Roots (new layout) --------
        cache_root_env = cache_root or os.getenv("AI_CACHE_ROOT", "/mnt/c/ai_cache")
        models_root_env = os.getenv("AI_MODELS_ROOT")
        output_root_env = os.getenv("AI_OUTPUT_ROOT")

        cache_root_path = Path(str(cache_root_env)).expanduser()

        # Backward compatibility: if someone passes the old .../cache path, lift one level
        if cache_root_path.name == "cache":
            cache_root_path = cache_root_path.parent

        force_new_layout = bool(models_root_env or output_root_env)
        self.legacy_mode = (not force_new_layout) and _looks_like_legacy_warehouse(
            cache_root_path
        )

        if self.legacy_mode:
            # Legacy single-root warehouse mode
            self.root = cache_root_path
            self.root.mkdir(parents=True, exist_ok=True)

            # Keep legacy attribute names
            self.cache_root = Path(self.root)
            self.cache_dir = self.cache_root / "cache"
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            self.models_root = self.cache_root / "models"
            self.outputs_root = self.cache_root / "outputs"
            self.datasets_root = self.cache_root / "datasets"
            self.rag_root = self.cache_root / "rag"
            self.worldpacks_root = self.cache_root / "worldpacks"

            self.hf_home = self.cache_dir / "hf"
            self.torch_home = self.cache_dir / "torch"
        else:
            # AI_WAREHOUSE 3.0 split roots
            self.cache_root = cache_root_path
            self.root = self.cache_root  # legacy alias
            self.cache_dir = self.cache_root  # legacy alias (no nested cache/)

            self.models_root = Path(
                os.getenv("AI_MODELS_ROOT", "/mnt/c/ai_models")
            ).expanduser()
            self.outputs_root = Path(
                os.getenv(
                    "AI_OUTPUT_ROOT",
                    "/mnt/c/ai_output/anime-adventure-lab",
                )
            ).expanduser()

            data_root = Path(os.getenv("AI_DATA_ROOT", "/mnt/c/ai_data")).expanduser()
            project_slug = os.getenv("AI_PROJECT_SLUG", "anime-adventure-lab")
            self.datasets_root = Path(
                os.getenv(
                    "AI_DATASETS_ROOT",
                    str(Path("/mnt/c/ai_datasets") / project_slug),
                )
            ).expanduser()
            self.rag_root = Path(
                os.getenv("AI_RAG_ROOT", str(self.outputs_root / "rag"))
            ).expanduser()
            self.worldpacks_root = Path(
                os.getenv("AI_WORLDPACKS_ROOT", str(self.datasets_root / "worldpacks"))
            ).expanduser()

            # Cache locations (match data_model_structure.md)
            self.hf_home = Path(
                os.getenv("HF_HOME", str(self.cache_root / "huggingface"))
            ).expanduser()
            self.torch_home = Path(
                os.getenv("TORCH_HOME", str(self.cache_root / "torch"))
            ).expanduser()

        # Ensure base dirs exist (best-effort; CI/restricted envs may not allow /mnt)
        for p in [
            self.cache_root,
            self.hf_home,
            self.torch_home,
            self.models_root,
            self.outputs_root,
            self.datasets_root,
            self.rag_root,
            self.worldpacks_root,
        ]:
            try:
                p.mkdir(parents=True, exist_ok=True)
            except (PermissionError, FileExistsError):
                # Avoid crashing on init in restricted environments or weird filesystem states.
                if p.exists() and p.is_dir():
                    continue
                logger.warning("Could not ensure directory exists: %s", p)

        self._memory_cache: Dict[str, Any] = {}
        self.app_dirs = {}
        self._setup_environment()
        self._create_directories()

    def _setup_environment(self) -> None:
        """Setup HuggingFace and PyTorch cache directories"""
        hf_root = Path(self.hf_home)
        cache_mappings = {
            # Root pointers
            "AI_CACHE_ROOT": str(self.cache_root),
            "AI_MODELS_ROOT": str(self.models_root),
            "AI_OUTPUT_ROOT": str(self.outputs_root),
            "AI_DATASETS_ROOT": str(self.datasets_root),
            "AI_RAG_ROOT": str(self.rag_root),
            "AI_WORLDPACKS_ROOT": str(self.worldpacks_root),
            # HuggingFace / Transformers cache (keep everything out of $HOME/.cache)
            "HF_HOME": str(hf_root),
            "TRANSFORMERS_CACHE": str(hf_root),
            "HF_DATASETS_CACHE": str(hf_root / "datasets"),
            "HUGGINGFACE_HUB_CACHE": str(hf_root / "hub"),
            # Torch cache
            "TORCH_HOME": str(self.torch_home),
            # General cache root
            "XDG_CACHE_HOME": str(self.cache_root),
        }

        for env_key, cache_path in cache_mappings.items():
            os.environ[env_key] = cache_path
            p = pathlib.Path(cache_path)
            try:
                p.mkdir(parents=True, exist_ok=True)
            except (PermissionError, FileExistsError):
                if p.exists() and p.is_dir():
                    continue
                logger.warning(
                    "Could not ensure directory exists for %s: %s", env_key, cache_path
                )

    def _create_directories(self) -> None:
        """Create application-specific directories"""
        models_root = self.models_root
        datasets_root = self.datasets_root
        outputs_root = self.outputs_root
        rag_root = self.rag_root
        worldpacks_root = self.worldpacks_root

        self.app_dirs = {
            # Model Registry file
            "TRAIN_REGISTRY": str(outputs_root / "training" / "model_registry.json"),
            # Cache dirs
            "CACHE_DIR": str(self.cache_root),
            "CACHE_HF": str(self.hf_home),
            "CACHE_TORCH": str(self.torch_home),
            # Models
            "MODELS_SD": str(models_root / "stable-diffusion"),
            "MODELS_SDXL": str(models_root / "stable-diffusion" / "xl"),
            "MODELS_CONTROLNET": str(models_root / "controlnet"),
            "MODELS_LORA": str(models_root / "lora"),
            "MODELS_LORA_SDXL": str(models_root / "lora_sdxl"),
            "MODELS_IPADAPTER": str(models_root / "ipadapter"),
            "MODELS_LLM": str(models_root / "llm"),
            "MODELS_LLM_LORA": str(models_root / "llm" / "lora"),
            "MODELS_VLM": str(models_root / "vlm"),
            "MODELS_TEXT2IMAGE": str(models_root / "stable-diffusion"),
            "MODELS_TEXT2VIDEO": str(models_root / "video"),
            "MODELS_TTS": str(models_root / "audio"),
            "MODELS_ENHANCEMENT": str(models_root / "enhancement"),
            "MODELS_TAGGING": str(models_root / "tagging"),
            "MODELS_EMBEDDING": str(models_root / "embeddings"),
            "MODELS_RERANKER": str(models_root / "reranker"),
            "MODELS_SAFETY": str(models_root / "safety"),
            "MODELS_AUDIO": str(models_root / "audio"),
            "MODELS_CLIP": str(models_root / "clip"),
            # Datasets
            "DATASETS_RAW": str(datasets_root / "raw"),
            "DATASETS_PROCESSED": str(datasets_root / "processed"),
            "DATASETS_METADATA": str(datasets_root / "metadata"),
            # Outputs
            "OUTPUT_DIR": str(outputs_root / "outputs"),
            "OUTPUT_BATCH": str(outputs_root / "batch"),
            "OUTPUT_TRAINING": str(outputs_root / "training"),
            "OUTPUT_GAMES": str(outputs_root / "games"),
            "OUTPUT_LORA": str(outputs_root / "lora"),
            "OUTPUT_RAG": str(outputs_root / "rag"),
            "OUTPUT_EXPORTS": str(outputs_root / "exports"),
            "LOGS_DIR": str(outputs_root / "logs"),
            # RAG
            "RAG_INDEX": str(rag_root / "indexes"),
            "RAG_DOCS": str(rag_root / "documents"),
            "RAG_EMBEDDINGS": str(rag_root / "embeddings"),
            "RAG_VECTOR_STORE": str(rag_root / "knowledge_base" / "vector_store"),
            "WORLDPACKS": str(worldpacks_root),
        }

        for dir_path in self.app_dirs.values():
            path_obj = pathlib.Path(dir_path)
            target_dir = path_obj.parent if path_obj.suffix else path_obj
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
            except (PermissionError, FileExistsError):
                if target_dir.exists() and target_dir.is_dir():
                    continue
                logger.warning("Could not ensure directory exists: %s", target_dir)

    def get_path(self, key: str) -> str:
        """Get directory path by key"""
        if key not in self.app_dirs:
            raise KeyError(f"Unknown cache directory: {key}")
        return self.app_dirs[key]

    def get_model_path(self, model_type: str, model_name: str) -> Path:
        """Get standardized model path"""
        return Path(self.models_root) / model_type / model_name

    def get_dataset_path(self, dataset_type: str, dataset_name: str) -> Path:
        """Get standardized dataset path"""
        return Path(self.datasets_root) / dataset_type / dataset_name

    def get_output_path(self, output_type: str = "multi-modal-lab") -> Path:
        """Get output directory path"""
        path = Path(self.outputs_root) / output_type
        path.mkdir(parents=True, exist_ok=True)
        return path

    def cache_model_info(self, model_key: str, model_info: Dict[str, Any]) -> None:
        """Cache model information with timestamp"""
        cache_info = {
            **model_info,
            "cached_at": datetime.now().isoformat(),
            "cache_key": model_key,
        }

        info_file = (
            Path(self.outputs_root)
            / "metadata"
            / "models"
            / f"{model_key}_info.json"
        )
        info_file.parent.mkdir(parents=True, exist_ok=True)

        with open(info_file, "w", encoding="utf-8") as f:
            json.dump(cache_info, f, indent=2, ensure_ascii=False)

    def get_model_info(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached model information"""
        info_file = (
            Path(self.outputs_root)
            / "metadata"
            / "models"
            / f"{model_key}_info.json"
        )

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
            "models_root": str(self.models_root),
            "outputs_root": str(self.outputs_root),
            "datasets_root": str(self.datasets_root),
            "disk_usage": {},
            "memory_cache_size": len(self._memory_cache),
            "gpu_available": torch.cuda.is_available(),
        }

        # Calculate disk usage
        for name, path in [
            ("cache", Path(self.cache_root)),
            ("models", Path(self.models_root)),
            ("datasets", Path(self.datasets_root)),
            ("outputs", Path(self.outputs_root)),
            ("rag", Path(self.rag_root)),
        ]:
            try:
                if path.exists():
                    total_size = sum(
                        f.stat().st_size for f in path.rglob("*") if f.is_file()
                    )
                    stats["disk_usage"][name] = {
                        "size_gb": round(total_size / (1024**3), 2),
                        "files_count": len(list(path.rglob("*"))),
                    }
            except Exception:
                # Avoid crashing in restricted environments (permissions / missing mounts)
                continue

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
