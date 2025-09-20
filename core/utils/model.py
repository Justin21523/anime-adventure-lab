# core/utils/model.py
"""
Model Management Utilities
Model loading, caching, and memory management
"""

import gc
import logging
import torch
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import hashlib
import json
import time
from contextlib import contextmanager

from ..config import get_config
from ..exceptions import ModelError

logger = logging.getLogger(__name__)


class ModelManager:
    """Centralized model management and caching"""

    def __init__(self):
        self.config = get_config()
        self.loaded_models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        self.memory_tracker: Dict[str, float] = {}

    def get_model_key(self, model_id: str, **kwargs) -> str:
        """Generate unique key for model configuration"""
        config_str = json.dumps(sorted(kwargs.items()), sort_keys=True)
        key_str = f"{model_id}:{config_str}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def estimate_model_memory(self, model_id: str) -> float:
        """Estimate model memory usage in MB"""
        # Rough estimates based on common models
        size_estimates = {
            "stabilityai/stable-diffusion-2-1": 5000,
            "runwayml/stable-diffusion-v1-5": 4000,
            "stabilityai/stable-diffusion-xl-base-1.0": 7000,
            "Salesforce/blip2-opt-2.7b": 3000,
            "liuhaotian/llava-v1.5-7b": 14000,
            "Qwen/Qwen-VL-Chat": 8000,
            "BAAI/bge-base-zh-v1.5": 500,
            "BAAI/bge-m3": 600,
            "sentence-transformers/all-MiniLM-L6-v2": 100,
        }

        # Check for exact match
        if model_id in size_estimates:
            return size_estimates[model_id]

        # Check for partial matches
        for known_model, size in size_estimates.items():
            if known_model.split("/")[-1] in model_id:
                return size

        # Default estimate based on model type keywords
        model_lower = model_id.lower()
        if "xl" in model_lower or "large" in model_lower:
            return 8000
        elif "base" in model_lower or "medium" in model_lower:
            return 4000
        elif "small" in model_lower or "mini" in model_lower:
            return 1000
        else:
            return 3000  # Default estimate

    def check_memory_availability(self, required_mb: float) -> bool:
        """Check if enough memory is available"""
        if not torch.cuda.is_available():
            return True  # Assume CPU has enough memory

        # Get available GPU memory
        free_memory, total_memory = torch.cuda.mem_get_info()
        free_mb = free_memory / 1024**2

        # Keep some buffer (20% or 2GB, whichever is smaller)
        buffer_mb = min(total_memory * 0.2 / 1024**2, 2000)
        available_mb = free_mb - buffer_mb

        return available_mb >= required_mb

    def unload_model(self, model_key: str) -> bool:
        """Unload model from memory"""
        try:
            if model_key in self.loaded_models:
                model = self.loaded_models.pop(model_key)

                # Move to CPU and clear CUDA cache
                if hasattr(model, "to"):
                    model.to("cpu")
                elif hasattr(model, "device"):
                    for component in model.components.values():
                        if hasattr(component, "to"):
                            component.to("cpu")

                # Clear references
                del model

                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Update memory tracker
                if model_key in self.memory_tracker:
                    freed_mb = self.memory_tracker.pop(model_key)
                    logger.info(
                        f"🗑️ Unloaded model {model_key}, freed ~{freed_mb:.0f}MB"
                    )

                return True

            return False

        except Exception as e:
            logger.error(f"❌ Failed to unload model {model_key}: {e}")
            return False

    def auto_unload_models(self, required_mb: float) -> bool:
        """Automatically unload models to free memory"""
        if self.check_memory_availability(required_mb):
            return True

        # Sort models by last access time (oldest first)
        models_by_age = sorted(
            self.model_metadata.items(), key=lambda x: x[1].get("last_access", 0)
        )

        freed_mb = 0
        for model_key, metadata in models_by_age:
            if model_key in self.loaded_models:
                model_size = self.memory_tracker.get(model_key, 0)
                self.unload_model(model_key)
                freed_mb += model_size

                # Check if we have enough memory now
                if self.check_memory_availability(required_mb):
                    logger.info(f"✅ Auto-unloaded models, freed {freed_mb:.0f}MB")
                    return True

        return self.check_memory_availability(required_mb)

    def load_model(
        self, model_id: str, model_class, force_reload: bool = False, **kwargs
    ) -> Any:
        """Load model with caching and memory management"""
        model_key = self.get_model_key(model_id, **kwargs)

        # Return cached model if available
        if not force_reload and model_key in self.loaded_models:
            self.model_metadata[model_key]["last_access"] = time.time()
            self.model_metadata[model_key]["access_count"] += 1
            logger.info(f"📋 Using cached model: {model_key}")
            return self.loaded_models[model_key]

        # Estimate memory requirements
        estimated_mb = self.estimate_model_memory(model_id)

        # Auto-unload models if needed
        if not self.auto_unload_models(estimated_mb):
            raise ModelError(
                f"Insufficient memory to load {model_id} (~{estimated_mb}MB required)"
            )

        try:
            logger.info(f"🔄 Loading model: {model_id}")
            start_time = time.time()

            # Record memory before loading
            if torch.cuda.is_available():
                memory_before = torch.cuda.memory_allocated() / 1024**2
            else:
                memory_before = 0

            # Load the model
            model = model_class.from_pretrained(model_id, **kwargs)

            # Record memory after loading
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated() / 1024**2
                actual_memory = memory_after - memory_before
            else:
                actual_memory = estimated_mb  # Fallback estimate

            load_time = time.time() - start_time

            # Cache the model
            self.loaded_models[model_key] = model
            self.memory_tracker[model_key] = actual_memory
            self.model_metadata[model_key] = {
                "model_id": model_id,
                "load_time": load_time,
                "memory_mb": actual_memory,
                "last_access": time.time(),
                "access_count": 1,
                "kwargs": kwargs,
            }

            logger.info(
                f"✅ Model loaded: {model_id} ({actual_memory:.0f}MB, {load_time:.1f}s)"
            )
            return model

        except Exception as e:
            logger.error(f"❌ Failed to load model {model_id}: {e}")
            raise ModelError(f"Failed to load model {model_id}: {e}")

    @contextmanager
    def temporary_model(self, model_id: str, model_class, **kwargs):
        """Context manager for temporary model usage"""
        model_key = self.get_model_key(model_id, **kwargs)
        was_cached = model_key in self.loaded_models

        try:
            model = self.load_model(model_id, model_class, **kwargs)
            yield model
        finally:
            # Only unload if it wasn't previously cached
            if not was_cached and model_key in self.loaded_models:
                self.unload_model(model_key)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        total_memory_mb = sum(self.memory_tracker.values())

        if torch.cuda.is_available():
            allocated_mb = torch.cuda.memory_allocated() / 1024**2
            cached_mb = torch.cuda.memory_reserved() / 1024**2
            free_mb, total_mb = torch.cuda.mem_get_info()
            free_mb = free_mb / 1024**2
            total_mb = total_mb / 1024**2
        else:
            allocated_mb = cached_mb = free_mb = total_mb = 0

        return {
            "tracked_models": len(self.loaded_models),
            "tracked_memory_mb": total_memory_mb,
            "gpu_allocated_mb": allocated_mb,
            "gpu_cached_mb": cached_mb,
            "gpu_free_mb": free_mb,
            "gpu_total_mb": total_mb,
            "models": {
                key: {
                    "model_id": meta["model_id"],
                    "memory_mb": self.memory_tracker.get(key, 0),
                    "access_count": meta["access_count"],
                    "last_access": meta["last_access"],
                }
                for key, meta in self.model_metadata.items()
            },
        }

    def cleanup_all(self):
        """Unload all models and cleanup"""
        logger.info("🧹 Cleaning up all models")

        model_keys = list(self.loaded_models.keys())
        for model_key in model_keys:
            self.unload_model(model_key)

        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_model_manager = None


def get_model_manager() -> ModelManager:
    """Get global model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
