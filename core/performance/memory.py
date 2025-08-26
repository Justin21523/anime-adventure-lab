# core/performance/memory.py
"""Memory and resource management"""
import gc
import psutil
from typing import Dict, Any, Optional
import torch


class MemoryManager:
    """GPU and system memory management"""

    def __init__(self):
        self.loaded_models = {}
        self.memory_usage = {}
        self._update_memory_info()

    def _update_memory_info(self):
        """Update current memory usage info"""
        try:
            # System memory
            system_memory = psutil.virtual_memory()
            self.memory_usage["system"] = {
                "total_gb": round(system_memory.total / (1024**3), 2),
                "available_gb": round(system_memory.available / (1024**3), 2),
                "percent": system_memory.percent,
            }

            # GPU memory
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0)
                self.memory_usage["gpu"] = {
                    "total_gb": round(gpu_memory.total_memory / (1024**3), 2),
                    "allocated_gb": round(
                        torch.cuda.memory_allocated(0) / (1024**3), 2
                    ),
                    "reserved_gb": round(torch.cuda.memory_reserved(0) / (1024**3), 2),
                    "percent": round(
                        (torch.cuda.memory_allocated(0) / gpu_memory.total_memory)
                        * 100,
                        1,
                    ),
                }
            else:
                self.memory_usage["gpu"] = {"available": False}

        except Exception as e:
            self.memory_usage["error"] = str(e)

    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory information"""
        self._update_memory_info()
        return self.memory_usage.copy()

    def unload_model(self, key: str) -> None:
        """Unload a specific model from memory"""
        if key in self.loaded_models:
            del self.loaded_models[key]
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def cleanup_all(self) -> None:
        """Clean up all loaded models and free memory"""
        self.loaded_models.clear()
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def register_model(self, key: str, model: Any) -> None:
        """Register a loaded model"""
        self.loaded_models[key] = {
            "model": model,
            "loaded_at": "2024-01-01T00:00:00",  # Mock timestamp
        }

    def get_loaded_models(self) -> Dict[str, Any]:
        """Get list of currently loaded models"""
        return {k: {"loaded_at": v["loaded_at"]} for k, v in self.loaded_models.items()}
