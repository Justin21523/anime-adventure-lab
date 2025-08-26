# core/train/registry.py
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from core.shared_cache import get_shared_cache


class ModelRegistry:
    """Keep track of trained models and their metadata"""

    def __init__(self):
        self.cache = get_shared_cache()
        self.registry_path = Path(self.cache.get_path("TRAIN_REGISTRY"))
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        """Load model registry from disk"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Failed to load registry: {e}")

        return {"models": {}, "version": "1.0"}

    def _save_registry(self):
        """Save registry to disk"""
        try:
            with open(self.registry_path, "w") as f:
                json.dump(self.registry, f, indent=2)
        except Exception as e:
            print(f"Failed to save registry: {e}")

    def register_model(
        self,
        model_id: str,
        model_type: str,  # lora, dreambooth, etc.
        model_path: str,
        metadata: Dict[str, Any],
    ):
        """Register a new trained model"""
        model_info = {
            "model_id": model_id,
            "model_type": model_type,
            "model_path": model_path,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata,
        }

        self.registry["models"][model_id] = model_info
        self._save_registry()

        print(f"Registered model: {model_id}")

    def add(self, model_id: str, info: Dict[str, Any]) -> None:
        """Add model to registry"""
        try:
            self._registry[model_id] = {
                **info,
                "registered_at": "2024-01-01T00:00:00",  # Mock timestamp
            }
            self._save_registry()
        except Exception as e:
            raise RuntimeError(f"Failed to add model {model_id}: {str(e)}")

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model information"""
        return self.registry["models"].get(model_id)

    def list_models(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all registered models"""
        models = list(self.registry["models"].values())

        if model_type:
            models = [m for m in models if m.get("model_type") == model_type]

        return models

    def update(self, model_id: str, patch: Dict[str, Any]) -> None:
        """Update model information"""
        try:
            if model_id not in self._registry:
                raise KeyError(f"Model {model_id} not found in registry")

            self._registry[model_id].update(patch)
            self._save_registry()

        except Exception as e:
            raise RuntimeError(f"Failed to update model {model_id}: {str(e)}")

    def remove(self, model_id: str) -> bool:
        """Remove model from registry"""
        try:
            if model_id in self._registry:
                del self._registry[model_id]
                self._save_registry()
                return True
            return False
        except Exception as e:
            print(f"Failed to remove model {model_id}: {e}")
            return False
