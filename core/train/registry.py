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

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model information"""
        return self.registry["models"].get(model_id)

    def list_models(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all registered models"""
        models = list(self.registry["models"].values())

        if model_type:
            models = [m for m in models if m.get("model_type") == model_type]

        return models
