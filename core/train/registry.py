# core/train/registry.py
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..exceptions import RegistryError, ValidationError
from core.shared_cache import get_shared_cache


class ModelRegistry:
    """Keep track of trained models and their metadata"""

    def __init__(self, cache_root: Optional[str] = None):
        self.cache = get_shared_cache()
        self.cache_root = self.cache.get_path("TRAIN_REGISTRY")
        self.registry_path = Path(self.cache.get_path("TRAIN_REGISTRY"))
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        self._registry: Dict[str, Any] = {"models": {}, "version": "1.0"}

        # Ensure directory exists
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        # Load existing registry - this will update self._registry
        self._load_registry()

    def _load_registry(self) -> None:
        """Load model registry from disk"""
        try:
            if self.registry_path.exists():
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    loaded_data = json.load(f)
                    # Validate loaded data structure
                    if isinstance(loaded_data, dict) and "models" in loaded_data:
                        self._registry = loaded_data
                    else:
                        print(f"Invalid registry format, using default")
                        self._registry = {"models": {}, "version": "1.0"}
            else:
                # Initialize with default structure and save
                self._registry = {"models": {}, "version": "1.0"}
                self._save_registry()
        except Exception as e:
            print(f"Failed to load registry: {e}")
            self._registry = {"models": {}, "version": "1.0"}

    def _save_registry(self):
        """Save registry to disk"""
        try:
            with open(self.registry_path, "w") as f:
                json.dump(self._registry, f, indent=2)
        except Exception as e:
            print(f"Failed to save registry: {e}")

    @property
    def registry(self) -> Dict[str, Any]:
        """Access to registry data (backward compatibility with existing code)"""
        return self._registry

    def register_model(
        self,
        model_id: str,
        model_type: str,  # lora, dreambooth, etc.
        base_model: str,
        model_path: str,
        config: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """Register a trained model"""
        try:
            if not model_id or not model_type or not base_model:
                raise ValidationError(
                    "model_id, model_type, and base_model are required"
                )

            model_info = {
                "model_id": model_id,
                "model_type": model_type,
                "base_model": base_model,
                "model_path": str(model_path),
                "config": config,
                "metrics": metrics or {},
                "tags": tags or [],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }

            self._registry["models"][model_id] = model_info
            self._save_registry()

            print(f"Registered model: {model_id}")
            return True

        except Exception as e:
            raise RegistryError(f"Failed to register model {model_id}: {e}")

    def add(self, model_id: str, info: Dict[str, Any]) -> None:
        """Add model to registry (alternative interface)"""
        try:
            self._registry["models"][model_id] = {
                **info,
                "registered_at": datetime.now().isoformat(),
            }
            self._save_registry()
        except Exception as e:
            raise RuntimeError(f"Failed to add model {model_id}: {str(e)}")

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model information"""
        return self._registry["models"].get(model_id)

    def list_models(
        self,
        model_type: Optional[str] = None,
        base_model: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """List models with optional filtering"""
        models = list(self._registry["models"].values())

        if model_type:
            models = [m for m in models if m.get("model_type") == model_type]

        if base_model:
            models = [m for m in models if m.get("base_model") == base_model]

        if tags:
            models = [
                m for m in models if any(tag in m.get("tags", []) for tag in tags)
            ]

        return models

    def update(self, model_id: str, patch: Dict[str, Any]) -> None:
        """Update model information"""
        try:
            if model_id not in self._registry["models"]:
                raise KeyError(f"Model {model_id} not found in registry")

            model_info = self._registry["models"][model_id]
            model_info.update(patch)
            model_info["updated_at"] = datetime.now().isoformat()

            self._save_registry()

        except Exception as e:
            raise RuntimeError(f"Failed to update model {model_id}: {str(e)}")

    def update_model(self, model_id: str, updates: Dict[str, Any]) -> bool:
        """Update model information (alternative interface)"""
        try:
            self.update(model_id, updates)
            return True
        except Exception as e:
            raise RegistryError(f"Failed to update model {model_id}: {e}")

    def remove(self, model_id: str) -> bool:
        """Remove model from registry"""
        try:
            if model_id in self._registry["models"]:
                del self._registry["models"][model_id]
                self._save_registry()
                return True
            return False
        except Exception as e:
            print(f"Failed to remove model {model_id}: {e}")
            return False

    def delete_model(self, model_id: str) -> bool:
        """Delete model from registry (alternative interface)"""
        return self.remove(model_id)

    def search_models(self, query: str) -> List[Dict[str, Any]]:
        """Search models by name, tags, or base model"""
        query = query.lower()
        results = []

        for model_info in self._registry["models"].values():
            # Search in model_id
            if query in model_info.get("model_id", "").lower():
                results.append(model_info)
                continue

            # Search in tags
            if any(query in tag.lower() for tag in model_info.get("tags", [])):
                results.append(model_info)
                continue

            # Search in base_model
            if query in model_info.get("base_model", "").lower():
                results.append(model_info)
                continue

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        models = self._registry["models"]
        model_types = {}
        base_models = {}

        for model_info in models.values():
            model_type = model_info.get("model_type", "unknown")
            base_model = model_info.get("base_model", "unknown")

            model_types[model_type] = model_types.get(model_type, 0) + 1
            base_models[base_model] = base_models.get(base_model, 0) + 1

        return {
            "total_models": len(models),
            "model_types": model_types,
            "base_models": base_models,
            "registry_path": str(self.registry_path),
        }

    def export_registry(self, export_path: Path) -> bool:
        """Export registry to file"""
        try:
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)

            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(self._registry, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            raise RegistryError(f"Failed to export registry: {e}")

    def import_registry(self, import_path: Path, merge: bool = True) -> bool:
        """Import registry from file"""
        try:
            import_path = Path(import_path)
            if not import_path.exists():
                raise RegistryError(f"Import file not found: {import_path}")

            with open(import_path, "r", encoding="utf-8") as f:
                imported_data = json.load(f)

            if not isinstance(imported_data, dict) or "models" not in imported_data:
                raise RegistryError("Invalid registry format")

            if merge:
                # Merge with existing registry
                self._registry["models"].update(imported_data["models"])
            else:
                # Replace existing registry
                self._registry = imported_data

            self._save_registry()
            return True

        except Exception as e:
            raise RegistryError(f"Failed to import registry: {e}")


# Global registry instance
_global_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Get global model registry instance"""
    global _global_registry
    if _global_registry is None:
        _global_registry = ModelRegistry()
    return _global_registry


# Backward compatibility aliases
def get_registry() -> ModelRegistry:
    """Backward compatibility alias"""
    return get_model_registry()
