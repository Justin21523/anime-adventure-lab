# core/t2i/lora_manager.py
import os
import json
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any
from peft import LoraConfig, get_peft_model
import logging
import time
from safetensors.torch import load_file
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from core.shared_cache import get_shared_cache

logger = logging.getLogger(__name__)


class LoRAManager:
    """Manage LoRA loading and unloading"""

    def __init__(self, cache_root: str):
        self.cache = get_shared_cache()
        self.cache_root = Path(cache_root)
        self.loaded_loras: Dict[str, dict] = {}
        self.lora_cache_dir = Path(self.cache.get_path("MODELS_LORA"))
        self._scan_loras()
        self._registry_path = Path(self.cache.get_path("MODELS_LORA")) / "registry.json"
        self._load_registry()

        self.loaded_loras = {}
        self.lora_metadata = {}
        self.max_concurrent_loras = 3
        self._load_lora_metadata()
        logger.info("LoRAManager initialized")

    def _scan_loras(self):
        """Scan for available LoRA models"""
        lora_path = Path(self.cache.get_path("MODELS_LORA"))
        if not lora_path.exists():
            return

        for folder in lora_path.iterdir():
            if folder.is_dir():
                model_card = folder / "MODEL_CARD.md"
                adapter_file = folder / "adapter_model.safetensors"

                if adapter_file.exists():
                    # Basic LoRA info
                    info = {
                        "id": folder.name,
                        "path": str(folder),
                        "model_type": "sd15",  # default
                        "rank": 16,  # default
                        "loaded": False,
                    }

                    # Try to parse metadata
                    if model_card.exists():
                        try:
                            content = model_card.read_text()
                            if "sdxl" in content.lower():
                                info["model_type"] = "sdxl"
                        except:
                            pass

                    self.loaded_loras[folder.name] = info

    def _load_registry(self):
        """Load LoRA registry"""
        if self._registry_path.exists():
            with open(self._registry_path, "r") as f:
                self._registry = json.load(f)
        else:
            self._registry = {}

    def _save_registry(self):
        """Save LoRA registry"""
        with open(self._registry_path, "w") as f:
            json.dump(self._registry, f, indent=2)

    def _load_lora_metadata(self):
        """Load LoRA metadata from cache"""
        metadata_file = self.lora_cache_dir / "lora_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                self.lora_metadata = json.load(f)

    def _save_lora_metadata(self):
        """Save LoRA metadata to cache"""
        metadata_file = self.lora_cache_dir / "lora_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(self.lora_metadata, f, indent=2)

    def list_available_loras(self) -> List[Dict]:
        """List all available LoRA adapters"""
        loras = []

        for lora_dir in self.lora_cache_dir.iterdir():
            if not lora_dir.is_dir():
                continue

            lora_id = lora_dir.name
            lora_info = self._get_lora_info(lora_id)

            if lora_info:
                loras.append(
                    {
                        "lora_id": lora_id,
                        "name": lora_info.get("name", lora_id),
                        "description": lora_info.get("description", ""),
                        "tags": lora_info.get("tags", []),
                        "model_compatibility": lora_info.get("compatible_models", []),
                        "file_size_mb": lora_info.get("file_size_mb", 0),
                        "created_date": lora_info.get("created_date", ""),
                        "loaded": lora_id in self.loaded_loras,
                        "load_count": lora_info.get("load_count", 0),
                    }
                )

        return sorted(loras, key=lambda x: x["name"])

    def _get_lora_info(self, lora_id: str) -> Optional[Dict]:
        """Get LoRA information from metadata or files"""
        if lora_id in self.lora_metadata:
            return self.lora_metadata[lora_id]

        lora_path = self.lora_cache_dir / lora_id
        if not lora_path.exists():
            return None

        # Try to extract info from files
        info = {
            "name": lora_id,
            "description": "",
            "compatible_models": ["sd15", "sdxl"],  # Default assumption
            "tags": [],
            "load_count": 0,
        }

        # Check for info.json
        info_file = lora_path / "info.json"
        if info_file.exists():
            with open(info_file, "r") as f:
                file_info = json.load(f)
                info.update(file_info)

        # Calculate file size
        total_size = 0
        for file_path in lora_path.rglob("*.safetensors"):
            total_size += file_path.stat().st_size
        info["file_size_mb"] = round(total_size / (1024 * 1024), 1)

        # Cache the info
        self.lora_metadata[lora_id] = info
        self._save_lora_metadata()

        return info

    def load_lora(self, pipeline, lora_id: str, weight: float = 1.0) -> bool:
        """Load LoRA adapter with proper diffusers integration"""
        try:
            lora_path = self.lora_cache_dir / lora_id

            if not lora_path.exists():
                raise FileNotFoundError(f"LoRA not found: {lora_id}")

            # Find safetensors file
            safetensor_files = list(lora_path.glob("*.safetensors"))
            if not safetensor_files:
                raise FileNotFoundError(f"No .safetensors file found in {lora_path}")

            lora_file = safetensor_files[0]  # Use first safetensors file

            # Load LoRA weights using diffusers
            pipeline.load_lora_weights(str(lora_file))

            # Fuse LoRA with specified weight
            if hasattr(pipeline, "fuse_lora"):
                pipeline.fuse_lora(lora_scale=weight)

            # Track loaded LoRA
            self.loaded_loras[lora_id] = {
                "path": str(lora_file),
                "weight": weight,
                "loaded_at": time.time(),
            }

            # Update load count
            if lora_id in self.lora_metadata:
                self.lora_metadata[lora_id]["load_count"] = (
                    self.lora_metadata[lora_id].get("load_count", 0) + 1
                )
                self._save_lora_metadata()

            logger.info(f"LoRA {lora_id} loaded with weight {weight}")
            return True

        except Exception as e:
            logger.error(f"Failed to load LoRA {lora_id}: {e}")
            return False

    def unload_lora(self, pipeline, lora_id: str) -> bool:
        """Unload specific LoRA"""
        try:
            if lora_id not in self.loaded_loras:
                logger.warning(f"LoRA {lora_id} is not loaded")
                return False

            # Unfuse LoRA
            if hasattr(pipeline, "unfuse_lora"):
                pipeline.unfuse_lora()

            # Remove from loaded tracking
            del self.loaded_loras[lora_id]

            logger.info(f"LoRA {lora_id} unloaded")
            return True

        except Exception as e:
            logger.error(f"Failed to unload LoRA {lora_id}: {e}")
            return False

    def unload_all_loras(self, pipeline) -> bool:
        """Unload all LoRAs from pipeline"""
        try:
            if not self.loaded_loras:
                return True

            # Unfuse all LoRAs
            if hasattr(pipeline, "unfuse_lora"):
                pipeline.unfuse_lora()

            # Unload LoRA weights
            if hasattr(pipeline, "unload_lora_weights"):
                pipeline.unload_lora_weights()

            # Clear tracking
            loaded_count = len(self.loaded_loras)
            self.loaded_loras.clear()

            logger.info(f"All LoRAs ({loaded_count}) unloaded")
            return True

        except Exception as e:
            logger.error(f"Failed to unload all LoRAs: {e}")
            return False

    def list_loras(self) -> List[dict]:
        """List available LoRAs"""
        loras = []
        lora_dir = Path(self.cache.get_path("MODELS_LORA"))

        # Scan for LoRA directories
        for lora_path in lora_dir.iterdir():
            if lora_path.is_dir():
                lora_info = self._registry.get(
                    lora_path.name,
                    {
                        "id": lora_path.name,
                        "name": lora_path.name,
                        "path": str(lora_path),
                        "loaded": lora_path.name in self.loaded_loras,
                    },
                )
                loras.append(lora_info)

        return loras

    def get_loaded(self) -> Dict[str, Any]:
        """Get currently loaded LoRAs"""
        return self.loaded_loras.copy()

    def get_lora_info(self, lora_id: str) -> Optional[Dict]:
        """Get detailed information about a specific LoRA"""
        return self._get_lora_info(lora_id)

    def validate_lora_compatibility(self, lora_id: str, model_type: str) -> bool:
        """Check if LoRA is compatible with model type"""
        lora_info = self._get_lora_info(lora_id)
        if not lora_info:
            return False

        compatible_models = lora_info.get("compatible_models", [])
        return model_type.lower() in [m.lower() for m in compatible_models]

    def get_recommended_weight(self, lora_id: str) -> float:
        """Get recommended weight for LoRA"""
        lora_info = self._get_lora_info(lora_id)
        if lora_info:
            return lora_info.get("recommended_weight", 1.0)
        return 1.0
