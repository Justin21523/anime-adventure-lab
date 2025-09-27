# core/t2i/model_config.py
"""Model configuration manager - REQUIRED FILE"""

import json
import logging
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ModelConfigManager:
    """Model configuration and metadata manager"""

    def __init__(self, cache_root: str):
        self.cache_root = Path(cache_root)
        self.models_dir = self.cache_root / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.models_dir / "model_registry.json"
        self.models = {}
        self.loaded_model = None
        self._initialize_default_models()

    def _initialize_default_models(self):
        """Initialize with default model configurations"""
        self.models = {
            "runwayml/stable-diffusion-v1-5": {
                "name": "Stable Diffusion 1.5",
                "type": "sd15",
                "vram_gb": 4.0,
                "loaded": False,
            },
            "stabilityai/stable-diffusion-xl-base-1.0": {
                "name": "Stable Diffusion XL",
                "type": "sdxl",
                "vram_gb": 8.0,
                "loaded": False,
            },
        }

    def list_available_models(self) -> List[Dict]:
        """List all available models"""
        return [
            {
                "model_id": model_id,
                "name": info["name"],
                "type": info["type"],
                "vram_requirement_gb": info["vram_gb"],
                "loaded": info["loaded"],
            }
            for model_id, info in self.models.items()
        ]

    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """Get model info by ID"""
        if model_id in self.models:
            info = self.models[model_id].copy()
            info["model_id"] = model_id
            return info
        return None

    def set_model_loaded(self, model_id: str, loaded: bool = True):
        """Set model loaded status"""
        # First unload all others
        for info in self.models.values():
            info["loaded"] = False

        # Set target model as loaded
        if model_id in self.models:
            self.models[model_id]["loaded"] = loaded
            self.loaded_model = model_id if loaded else None
