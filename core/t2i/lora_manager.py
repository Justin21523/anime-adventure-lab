# core/t2i/lora_manager.py
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from peft import LoraConfig, get_peft_model

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from core.shared_cache import get_shared_cache


class LoRAManager:
    """Manage LoRA loading and unloading"""

    def __init__(self):
        self.cache = get_shared_cache()
        self.loaded_loras: Dict[str, dict] = {}
        self._scan_loras()
        self._registry_path = Path(self.cache.get_path("MODELS_LORA")) / "registry.json"
        self._load_registry()

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

    def load_lora(self, pipeline, lora_id: str, weight: float = 1.0) -> bool:
        """Load LoRA adapter to pipeline"""
        try:
            lora_path = Path(self.cache.get_path("MODELS_LORA")) / lora_id

            if not lora_path.exists():
                raise FileNotFoundError(f"LoRA not found: {lora_id}")

            # Mock LoRA loading
            # In real implementation:
            # from peft import PeftModel
            # self.loaded_loras[lora_id] = PeftModel.from_pretrained(...)

            self.loaded_loras[lora_id] = {
                "path": str(lora_path),
                "weight": weight or 1.0,
                "loaded_at": "2024-01-01T00:00:00",
            }

            return True

        except Exception as e:
            raise RuntimeError(f"Failed to load LoRA {lora_id}: {str(e)}")

    def unload_loras(self, pipeline):
        """Unload all LoRAs from pipeline"""
        try:
            pipeline.unfuse_lora()
            for lora_id in self.loaded_loras:
                self.loaded_loras[lora_id]["loaded"] = False
        except:
            pass

    def get_loaded(self) -> Dict[str, Any]:
        """Get currently loaded LoRAs"""
        return self.loaded_loras.copy()
