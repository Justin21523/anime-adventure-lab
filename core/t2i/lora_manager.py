# core/t2i/lora_manager.py
import os
import json
from pathlib import Path
from typing import Dict, List, Optional
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

    def list_loras(self) -> List[dict]:
        """List all available LoRAs"""
        return list(self.loaded_loras.values())

    def load_lora(self, pipeline, lora_id: str, weight: float = 1.0) -> bool:
        """Load LoRA adapter to pipeline"""
        if lora_id not in self.loaded_loras:
            raise ValueError(f"LoRA {lora_id} not found")

        lora_path = self.loaded_loras[lora_id]["path"]

        try:
            pipeline.load_lora_weights(lora_path)
            pipeline.fuse_lora(lora_scale=weight)
            self.loaded_loras[lora_id]["loaded"] = True
            return True
        except Exception as e:
            print(f"Failed to load LoRA {lora_id}: {e}")
            return False

    def unload_loras(self, pipeline):
        """Unload all LoRAs from pipeline"""
        try:
            pipeline.unfuse_lora()
            for lora_id in self.loaded_loras:
                self.loaded_loras[lora_id]["loaded"] = False
        except:
            pass
