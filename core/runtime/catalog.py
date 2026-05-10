from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.config import get_config

logger = logging.getLogger(__name__)


def load_runtime_preset_catalog() -> Dict[str, Any]:
    """
    Load runtime presets from `configs/runtime_presets.yaml`.

    Structure:
      default_preset_id: str
      presets:
        <preset_id>:
          name: str
          description: str
          llm: {...}
          t2i: {...}
    """
    cfg = get_config()
    raw = cfg.load_yaml("runtime_presets.yaml") or {}

    default_preset_id = str(raw.get("default_preset_id") or "").strip()
    if not default_preset_id:
        default_preset_id = "rtx_5080_16gb"

    presets_raw = raw.get("presets") or {}
    presets: List[Dict[str, Any]] = []
    if isinstance(presets_raw, dict):
        for preset_id, payload in presets_raw.items():
            if not isinstance(payload, dict):
                continue
            presets.append({"preset_id": str(preset_id), **payload})
    else:
        logger.warning("runtime_presets.yaml: presets is not a dict")

    if not presets:
        # Minimal safe fallback (keeps UI from breaking)
        presets = [
            {
                "preset_id": default_preset_id,
                "name": "Default",
                "description": "Fallback runtime preset",
                "llm": {
                    "model_name": "Qwen/Qwen2.5-7B-Instruct",
                    "torch_dtype": "float16",
                    "device_map": "auto",
                    "use_quantization": True,
                    "quantization_bits": 4,
                },
                "t2i": {
                    "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
                    "torch_dtype": "float16",
                    "enable_attention_slicing": True,
                    "enable_vae_slicing": True,
                    "enable_vae_tiling": True,
                    "enable_cpu_offload": False,
                    "enable_sequential_cpu_offload": False,
                    "default_width": 1024,
                    "default_height": 1024,
                    "default_steps": 30,
                    "default_guidance_scale": 6.0,
                    "max_width": 1024,
                    "max_height": 1024,
                    "max_steps": 50,
                },
            }
        ]

    return {"default_preset_id": default_preset_id, "presets": presets}


def get_runtime_preset(preset_id: str) -> Optional[Dict[str, Any]]:
    pid = str(preset_id or "").strip()
    if not pid:
        return None
    catalog = load_runtime_preset_catalog()
    for item in catalog.get("presets") or []:
        if str(item.get("preset_id") or "").strip() == pid:
            return item
    return None
