# core/vlm/config.py
"""VLM configuration"""
from typing import Dict, Any
from ..config import get_config as get_app_config


def get_vlm_settings() -> Dict[str, Any]:
    """Get VLM-specific settings"""
    config = get_app_config()
    return {
        "default_model": config.model.default_vlm_model,
        "device": config.model.device_map,
        "use_fp16": config.model.use_fp16,
        "max_length": 50,
        "num_beams": 3,
        "temperature": 0.7,
    }


def get_summary() -> Dict[str, Any]:
    """Get VLM summary for health checks"""
    settings = get_vlm_settings()
    return {
        "available_models": ["blip2", "llava", "qwen-vl"],
        "default_model": settings["default_model"],
        "device": settings["device"],
        "capabilities": ["caption", "vqa", "analyze"],
    }
