# api/dependencies.py
"""
FastAPI dependencies for VLM services
Provides dependency injection for VLM captioner and related services
"""
import os, pathlib, torch
from functools import lru_cache
from typing import Iterator, Dict, Any, Optional, List
from fastapi import Depends, HTTPException

from core.vlm.captioner import VLMCaptioner
from core.vlm.tagger import WD14Tagger
from core.config import get_config
from core.rag.engine import ChineseRAGEngine

# Global instances (lazy loaded)
_vlm_captioner = None
_wd14_tagger = None

AI_CACHE_ROOT = os.getenv("AI_CACHE_ROOT", "../ai_warehouse/cache")
for k, v in {
    "HF_HOME": f"{AI_CACHE_ROOT}/hf",
    "TRANSFORMERS_CACHE": f"{AI_CACHE_ROOT}/hf/transformers",
    "HF_DATASETS_CACHE": f"{AI_CACHE_ROOT}/hf/datasets",
    "HUGGINGFACE_HUB_CACHE": f"{AI_CACHE_ROOT}/hf/hub",
    "TORCH_HOME": f"{AI_CACHE_ROOT}/torch",
}.items():
    os.environ[k] = v
    pathlib.Path(v).mkdir(parents=True, exist_ok=True)

APP_DIRS = {
    "MODELS_SD": f"{AI_CACHE_ROOT}/models/sd",
    "MODELS_SDXL": f"{AI_CACHE_ROOT}/models/sdxl",
    "MODELS_CONTROLNET": f"{AI_CACHE_ROOT}/models/controlnet",
    "MODELS_LORA": f"{AI_CACHE_ROOT}/models/lora",
    "MODELS_IPADAPTER": f"{AI_CACHE_ROOT}/models/ipadapter",
    "DATASETS_META": f"{AI_CACHE_ROOT}/datasets/metadata",
    "OUTPUT_DIR": f"{AI_CACHE_ROOT}/outputs/saga-forge",
    # RAG
    "RAG_INDEX": f"{AI_CACHE_ROOT}/rag/indexes",
    "RAG_DOCS": f"{AI_CACHE_ROOT}/rag/documents",
    "RAG_EMBEDDINGS": f"{AI_CACHE_ROOT}/rag/embeddings",
    "WORLDPACKS": f"{AI_CACHE_ROOT}/worldpacks",
}
for p in APP_DIRS.values():
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_vlm_config() -> Dict[str, Any]:
    """Get VLM configuration with low-VRAM defaults"""
    config = get_config()
    vlm_config = config.get("vlm", {})

    # Set sensible defaults for low-VRAM environments
    defaults = {
        "default_model": "blip2",
        "device": "auto",
        "low_vram": True,
        "models": {
            "blip2": "Salesforce/blip2-opt-2.7b",  # Smaller BLIP2 model
            "llava": "liuhaotian/llava-v1.6-mistral-7b",
        },
        "wd14": {"model": "SmilingWolf/wd-v1-4-convnext-tagger-v2", "threshold": 0.35},
        "consistency": {
            "embed_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "semantic_threshold": 0.6,
        },
    }

    # Merge with defaults
    for key, value in defaults.items():
        if key not in vlm_config:
            vlm_config[key] = value
        elif isinstance(value, dict) and isinstance(vlm_config[key], dict):
            for sub_key, sub_value in value.items():
                if sub_key not in vlm_config[key]:
                    vlm_config[key][sub_key] = sub_value

    return vlm_config


def get_vlm_captioner() -> VLMCaptioner:
    """Get shared VLM captioner instance"""
    global _vlm_captioner

    if _vlm_captioner is None:
        try:
            config = get_vlm_config()
            _vlm_captioner = VLMCaptioner(config)
        except Exception as e:
            raise HTTPException(
                status_code=503, detail=f"Failed to initialize VLM captioner: {str(e)}"
            )

    return _vlm_captioner


def get_wd14_tagger() -> Optional[WD14Tagger]:
    """Get shared WD14 tagger instance"""
    global _wd14_tagger

    if _wd14_tagger is None:
        try:
            config = get_vlm_config()
            wd14_config = config.get("wd14", {})

            _wd14_tagger = WD14Tagger(
                model_name=wd14_config.get(
                    "model", "SmilingWolf/wd-v1-4-convnext-tagger-v2"
                ),
                device=config.get("device", "auto"),
                threshold=wd14_config.get("threshold", 0.35),
            )
        except Exception as e:
            # WD14 is optional, don't fail the entire service
            print(f"Warning: Failed to initialize WD14 tagger: {e}")
            return None

    return _wd14_tagger


async def cleanup_vlm_services():
    """Cleanup VLM services on shutdown"""
    global _vlm_captioner, _wd14_tagger

    if _vlm_captioner:
        _vlm_captioner.unload_all()
        _vlm_captioner = None

    if _wd14_tagger:
        _wd14_tagger.unload()
        _wd14_tagger = None


def get_cache_root() -> str:
    return AI_CACHE_ROOT


def get_app_dir() -> Dict:
    return APP_DIRS


# Dependency functions for FastAPI
def vlm_captioner_dependency() -> Optional[VLMCaptioner]:
    """FastAPI dependency for VLM captioner"""
    return get_vlm_captioner()


def wd14_tagger_dependency() -> Optional[WD14Tagger]:
    """FastAPI dependency for WD14 tagger"""
    return get_wd14_tagger()


def get_rag_engine():
    # Initialize RAG engine
    return ChineseRAGEngine(APP_DIRS["RAG_INDEX"])
