# api/dependencies.py
"""
Centralized dependency providers for FastAPI routers.
- Single source of truth for settings and cache paths
- Lazy-loaded global singletons for engines (t2i / lora / controlnet / llm / vlm / rag / game)
- No side effects at import time (no mkdir, no downloads here)
"""
from __future__ import annotations
from functools import lru_cache
import os, pathlib, torch
from functools import lru_cache
from typing import Iterator, Dict, Any, Optional, List
from fastapi import Depends, HTTPException

from core import registry
from core.config import get_config
from core.shared_cache import get_shared_cache, bootstrap_cache
from core.safety.detector import SafetyEngine
from core.safety.license import LicenseManager
from core.safety.watermark import AttributionManager, ComplianceLogger
from core.story.engine import StoryEngine
from core.story.persona import PersonaManager
from core.performance import gpu_available  # optional, but useful for /health

# Replace with your actual implementation when ready
# from core.t2i.pipeline import RealT2IEngine
# from core.llm.adapter import RealLLM
# from core.vlm.captioner import RealVLM
# from core.rag.engine import RealRAG
# from core.story.engine import RealStoryEngine


# Lazy singletons — keep all globals in one place

_settings = None  # App settings (Pydantic)
_cache = None  # Shared cache handle

_t2i = None  # Text-to-image engine (exposes txt2img(); has .lora and .control)
_llm = None  # Chat LLM engine       (exposes chat(messages))
_vlm = None  # VLM engine            (exposes caption(path), vqa(path, q))
_rag = None  # RAG engine            (exposes ask(query))
_game = None  # Story/game engine     (exposes new(seed?), step(state, choice))
_safety = None
_license_mgr = None
_attr = None
_compliance = None
_story_engine = None
_persona_mgr = None


# Settings & Cache (never re-define env or paths here)
def get_settings():
    """App-wide settings (single source of truth)."""
    global _settings
    if _settings is None:
        _settings = get_config()
    return _settings


def get_cache():
    """Shared cache directories; ensure bootstrap has been run at least once."""
    global _cache
    if _cache is None:
        # Safe to call multiple times; bootstrap is idempotent
        bootstrap_cache()
        _cache = get_shared_cache()
    return _cache


# Factory helpers — try real engines first, otherwise fall back to stubs
def _new_t2i():
    try:
        # return RealT2IEngine()
        return _MinimalT2I()
    except Exception as e:
        raise RuntimeError(f"T2I engine init failed: {e}") from e


def _new_llm():
    try:

        # return RealLLM()
        return _MinimalLLM()
    except Exception as e:
        raise RuntimeError(f"LLM engine init failed: {e}") from e


def _new_vlm():
    try:

        # return RealVLM()
        return _MinimalVLM()
    except Exception as e:
        raise RuntimeError(f"VLM engine init failed: {e}") from e


def _new_rag():
    try:

        # return RealRAG()
        return _MinimalRAG()
    except Exception as e:
        raise RuntimeError(f"RAG engine init failed: {e}") from e


def _new_game():
    try:

        # return RealStoryEngine()
        return _MinimalStory()
    except Exception as e:
        raise RuntimeError(f"Game engine init failed: {e}") from e


# Public dependency getters (use these in routers; do NOT `new` engines in routers)
# ---------------------------------------------------------------------
def get_registry() -> Dict[str, Any]:
    """
    Optional: expose a dict-like registry for advanced cases.
    Most routers should just call get_t2i/get_llm/... directly.
    """
    return {
        "settings": get_settings(),
        "cache": get_cache(),
        "t2i": get_t2i(),
        "lora": get_lora(),
        "controlnet": get_controlnet(),
        "llm": get_llm(),
        "vlm": get_vlm(),
        "rag": get_rag(),
        "game": get_game(),
        "health": {"gpu_available": gpu_available()},
    }


def get_t2i():
    """Shared T2I engine singleton."""
    global _t2i
    if _t2i is None:
        try:
            _t2i = _new_t2i()
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))
    return _t2i


def get_lora():
    """LoRA manager derived from the T2I engine."""
    t2i = get_t2i()
    if not hasattr(t2i, "lora"):
        raise HTTPException(status_code=501, detail="LoRA manager not available")
    return t2i.lora


def get_controlnet():
    """ControlNet processor derived from the T2I engine."""
    t2i = get_t2i()
    if not hasattr(t2i, "control"):
        raise HTTPException(status_code=501, detail="ControlNet not available")
    return t2i.control


def get_llm():
    """Shared LLM chat engine singleton."""
    global _llm
    if _llm is None:
        try:
            _llm = _new_llm()
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))
    return _llm


def get_vlm():
    """Shared VLM engine singleton (caption & VQA)."""
    global _vlm
    if _vlm is None:
        try:
            _vlm = _new_vlm()
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))
    return _vlm


def get_rag():
    """Shared RAG engine singleton."""
    global _rag
    if _rag is None:
        try:
            _rag = _new_rag()
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))
    return _rag


def get_game():
    """Shared game/story engine singleton."""
    global _game
    if _game is None:
        try:
            _game = _new_game()
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))
    return _game


def get_safety_engine():
    global _safety
    if _safety is None:
        _safety = SafetyEngine()
    return _safety


def get_license_manager():
    global _license_mgr
    if _license_mgr is None:
        cache = get_shared_cache()
        _license_mgr = LicenseManager(cache.cache_root)
    return _license_mgr


def get_attribution_manager():
    global _attr
    if _attr is None:
        cache = get_shared_cache()
        _attr = AttributionManager(cache.cache_root)
    return _attr


def get_compliance_logger():
    global _compliance
    if _compliance is None:
        cache = get_shared_cache()
        _compliance = ComplianceLogger(cache.cache_root)
    return _compliance


def get_story_engine(llm=None):
    # If llm DI is needed, import here to avoid circulars
    from .dependencies import get_llm  # local import to prevent cycle

    global _story_engine
    if _story_engine is None:
        _story_engine = StoryEngine(get_llm())
    return _story_engine


def get_persona_manager():
    global _persona_mgr
    if _persona_mgr is None:
        _persona_mgr = PersonaManager()
    return _persona_mgr


# Optional: settings slices
@lru_cache
def get_vlm_settings():
    """Expose VLM-specific settings slice if you keep one in AppConfig."""
    cfg = get_settings()
    # adapt to your actual config structure
    return getattr(cfg, "vlm", None)
