# api/routers/__init__.py
"""
API Routers Module
Centralized router registration and imports
"""

from __future__ import annotations

import logging
from importlib import import_module

from fastapi import APIRouter

logger = logging.getLogger(__name__)


def _load_router(module_name: str) -> APIRouter:
    """
    Best-effort router import.

    目的：讓 API 在「可選依賴缺失」時仍能啟動（例如 tests / lite 部署），
    缺少的功能會以空 router 取代並在 log 提示。
    """
    try:
        module = import_module(f"{__name__}.{module_name}")
        router = getattr(module, "router")
        if isinstance(router, APIRouter):
            return router
        logger.warning("Router %s imported but has no APIRouter 'router' attribute", module_name)
        return APIRouter()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Router %s unavailable: %s", module_name, exc)
        return APIRouter()


caption_router = _load_router("caption")
vqa_router = _load_router("vqa")
chat_router = _load_router("chat")
rag_router = _load_router("rag")
agent_router = _load_router("agent")
game_router = _load_router("game")
health_router = _load_router("health")
admin_router = _load_router("admin")
batch_router = _load_router("batch")
controlnet_router = _load_router("controlnet")
export_router = _load_router("export")
finetune_router = _load_router("finetune")
jobs_router = _load_router("jobs")
llm_router = _load_router("llm")
lora_router = _load_router("lora")
queue_router = _load_router("queue")
runtime_router = _load_router("runtime")
monitoring_router = _load_router("monitoring")
performance_router = _load_router("performance")
safety_router = _load_router("safety")
story_router = _load_router("story")
t2i_router = _load_router("t2i")
vlm_router = _load_router("vlm")
worlds_router = _load_router("worlds")
datasets_router = _load_router("datasets")
models_router = _load_router("models")
ws_router = _load_router("ws")

__all__ = [
    "caption_router",
    "vqa_router",
    "chat_router",
    "rag_router",
    "agent_router",
    "game_router",
    "health_router",
    "admin_router",
    "batch_router",
    "controlnet_router",
    "export_router",
    "finetune_router",
    "jobs_router",
    "llm_router",
    "lora_router",
    "queue_router",
    "runtime_router",
    "monitoring_router",
    "performance_router",
    "safety_router",
    "story_router",
    "t2i_router",
    "vlm_router",
    "worlds_router",
    "datasets_router",
    "models_router",
    "ws_router",
]
