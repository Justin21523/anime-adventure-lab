# api/routers/__init__.py
"""
API Routers Module
Centralized router registration and imports
"""

from .caption import router as caption_router
from .vqa import router as vqa_router
from .chat import router as chat_router
from .rag import router as rag_router
from .agent import router as agent_router
from .game import router as game_router
from .health import router as health_router
from .admin import router as admin_router
from .batch import router as batch_router
from .controlnet import router as controlnet_router
from .export import router as export_router
from .finetune import router as finetune_router
from .llm import router as llm_router
from .lora import router as lora_router
from .monitoring import router as monitoring_router
from .safety import router as safety_router
from .story import router as story_router
from .t2i import router as t2i_router
from .vlm import router as vlm_router

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
    "llm_router",
    "lora_router",
    "monitoring_router",
    "safety_router",
    "story_router",
    "t2i_router",
    "vlm_router",
]
