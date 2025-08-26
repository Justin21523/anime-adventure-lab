# api/routers/__init__.py
"""
Unified router registration for all API endpoints
"""
from .health import router as health_router
from .chat import router as chat_router
from .game import router as game_router
from .rag import router as rag_router
from .t2i import router as t2i_router
from .controlnet import router as controlnet_router
from .lora import router as lora_router
from .batch import router as batch_router
from .caption import router as caption_router
from .vqa import router as vqa_router
from .admin import router as admin_router

__version__ = "0.1.0"
# Export all routers for main app
routers = [
    ("health", health_router),
    ("chat", chat_router),
    ("game", game_router),
    ("rag", rag_router),
    ("t2i", t2i_router),
    ("controlnet", controlnet_router),
    ("lora", lora_router),
    ("batch", batch_router),
    ("caption", caption_router),
    ("vqa", vqa_router),
    ("admin", admin_router),
]

__all__ = [
    "routers",
    "health_router",
    "chat_router",
    "game_router",
    "rag_router",
    "t2i_router",
    "controlnet_router",
    "lora_router",
    "batch_router",
    "caption_router",
    "vqa_router",
    "admin_router",
]
