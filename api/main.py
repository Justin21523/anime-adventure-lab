# api/main.py
"""
SagaForge FastAPI Application
Main entry point for the API server
"""

import logging
import uvicorn
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import sys
import types

# Inject a minimal torch stub if not available (API uses llama.cpp server, not torch)
if "torch" not in sys.modules:
    _fake_torch = types.ModuleType("torch")
    _fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    _fake_torch.__version__ = "2.0.0+cpu"
    # Common attributes that config.py and others expect
    for _attr in ["float16", "float32", "bfloat16", "device", "Tensor", "nn", "optim", "load", "save"]:
        setattr(_fake_torch, _attr, type('T', (), {}()))
    sys.modules["torch"] = _fake_torch

try:
    import torch
except ImportError:
    pass  # stub is in place

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Import core modules
from core.config import get_config, setup_logging
from core.shared_cache import bootstrap_cache
from core.exceptions import MultiModalLabError
from core.llm.adapter import get_llm_adapter
from core.vlm.engine import get_vlm_engine

# Import all routers
from api.routers import (
    health_router,
    caption_router,
    vqa_router,
    chat_router,
    rag_router,
    agent_router,
    game_router,
    admin_router,
    batch_router,
    controlnet_router,
    export_router,
    finetune_router,
    jobs_router,
    llm_router,
    lora_router,
    queue_router,
    runtime_router,
    ws_router,
    monitoring_router,
    performance_router,
    safety_router,
    story_router,
    t2i_router,
    vlm_router,
    worlds_router,
    datasets_router,
    models_router,
)


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("🚀 Starting Multi-Modal Lab API...")

    # Load configuration
    config = get_config()
    setup_logging(config)

    # Temporary: Enable DEBUG logging for turn processing debugging
    import logging
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger('core.story').setLevel(logging.DEBUG)
    logging.getLogger('api.routers').setLevel(logging.DEBUG)

    app.state.config = config

    # Bootstrap shared cache
    try:
        cache = bootstrap_cache()
        app.state.cache = cache
    except Exception as e:
        logger.error(f"❌ FATAL: Failed to bootstrap shared cache: {e}", exc_info=True)
        # Don't re-raise immediately, try to continue or provide better error
        raise

    logger.info(
        f"✅ Multi-Modal Lab API ready at http://{config.api.host}:{config.api.port}"
    )

    # Pre-load critical models only when explicitly configured. Heavy local
    # models should not block API startup; routers load them lazily on demand.
    if bool(config.get("features.preload_models", False)):
        try:
            vlm_engine = get_vlm_engine()
            vlm_engine.load_caption_model()
            logger.info("Caption model pre-loaded")
        except Exception as e:
            logger.warning(f"Failed to pre-load models: {e}")

    yield

    # Shutdown
    logger.info("Multi-Modal Lab API shutting down...")
    try:
        llm_adapter = get_llm_adapter()
        llm_adapter.unload_all()

        vlm_engine = get_vlm_engine()
        vlm_engine.unload_models()

        logger.info("Models unloaded successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI app
def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    config = get_config()

    app = FastAPI(
        title="CharaForge Multi-Modal Lab API",
        description="LLM + RAG + T2I + VLM Adventure Game Engine",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "*"],  # Configure for production
    )

    # Global exception handler
    @app.exception_handler(MultiModalLabError)
    async def handle_lab_error(request, exc: MultiModalLabError):
        """Handle custom lab exceptions"""
        return JSONResponse(
            status_code=400,
            content={
                "error": exc.error_code,
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(Exception)
    async def handle_general_error(request, exc: Exception):
        """Handle general exceptions"""
        logger = logging.getLogger(__name__)
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_ERROR",
                "message": "Internal server error occurred",
            },
        )

    # Include all routers with proper prefix
    API_PREFIX = config.api.prefix

    # Core functionality routers
    app.include_router(health_router, prefix=API_PREFIX, tags=["Health"])
    app.include_router(caption_router, prefix=API_PREFIX, tags=["Caption"])
    app.include_router(vqa_router, prefix=API_PREFIX, tags=["VQA"])
    app.include_router(chat_router, prefix=API_PREFIX, tags=["Chat"])
    app.include_router(rag_router, prefix=API_PREFIX, tags=["RAG"])
    app.include_router(agent_router, prefix=API_PREFIX, tags=["Agent"])
    app.include_router(game_router, prefix=API_PREFIX, tags=["Game"])

    # Advanced functionality routers
    app.include_router(t2i_router, prefix=API_PREFIX, tags=["Text2Image"])
    app.include_router(controlnet_router, prefix=API_PREFIX, tags=["ControlNet"])
    app.include_router(lora_router, prefix=API_PREFIX, tags=["LoRA"])

    # Management and ops routers
    app.include_router(admin_router, prefix=API_PREFIX, tags=["Admin"])
    app.include_router(batch_router, prefix=API_PREFIX, tags=["Batch"])
    app.include_router(monitoring_router, prefix=API_PREFIX, tags=["Monitoring"])
    app.include_router(performance_router, prefix=API_PREFIX, tags=["Performance"])
    app.include_router(safety_router, prefix=API_PREFIX, tags=["Safety"])

    # Training and export routers
    app.include_router(finetune_router, prefix=API_PREFIX, tags=["Finetune"])
    app.include_router(jobs_router, prefix=API_PREFIX, tags=["Jobs"])
    app.include_router(export_router, prefix=API_PREFIX, tags=["Export"])
    app.include_router(queue_router, prefix=API_PREFIX, tags=["Queue"])

    # Model-specific routers
    app.include_router(llm_router, prefix=API_PREFIX, tags=["LLM"])
    app.include_router(vlm_router, prefix=API_PREFIX, tags=["VLM"])
    app.include_router(runtime_router, prefix=API_PREFIX, tags=["Runtime"])
    app.include_router(story_router, prefix=API_PREFIX, tags=["Story"])
    app.include_router(worlds_router, prefix=API_PREFIX, tags=["Worlds"])
    app.include_router(datasets_router, prefix=API_PREFIX, tags=["Datasets"])
    app.include_router(models_router, prefix=API_PREFIX, tags=["Models"])
    app.include_router(ws_router, prefix=API_PREFIX, tags=["WebSocket"])

    # Root redirect
    @app.get("/")
    async def root():
        """Root endpoint redirect to docs"""
        return {
            "message": "CharaForge Multi-Modal Lab API",
            "version": "0.1.0",
            "docs_url": "/docs",
            "health_check": f"{API_PREFIX}/health",
            "status": f"{API_PREFIX}/status",
        }

    @app.get("/healthz", include_in_schema=False)
    async def healthz():
        """Lightweight health check for Docker/K8s."""
        return {"status": "healthy"}

    return app


# Create app instance
app = create_app()


def main():
    """Run the application"""
    config = get_config()

    uvicorn.run(
        "api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.debug,
        log_level="info" if not config.api.debug else "debug",
    )


if __name__ == "__main__":
    main()
