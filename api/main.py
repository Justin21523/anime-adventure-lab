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
import torch
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Import core modules
from core.shared_cache import bootstrap_cache
from core.config import get_config
from core.shared_cache import bootstrap_cache
from core.exceptions import MultiModalLabError
from core.llm.adapter import get_llm_adapter
from core.vlm.engine import get_vlm_engine

# Import routers (will be implemented in later stages)
from api.routers import (
    admin,
    batch,
    export,
    finetune,
    health,
    llm,
    monitoring,
    rag,
    safety,
    story,
    t2i,
    vlm,
    game,
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
    logger.info("🚀 Starting SagaForge API...")

    # Bootstrap shared cache
    cache = bootstrap_cache()
    app.state.cache = cache

    # Load configuration
    config = get_config()
    app.state.config = config

    logger.info(f"✅ SagaForge API ready at http://{config.api.host}:{config.api.port}")
    # Pre-load critical models if configured
    if config.get_feature_flag("preload_models"):
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
        title="SagaForge API",
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
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"Global exception: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "message": "SagaForge API - Stage 5: T2I Pipeline",
            "docs": "/docs",
            "health": "/healthz",
            "endpoints": {
                "t2i": "/api/v1/t2i",
            },
        }

    # Include routers
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    # Future routers (placeholders)
    app.include_router(llm.router, prefix="/api/v1", tags=["llm"])
    app.include_router(rag.router, prefix="/api/v1", tags=["rag"])
    app.include_router(t2i.router, prefix="/api/v1", tags=["t2i"])
    app.include_router(vlm.router, prefix="/api/v1", tags=["vlm"])
    app.include_router(finetune.router, prefix="/api/v1", tags=["finetune"])
    app.include_router(game.router, prefix="/api/v1", tags=["Game"])
    app.include_router(batch.router, prefix="/api/v1", tags=["batch"])

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
