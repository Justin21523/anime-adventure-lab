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
from fastapi.responses import JSONResponse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Import core modules
from core.shared_cache import bootstrap_cache
from core.config import get_config

cache = bootstrap_cache()

# Import routers (will be implemented in later stages)
from api.routers import health, llm, rag, story, t2i

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

    yield

    # Shutdown
    logger.info("🛑 Shutting down SagaForge API...")


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

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"Global exception: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )

    # Health check
    @app.get("/healthz")
    async def health_check():
        """Global health check"""
        try:
            # Basic health indicators
            import torch

            gpu_available = torch.cuda.is_available()

            health_status = {
                "status": "healthy",
                "version": "0.5.0",
                "stage": "Stage 5 - T2I Pipeline",
                "gpu_available": gpu_available,
                "cache_root": os.getenv("AI_CACHE_ROOT", "default"),
            }

            # GPU memory info if available
            if gpu_available:
                health_status["gpu_memory_mb"] = (
                    torch.cuda.memory_allocated(0) / 1024**2
                )

            return health_status

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Health check failed: {str(e)}"
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
    # app.include_router(vlm.router, prefix="/api/v1", tags=["vlm"])
    # app.include_router(finetune.router, prefix="/api/v1", tags=["finetune"])
    # app.include_router(batch.router, prefix="/api/v1", tags=["batch"])

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
