# api/main.py
"""
SagaForge FastAPI Application
Main entry point for the API server
"""

import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import os

# Import core modules
from core.config import get_config, setup_logging
from core.shared_cache import bootstrap_cache
from core.exceptions import MultiModalLabError

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
            from core.vlm.engine import get_vlm_engine

            vlm_engine = get_vlm_engine()
            vlm_engine.load_caption_model()
            logger.info("Caption model pre-loaded")
        except Exception as e:
            logger.warning(f"Failed to pre-load models: {e}")

    yield

    # Shutdown
    logger.info("Multi-Modal Lab API shutting down...")
    try:
        from core.llm.runtime import get_runtime_llm

        llm_adapter = get_runtime_llm()
        llm_adapter.unload_all()
    except Exception as e:
        logger.warning("LLM shutdown cleanup skipped: %s", e)
    try:
        from core.vlm.engine import get_vlm_engine

        vlm_engine = get_vlm_engine()
        vlm_engine.unload_models()
    except Exception as e:
        logger.debug("VLM shutdown cleanup skipped: %s", e)


# Create FastAPI app
def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    config = get_config()

    app = FastAPI(
        title="SagaForge Story Workbench API",
        description="Transactional Story, World, RAG, job, and artifact services.",
        version="0.2.0",
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

    trusted_hosts = [
        host.strip()
        for host in os.getenv(
            "API_TRUSTED_HOSTS", "localhost,127.0.0.1,testserver,api"
        ).split(",")
        if host.strip()
    ]
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)

    from api.middleware import setup_security_middleware

    setup_security_middleware(app)

    from api.problem import problem_response

    # Global exception handler
    @app.exception_handler(MultiModalLabError)
    async def handle_lab_error(request: Request, exc: MultiModalLabError):
        return problem_response(
            request,
            status_code=400,
            title="Application error",
            detail=exc.message,
            code=exc.error_code,
            errors=exc.details,
        )

    @app.exception_handler(HTTPException)
    async def handle_http_error(request: Request, exc: HTTPException):
        detail = exc.detail
        code = "HTTP_ERROR"
        errors = None
        if isinstance(detail, dict):
            code = str(detail.get("code") or code)
            errors = detail
            detail = str(detail.get("message") or detail.get("detail") or code)
        return problem_response(
            request,
            status_code=exc.status_code,
            title="Request failed",
            detail=str(detail),
            code=code,
            errors=errors,
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        return problem_response(
            request,
            status_code=422,
            title="Validation failed",
            detail="The request did not pass validation",
            code="VALIDATION_ERROR",
            errors=exc.errors(),
        )

    @app.exception_handler(Exception)
    async def handle_general_error(request: Request, exc: Exception):
        logger = logging.getLogger(__name__)
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return problem_response(
            request,
            status_code=500,
            title="Internal Server Error",
            detail="Internal server error occurred",
            code="INTERNAL_ERROR",
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

    # Story-first transactional API. This surface is intentionally small and
    # does not import model runtimes in the API process.
    from api.routers.v2 import router as v2_router
    from api.routers.auth import router as auth_router

    app.include_router(v2_router, prefix="/api/v2", tags=["V2"])
    app.include_router(auth_router, prefix="/api/v2", tags=["Authentication"])

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
