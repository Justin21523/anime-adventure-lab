from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import time, logging, os

logger = logging.getLogger("saga")
logging.basicConfig(level=logging.INFO)

def setup_middleware(app: FastAPI):
    origins = os.getenv("API_CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_timing(request: Request, call_next):
        start = time.time()
        resp = await call_next(request)
        logger.info("%s %s -> %s in %.1fms", request.method, request.url.path, resp.status_code, (time.time()-start)*1000)
        return resp
