# api/middleware.py
import logging
import os
import re
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from api.problem import problem_response
from api.security import (
    APIKeyAuth,
    RateLimiter,
    get_client_key,
    parse_api_keys,
    resolve_api_key,
)
from api.session_auth import (
    CSRF_COOKIE,
    SESSION_COOKIE,
    csrf_matches,
    verify_browser_session,
)

logger = logging.getLogger("saga")

_rate_limiter = RateLimiter(window_seconds=60)


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def setup_security_middleware(app: FastAPI) -> None:
    """Install the baseline API authentication and rate-limit boundary.

    Authentication is opt-in for local development, but production startup is
    expected to set ``ENABLE_API_AUTH=1`` and one or more non-placeholder keys.
    The middleware intentionally accepts credentials from headers only so API
    keys are not leaked into URLs or proxy logs.
    """

    exempt_paths = {
        "/",
        "/healthz",
        "/api/v1/health",
        "/api/v1/ready",
        "/api/v1/status",
        "/api/v2/system/capabilities",
    }
    documentation_paths = {"/docs", "/redoc", "/openapi.json"}

    @app.middleware("http")
    async def secure_api_requests(request: Request, call_next):
        started_at = time.perf_counter()
        incoming_request_id = request.headers.get("X-Request-ID", "")
        if re.fullmatch(r"[A-Za-z0-9._:-]{8,128}", incoming_request_id):
            request_id = incoming_request_id
        else:
            request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        path = request.url.path
        login_request = path == "/api/v2/auth/session" and request.method == "POST"
        public_docs = _truthy(os.getenv("PUBLIC_API_DOCS"))
        if (
            (not path.startswith("/api/") and path not in documentation_paths)
            or path in exempt_paths
            or (path in documentation_paths and public_docs)
            or request.method == "OPTIONS"
        ):
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            logger.info(
                "request_complete",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": path,
                    "status_code": response.status_code,
                    "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
                },
            )
            return response

        presented = request.headers.get("X-API-Key")
        authorization = request.headers.get("Authorization", "")
        if not presented and authorization.lower().startswith("bearer "):
            presented = authorization.split(" ", 1)[1].strip()

        auth_enabled = _truthy(os.getenv("ENABLE_API_AUTH"))
        secret = str(os.getenv("API_SECRET_KEY") or "").strip()
        admin_keys = parse_api_keys(os.getenv("API_ADMIN_KEYS"))
        user_keys = parse_api_keys(os.getenv("API_USER_KEYS"))
        if secret and secret != "your-secret-key-here-change-in-production":
            admin_keys.add(secret)

        browser_session = None
        auth = resolve_api_key(presented, admin_keys=admin_keys, user_keys=user_keys)
        if auth is None:
            browser_session = verify_browser_session(
                request.cookies.get(SESSION_COOKIE)
            )
            if browser_session is not None:
                auth = APIKeyAuth(role="admin", scopes={"*"}, source="session")

        if auth_enabled and not login_request:
            if (
                not admin_keys
                and not user_keys
                and not os.getenv("API_ADMIN_PASSWORD_HASH")
            ):
                logger.error("API auth is enabled but no API keys are configured")
                return problem_response(
                    request,
                    status_code=503,
                    title="Authentication unavailable",
                    detail="API authentication is not configured",
                    code="AUTH_NOT_CONFIGURED",
                )
            if auth is None:
                return problem_response(
                    request,
                    status_code=401,
                    title="Unauthorized",
                    detail="Valid API credentials are required",
                    code="UNAUTHORIZED",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            request.state.auth = auth

            if browser_session is not None and request.method not in {
                "GET",
                "HEAD",
                "OPTIONS",
            }:
                if not csrf_matches(
                    browser_session,
                    request.cookies.get(CSRF_COOKIE),
                    request.headers.get("X-CSRF-Token"),
                ):
                    return problem_response(
                        request,
                        status_code=403,
                        title="Forbidden",
                        detail="A valid CSRF token is required",
                        code="CSRF_VALIDATION_FAILED",
                    )

        limit_env = (
            "API_LOGIN_RATE_LIMIT_PER_MINUTE"
            if login_request
            else "API_RATE_LIMIT_PER_MINUTE"
        )
        default_limit = "10" if login_request else "120"
        try:
            limit = max(1, int(os.getenv(limit_env, default_limit)))
        except ValueError:
            limit = int(default_limit)
        client_key = get_client_key(
            request,
            None if login_request else presented or request.cookies.get(SESSION_COOKIE),
        )
        result = _rate_limiter.check(
            key=f"{'login' if login_request else 'api'}:{client_key}",
            limit=limit,
            redis_url=os.getenv("REDIS_URL"),
        )
        if not result.allowed:
            return problem_response(
                request,
                status_code=429,
                title="Too Many Requests",
                detail="Too many requests",
                code="RATE_LIMITED",
                headers={
                    "Retry-After": str(max(1, result.reset_epoch - int(time.time()))),
                    "X-RateLimit-Limit": str(result.limit),
                    "X-RateLimit-Remaining": str(result.remaining),
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(result.limit)
        response.headers["X-RateLimit-Remaining"] = str(result.remaining)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request_complete",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": path,
                "status_code": response.status_code,
                "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
            },
        )
        return response


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
        logger.info(
            "%s %s -> %s in %.1fms",
            request.method,
            request.url.path,
            resp.status_code,
            (time.time() - start) * 1000,
        )
        return resp
