from __future__ import annotations

import hmac
import os

from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from api.session_auth import (
    CSRF_COOKIE,
    SESSION_COOKIE,
    SESSION_MAX_AGE,
    create_browser_session,
    verify_admin_password,
)

router = APIRouter()


class LoginRequest(BaseModel):
    username: str = Field(default="admin", min_length=1, max_length=120)
    password: str = Field(min_length=8, max_length=1024)


def _secure_cookie() -> bool:
    explicit = os.getenv("API_COOKIE_SECURE")
    if explicit is not None:
        return explicit.strip().lower() in {"1", "true", "yes", "on"}
    return os.getenv("APP_ENV", "development").strip().lower() == "production"


@router.post("/auth/session")
def login(request: LoginRequest, response: Response):
    expected_username = os.getenv("API_ADMIN_USERNAME", "admin")
    valid_password = verify_admin_password(request.password)
    valid_username = hmac.compare_digest(request.username, expected_username)
    if not valid_username or not valid_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Session"},
        )
    token, session = create_browser_session(request.username)
    cookie_options = {
        "secure": _secure_cookie(),
        "samesite": "lax",
        "path": "/",
        "max_age": SESSION_MAX_AGE,
    }
    response.set_cookie(SESSION_COOKIE, token, httponly=True, **cookie_options)
    response.set_cookie(
        CSRF_COOKIE, session.csrf_token, httponly=False, **cookie_options
    )
    return {
        "authenticated": True,
        "username": session.username,
        "role": session.role,
        "csrf_token": session.csrf_token,
        "expires_at": session.expires_at,
    }


@router.get("/auth/session")
def current_session(request: Request):
    auth = getattr(request.state, "auth", None)
    if auth is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"authenticated": True, "role": auth.role, "source": auth.source}


@router.delete("/auth/session", status_code=status.HTTP_204_NO_CONTENT)
def logout(response: Response):
    response.delete_cookie(SESSION_COOKIE, path="/")
    response.delete_cookie(CSRF_COOKIE, path="/")
