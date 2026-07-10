from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

SESSION_COOKIE = "saga_session"
CSRF_COOKIE = "saga_csrf"
SESSION_MAX_AGE = 8 * 60 * 60


@dataclass(frozen=True)
class BrowserSession:
    username: str
    role: str
    csrf_token: str
    expires_at: int


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _secret() -> bytes:
    value = str(
        os.getenv("API_SESSION_SECRET") or os.getenv("API_SECRET_KEY") or ""
    ).strip()
    if not value or value == "your-secret-key-here-change-in-production":
        raise RuntimeError("Browser session secret is not configured")
    return value.encode("utf-8")


def create_browser_session(username: str = "admin") -> tuple[str, BrowserSession]:
    now = int(time.time())
    session = BrowserSession(
        username=username,
        role="admin",
        csrf_token=secrets.token_urlsafe(32),
        expires_at=now + SESSION_MAX_AGE,
    )
    payload = _b64encode(
        json.dumps(
            {
                "sub": session.username,
                "role": session.role,
                "csrf": session.csrf_token,
                "iat": now,
                "exp": session.expires_at,
            },
            separators=(",", ":"),
        ).encode("utf-8")
    )
    signature = _b64encode(
        hmac.new(_secret(), payload.encode("ascii"), hashlib.sha256).digest()
    )
    return f"{payload}.{signature}", session


def verify_browser_session(token: str | None) -> BrowserSession | None:
    if not token or "." not in token:
        return None
    payload, signature = token.rsplit(".", 1)
    try:
        expected = _b64encode(
            hmac.new(_secret(), payload.encode("ascii"), hashlib.sha256).digest()
        )
        if not hmac.compare_digest(signature, expected):
            return None
        data: dict[str, Any] = json.loads(_b64decode(payload))
        expires_at = int(data.get("exp") or 0)
        if expires_at <= int(time.time()):
            return None
        if data.get("role") != "admin":
            return None
        return BrowserSession(
            username=str(data.get("sub") or "admin"),
            role="admin",
            csrf_token=str(data.get("csrf") or ""),
            expires_at=expires_at,
        )
    except (ValueError, TypeError, json.JSONDecodeError, RuntimeError):
        return None


def verify_admin_password(password: str) -> bool:
    encoded = str(os.getenv("API_ADMIN_PASSWORD_HASH") or "").strip()
    if not encoded:
        return False
    try:
        return PasswordHasher().verify(encoded, password)
    except (VerifyMismatchError, InvalidHashError):
        return False


def csrf_matches(
    session: BrowserSession, cookie: str | None, header: str | None
) -> bool:
    if not cookie or not header:
        return False
    return hmac.compare_digest(cookie, session.csrf_token) and hmac.compare_digest(
        header, session.csrf_token
    )
