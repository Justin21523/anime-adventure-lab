from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.smoke
def test_story_dependency_builds_engine_with_config_directory(monkeypatch, tmp_path):
    import api.dependencies as dependencies

    monkeypatch.setenv("AI_OUTPUT_ROOT", str(tmp_path / "outputs"))
    monkeypatch.setenv("LLM_MOCK", "1")
    dependencies._story_engine = None

    engine = dependencies.get_story_engine()

    assert engine.__class__.__name__ == "StoryEngine"
    assert isinstance(engine.sessions_path, Path)


@pytest.mark.smoke
def test_generic_agent_tool_api_is_disabled_by_default(client, monkeypatch):
    monkeypatch.delenv("AGENT_ENABLE_GENERIC_TOOL_API", raising=False)

    response = client.post(
        "/api/v1/agent/tools/call",
        json={"tool_name": "calculator", "parameters": {"expression": "2+2"}},
    )

    assert response.status_code == 403


@pytest.mark.smoke
def test_api_auth_uses_header_credentials_when_enabled(client, monkeypatch):
    monkeypatch.setenv("ENABLE_API_AUTH", "1")
    monkeypatch.setenv("API_SECRET_KEY", "test-admin-secret")

    denied = client.get("/api/v2/auth/session")
    allowed = client.get(
        "/api/v2/auth/session",
        headers={"X-API-Key": "test-admin-secret"},
    )

    assert denied.status_code == 401
    assert allowed.status_code == 200


@pytest.mark.smoke
def test_file_operations_reject_repository_paths(monkeypatch, tmp_path):
    from core.agents.tools.file_ops import SafeFileOperations

    monkeypatch.setenv("AI_OUTPUT_ROOT", str(tmp_path / "outputs"))
    monkeypatch.delenv("AGENT_FILE_ROOTS", raising=False)
    operations = SafeFileOperations()

    assert operations._is_path_allowed(str(tmp_path / "outputs" / "artifact.json"))
    assert not operations._is_path_allowed(str(Path(__file__).resolve()))


def test_database_and_cache_settings_accept_runtime_environment(monkeypatch):
    from core.config import CacheConfig, DatabaseConfig

    monkeypatch.setenv("DATABASE_URL", "postgresql://demo:secret@db/demo")
    monkeypatch.setenv("AI_CACHE_ROOT", "/tmp/anime-cache")
    monkeypatch.setenv("REDIS_URL", "redis://redis:6379/4")

    assert DatabaseConfig().url == "postgresql://demo:secret@db/demo"
    cache = CacheConfig()
    assert cache.root == "/tmp/anime-cache"
    assert cache.redis_url == "redis://redis:6379/4"


def test_browser_session_uses_http_only_cookie_and_csrf(client, monkeypatch):
    from argon2 import PasswordHasher

    monkeypatch.setenv("ENABLE_API_AUTH", "1")
    monkeypatch.setenv("API_SECRET_KEY", "test-session-signing-secret")
    monkeypatch.setenv(
        "API_ADMIN_PASSWORD_HASH", PasswordHasher().hash("correct-horse")
    )
    monkeypatch.setenv("API_COOKIE_SECURE", "0")
    monkeypatch.delenv("AGENT_ENABLE_GENERIC_TOOL_API", raising=False)

    login = client.post(
        "/api/v2/auth/session",
        json={"username": "admin", "password": "correct-horse"},
    )
    assert login.status_code == 200
    assert "HttpOnly" in login.headers["set-cookie"]

    current = client.get("/api/v2/auth/session")
    missing_csrf = client.post(
        "/api/v1/agent/tools/call",
        json={"tool_name": "calculator", "parameters": {"expression": "2+2"}},
    )
    csrf = login.json()["csrf_token"]
    valid_csrf = client.post(
        "/api/v1/agent/tools/call",
        headers={"X-CSRF-Token": csrf},
        json={"tool_name": "calculator", "parameters": {"expression": "2+2"}},
    )

    assert current.status_code == 200
    assert current.json()["source"] == "session"
    assert missing_csrf.status_code == 403
    assert missing_csrf.json()["code"] == "CSRF_VALIDATION_FAILED"
    assert valid_csrf.status_code == 403
    assert "Generic agent tool execution is disabled" in valid_csrf.json()["detail"]


def test_problem_details_include_stable_request_id(client):
    response = client.post(
        "/api/v2/worlds",
        headers={"X-Request-ID": "portfolio-test-001"},
        json={"world_id": "invalid id", "name": ""},
    )

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.headers["x-request-id"] == "portfolio-test-001"
    assert response.json()["request_id"] == "portfolio-test-001"
    assert response.json()["code"] == "VALIDATION_ERROR"


def test_private_api_docs_require_authentication(monkeypatch):
    from fastapi.testclient import TestClient

    from api.main import app

    monkeypatch.setenv("ENABLE_API_AUTH", "1")
    monkeypatch.setenv("API_SECRET_KEY", "private-docs-test-secret")
    monkeypatch.delenv("PUBLIC_API_DOCS", raising=False)

    with TestClient(app) as isolated_client:
        response = isolated_client.get("/openapi.json")

    assert response.status_code == 401
    assert response.json()["code"] == "UNAUTHORIZED"


def test_login_has_separate_rate_limit(client, monkeypatch):
    monkeypatch.setenv("ENABLE_API_AUTH", "1")
    monkeypatch.setenv("API_SECRET_KEY", "login-rate-test-secret")
    monkeypatch.setenv("API_LOGIN_RATE_LIMIT_PER_MINUTE", "1")
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "1")
    headers = {"X-Forwarded-For": "198.51.100.77"}
    payload = {"username": "admin", "password": "definitely-wrong"}

    first = client.post("/api/v2/auth/session", headers=headers, json=payload)
    second = client.post("/api/v2/auth/session", headers=headers, json=payload)

    assert first.status_code == 401
    assert second.status_code == 429
    assert second.json()["code"] == "RATE_LIMITED"


def test_core_worker_profile_does_not_import_torch():
    environment = os.environ.copy()
    environment["WORKER_PROFILE"] = "core"
    environment.pop("ENABLE_EXPERIMENTAL_WORKER_TASKS", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import workers.celery_app as worker; "
                "assert worker.task_modules == "
                "['workers.tasks.story_v2', 'workers.tasks.rag_v2', "
                "'workers.tasks.maintenance_v2']; "
                "assert 'torch' not in sys.modules"
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
