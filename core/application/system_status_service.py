from __future__ import annotations

from datetime import datetime, timezone
import os
import time

from sqlalchemy import text

from core.persistence.database import get_database
from core.storage.object_store import get_object_store


class SystemStatusService:
    _cached_at = 0.0
    _cached: dict | None = None

    def check(self) -> dict:
        now_monotonic = time.monotonic()
        if self.__class__._cached and now_monotonic - self.__class__._cached_at < 5:
            return self.__class__._cached
        services: dict[str, dict[str, str | None]] = {}
        migration_revision: str | None = None

        try:
            with get_database().engine.connect() as connection:
                connection.execute(text("SELECT 1"))
                services["postgres"] = {"status": "healthy", "detail": None}
                try:
                    migration_revision = connection.scalar(
                        text("SELECT version_num FROM alembic_version LIMIT 1")
                    )
                except Exception:
                    migration_revision = None
                if connection.dialect.name == "postgresql":
                    extension = connection.scalar(
                        text(
                            "SELECT extversion FROM pg_extension WHERE extname='vector'"
                        )
                    )
                    services["pgvector"] = {
                        "status": "healthy" if extension else "unavailable",
                        "detail": str(extension) if extension else "extension missing",
                    }
                else:
                    services["pgvector"] = {
                        "status": "degraded",
                        "detail": "SQLite compatibility mode",
                    }
        except Exception as exc:  # noqa: BLE001
            services["postgres"] = {
                "status": "unavailable",
                "detail": type(exc).__name__,
            }
            services["pgvector"] = {
                "status": "unavailable",
                "detail": "database unavailable",
            }

        try:
            import redis

            client = redis.from_url(
                os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                socket_connect_timeout=0.5,
                socket_timeout=0.5,
            )
            client.ping()
            services["redis"] = {"status": "healthy", "detail": None}
        except Exception as exc:  # noqa: BLE001
            services["redis"] = {
                "status": "unavailable",
                "detail": type(exc).__name__,
            }

        try:
            store = get_object_store()
            healthy = store.client.bucket_exists("uploads")
            services["minio"] = {
                "status": "healthy" if healthy else "unavailable",
                "detail": None if healthy else "uploads bucket missing",
            }
        except Exception as exc:  # noqa: BLE001
            services["minio"] = {
                "status": "unavailable",
                "detail": type(exc).__name__,
            }

        try:
            from workers.celery_app import celery_app

            replies = celery_app.control.inspect(timeout=0.5).ping() or {}
            services["worker"] = {
                "status": "healthy" if replies else "unavailable",
                "detail": f"{len(replies)} worker(s)" if replies else "no heartbeat",
            }
        except Exception as exc:  # noqa: BLE001
            services["worker"] = {
                "status": "unavailable",
                "detail": type(exc).__name__,
            }

        overall = (
            "healthy"
            if all(item["status"] == "healthy" for item in services.values())
            else "degraded"
        )
        result = {
            "status": overall,
            "api_version": "v2",
            "migration_revision": migration_revision,
            "services": services,
            "story_runtime": os.getenv("STORY_RUNTIME_MODE", "llm"),
            "rag_runtime": os.getenv("RAG_RUNTIME_MODE", "model"),
            "worker_profile": os.getenv("WORKER_PROFILE", "core"),
            "checked_at": datetime.now(timezone.utc),
        }
        self.__class__._cached = result
        self.__class__._cached_at = now_monotonic
        return result
