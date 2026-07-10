from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import v2
from core.application import (
    DocumentApplicationService,
    StoryApplicationService,
    WorldApplicationService,
    JobMaintenanceService,
    JobEventService,
)
from core.persistence.database import Database
from core.persistence.models import Base
from core.persistence.unit_of_work import UnitOfWork


def test_v2_turn_endpoint_requires_key_and_replays(tmp_path, monkeypatch):
    database = Database(f"sqlite:///{tmp_path / 'v2.db'}")
    Base.metadata.create_all(database.engine)

    def uow_factory():
        return UnitOfWork(database)

    worlds = WorldApplicationService(uow_factory)
    stories = StoryApplicationService(uow_factory)
    documents = DocumentApplicationService(uow_factory)
    worlds.create("demo", "Demo", {})
    session = stories.create_session(
        world_id="demo",
        player_name="Rin",
        persona_id=None,
        runtime_preset_id=None,
    )

    monkeypatch.setattr(v2, "StoryApplicationService", lambda: stories)
    monkeypatch.setattr(v2, "DocumentApplicationService", lambda: documents)
    monkeypatch.setattr(v2, "WorldApplicationService", lambda: worlds)
    dispatched: list[str] = []
    monkeypatch.setattr(v2, "_dispatch_story_turn", dispatched.append)
    app = FastAPI()
    app.include_router(v2.router, prefix="/api/v2")
    client = TestClient(app)
    url = f"/api/v2/story-sessions/{session.id}/turns"

    missing = client.post(url, json={"player_input": "open the door"})
    created = client.post(
        url,
        headers={"Idempotency-Key": "demo-turn-001"},
        json={"player_input": "open the door"},
    )
    replayed = client.post(
        url,
        headers={"Idempotency-Key": "demo-turn-001"},
        json={"player_input": "must not replace original"},
    )

    assert missing.status_code == 422
    assert created.status_code == 202
    assert created.headers["location"].endswith(created.json()["job_id"])
    assert replayed.status_code == 202
    assert replayed.headers["idempotent-replayed"] == "true"
    assert created.json()["job_id"] == replayed.json()["job_id"]
    assert dispatched == [created.json()["job_id"]]

    world = client.get("/api/v2/worlds/demo")
    updated = client.put(
        "/api/v2/worlds/demo",
        headers={"If-Match": world.headers["etag"]},
        json={"name": "Demo v2", "pack": {"setting": "archive"}},
    )
    stale = client.put(
        "/api/v2/worlds/demo",
        headers={"If-Match": world.headers["etag"]},
        json={"name": "stale", "pack": {}},
    )

    assert updated.status_code == 200
    assert updated.headers["etag"] == '"2"'
    assert stale.status_code == 412
    assert stale.json()["detail"]["code"] == "WORLD_VERSION_CONFLICT"


def test_v2_document_upload_persists_job_before_dispatch(tmp_path, monkeypatch):
    database = Database(f"sqlite:///{tmp_path / 'documents.db'}")
    Base.metadata.create_all(database.engine)

    def uow_factory():
        return UnitOfWork(database)

    worlds = WorldApplicationService(uow_factory)
    documents = DocumentApplicationService(uow_factory)
    worlds.create("demo", "Demo", {})

    class Store:
        uploads: list[tuple[str, str, bytes, str]] = []

        def put_bytes(self, bucket, key, payload, content_type):
            self.uploads.append((bucket, key, payload, content_type))

        def delete(self, bucket, key):
            return None

    store = Store()
    dispatched: list[str] = []
    monkeypatch.setattr(v2, "DocumentApplicationService", lambda: documents)
    monkeypatch.setattr(v2, "get_object_store", lambda: store)
    monkeypatch.setattr(v2, "_dispatch_document_index", dispatched.append)
    app = FastAPI()
    app.include_router(v2.router, prefix="/api/v2")
    client = TestClient(app)

    response = client.post(
        "/api/v2/worlds/demo/documents",
        files={"file": ("lore.md", b"# Moon Archive", "text/markdown")},
    )

    assert response.status_code == 202
    assert response.json()["document"]["status"] == "queued"
    assert response.json()["job"]["status"] == "queued"
    assert len(store.uploads) == 1
    assert dispatched == [response.json()["job"]["job_id"]]


def test_v2_job_list_retry_and_system_status(tmp_path, monkeypatch):
    database = Database(f"sqlite:///{tmp_path / 'jobs.db'}")
    Base.metadata.create_all(database.engine)

    def uow_factory():
        return UnitOfWork(database)

    worlds = WorldApplicationService(uow_factory)
    stories = StoryApplicationService(uow_factory)
    jobs = JobMaintenanceService(uow_factory)
    events = JobEventService(uow_factory)
    worlds.create("demo", "Demo", {})
    session = stories.create_session(
        world_id="demo", player_name="Rin", persona_id=None, runtime_preset_id=None
    )
    enqueued = stories.enqueue_turn(
        session_id=session.id,
        idempotency_key="failed-turn",
        player_input="go",
        request_id="request-demo",
    )
    stories.claim_job(enqueued.job.id, execution_id="worker-old")
    stories.fail_job(
        enqueued.job.id,
        error_code="MODEL_DOWN",
        message="offline",
        execution_id="worker-old",
    )

    monkeypatch.setattr(v2, "StoryApplicationService", lambda: stories)
    monkeypatch.setattr(v2, "JobMaintenanceService", lambda: jobs)
    monkeypatch.setattr(v2, "JobEventService", lambda: events)
    monkeypatch.setattr(v2, "_dispatch_job", lambda *_args: None)

    class Status:
        def check(self):
            return {
                "status": "degraded",
                "api_version": "v2",
                "migration_revision": None,
                "services": {
                    "worker": {"status": "unavailable", "detail": "no heartbeat"}
                },
                "story_runtime": "deterministic",
                "rag_runtime": "deterministic",
                "worker_profile": "core",
                "checked_at": "2026-07-10T00:00:00Z",
            }

    monkeypatch.setattr(v2, "SystemStatusService", Status)
    app = FastAPI()
    app.include_router(v2.router, prefix="/api/v2")
    client = TestClient(app)

    listed = client.get("/api/v2/jobs?status=failed")
    retried = client.post(f"/api/v2/jobs/{enqueued.job.id}/retry")
    status = client.get("/api/v2/system/status")
    event_log = client.get(f"/api/v2/jobs/{enqueued.job.id}/events")

    assert listed.json()[0]["request_id"] == "request-demo"
    assert listed.json()[0]["error"]["code"] == "MODEL_DOWN"
    assert retried.status_code == 200
    assert retried.json()["status"] == "queued"
    assert status.json()["services"]["worker"]["status"] == "unavailable"
    assert [event["event_type"] for event in event_log.json()] == [
        "queued",
        "claimed",
        "failed",
        "retry_requested",
        "dispatch_succeeded",
    ]
    assert event_log.json()[0]["request_id"] == "request-demo"
