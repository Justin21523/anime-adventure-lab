from __future__ import annotations

import pytest

from core.application import (
    DocumentApplicationService,
    ReviewApplicationService,
    StoryApplicationService,
    StoryTurnInProgressError,
    WorldApplicationService,
    JobMaintenanceService,
    JobEventService,
    RagRetrievalService,
)
from core.persistence.database import Database
from core.persistence.models import (
    Base,
    DocumentChunkRecord,
    JobRecord,
    JobEventRecord,
    ReviewProposalRecord,
)
from core.persistence.unit_of_work import UnitOfWork
from core.migration import import_legacy_data


@pytest.fixture()
def uow_factory(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'application.db'}")
    Base.metadata.create_all(database.engine)
    return lambda: UnitOfWork(database)


def test_world_and_story_session_are_transactional(uow_factory):
    worlds = WorldApplicationService(uow_factory)
    stories = StoryApplicationService(uow_factory)

    world = worlds.create("neon_archive", "霓虹檔案城", {"setting": "cyberpunk"})
    story = stories.create_session(
        world_id=world.id,
        player_name="凜",
        persona_id="wise_sage",
        runtime_preset_id=None,
    )

    assert worlds.get("neon_archive").pack["setting"] == "cyberpunk"
    assert stories.get_session(story.id).state["turn_count"] == 0


def test_world_update_rejects_stale_version(uow_factory):
    worlds = WorldApplicationService(uow_factory)
    world = worlds.create("moon_archive", "月蝕檔案館", {})
    updated = worlds.update(world.id, world.version, {"name": "月蝕檔案館 v2"})

    assert updated.version == 2
    with pytest.raises(RuntimeError, match="WORLD_VERSION_CONFLICT"):
        worlds.update(world.id, 1, {"name": "stale"})


def test_legacy_import_supports_dry_run_and_idempotent_apply(uow_factory, tmp_path):
    worlds_dir = tmp_path / "worlds"
    sessions_dir = tmp_path / "sessions"
    worlds_dir.mkdir()
    sessions_dir.mkdir()
    (worlds_dir / "default.json").write_text(
        '{"world_id":"default","name":"Default"}', encoding="utf-8"
    )
    (sessions_dir / "legacy.json").write_text(
        '{"session_id":"legacy","world_id":"default","player_name":"Rin","current_state":{}}',
        encoding="utf-8",
    )

    dry = import_legacy_data(
        worldpacks_dir=worlds_dir,
        sessions_dir=sessions_dir,
        dry_run=True,
        uow_factory=uow_factory,
    )
    applied = import_legacy_data(
        worldpacks_dir=worlds_dir,
        sessions_dir=sessions_dir,
        dry_run=False,
        uow_factory=uow_factory,
    )
    repeated = import_legacy_data(
        worldpacks_dir=worlds_dir,
        sessions_dir=sessions_dir,
        dry_run=False,
        uow_factory=uow_factory,
    )

    assert (dry.worlds_imported, dry.sessions_imported) == (1, 0)
    assert (applied.worlds_imported, applied.sessions_imported) == (1, 1)
    assert {item["reason"] for item in repeated.skipped} == {
        "world_exists",
        "session_exists",
    }


def test_story_turn_job_is_idempotent_serial_and_transactional(uow_factory):
    worlds = WorldApplicationService(uow_factory)
    stories = StoryApplicationService(uow_factory)
    worlds.create("clockwork", "發條城", {"name": "發條城"})
    session = stories.create_session(
        world_id="clockwork",
        player_name="凜",
        persona_id=None,
        runtime_preset_id=None,
    )

    first = stories.enqueue_turn(
        session_id=session.id,
        idempotency_key="turn-0001",
        player_input="調查鐘樓",
    )
    replay = stories.enqueue_turn(
        session_id=session.id,
        idempotency_key="turn-0001",
        player_input="這段內容不得被重算",
    )

    assert replay.replayed is True
    assert replay.job.id == first.job.id
    assert replay.turn.player_input == "調查鐘樓"
    with pytest.raises(StoryTurnInProgressError):
        stories.enqueue_turn(
            session_id=session.id,
            idempotency_key="turn-0002",
            player_input="同時送出另一回合",
        )

    claim = stories.claim_job(first.job.id, execution_id="celery-story-1")
    assert claim.claimed is True
    redelivery = stories.claim_job(first.job.id, execution_id="celery-story-1")
    duplicate_claim = stories.claim_job(first.job.id, execution_id="celery-story-2")
    assert redelivery.claimed is True
    assert redelivery.job.attempt_count == 1
    assert duplicate_claim.claimed is False
    stories.complete_job(
        first.job.id,
        {
            "narrative": "鐘聲揭開了密道。",
            "choices": [{"id": "enter", "text": "進入密道"}],
            "state_delta": {"tower_open": True},
            "trace": {"processor": "test"},
        },
    )

    completed = stories.get_job(first.job.id)
    persisted_session = stories.get_session(session.id)
    turns = stories.list_turns(session.id)
    assert completed.status == "completed"
    assert persisted_session.state["turn_count"] == 1
    assert persisted_session.state["tower_open"] is True
    assert turns[0].narrative == "鐘聲揭開了密道。"
    assert [
        event.event_type for event in JobEventService(uow_factory).list(first.job.id)
    ] == [
        "queued",
        "claimed",
        "completed",
    ]


def test_failed_turn_releases_session_for_retry(uow_factory):
    worlds = WorldApplicationService(uow_factory)
    stories = StoryApplicationService(uow_factory)
    worlds.create("retry-world", "重試世界", {})
    session = stories.create_session(
        world_id="retry-world",
        player_name="Rin",
        persona_id=None,
        runtime_preset_id=None,
    )
    enqueued = stories.enqueue_turn(
        session_id=session.id,
        idempotency_key="attempt-1",
        player_input="first",
    )
    stories.claim_job(enqueued.job.id)
    stories.fail_job(enqueued.job.id, error_code="MODEL_DOWN", message="offline")

    retry = stories.enqueue_turn(
        session_id=session.id,
        idempotency_key="attempt-2",
        player_input="retry",
    )
    assert retry.turn.turn_number == 2
    assert stories.get_job(enqueued.job.id).status == "failed"


def test_document_index_job_is_durable_and_idempotent(uow_factory):
    worlds = WorldApplicationService(uow_factory)
    documents = DocumentApplicationService(uow_factory)
    worlds.create("lore", "Lore", {})

    registered = documents.register(
        world_id="lore",
        filename="lore.txt",
        object_key="worlds/lore/documents/checksum/lore.txt",
        content_type="text/plain",
        checksum="a" * 64,
        size_bytes=42,
    )
    replayed = documents.register(
        world_id="lore",
        filename="renamed.txt",
        object_key="worlds/lore/documents/checksum/renamed.txt",
        content_type="text/plain",
        checksum="a" * 64,
        size_bytes=42,
    )
    job, document, claimed = documents.claim_job(registered.job.id)
    documents.complete_job(
        job.id,
        [{"content": "Moon archive", "embedding": [0.0] * 1024}],
    )

    assert replayed.replayed is True
    assert replayed.document.id == registered.document.id
    assert claimed is True
    assert document.status == "indexing"
    assert documents.list("lore")[0].status == "ready"
    with uow_factory() as uow:
        assert uow.session.query(DocumentChunkRecord).count() == 1


def test_deterministic_embedding_has_pgvector_dimension():
    from workers.tasks.rag_v2 import _deterministic_embedding

    vector = _deterministic_embedding("月蝕 archive 月蝕")
    assert len(vector) == 1024
    assert sum(value * value for value in vector) == pytest.approx(1.0)


def test_review_proposal_applies_only_after_version_checked_approval(uow_factory):
    worlds = WorldApplicationService(uow_factory)
    reviews = ReviewApplicationService(uow_factory)
    world = worlds.create(
        "review-world",
        "Review World",
        {"name": "Review World", "flags": {"gate": False}},
    )
    proposal = reviews.create(
        world_id=world.id,
        session_id=None,
        patch={"flags": {"gate": True}},
        reasoning="The completed story turn unlocked the gate.",
    )

    assert worlds.get(world.id).pack["flags"]["gate"] is False
    with pytest.raises(RuntimeError, match="WORLD_VERSION_CONFLICT"):
        reviews.approve(proposal.id, expected_world_version=99)

    approved = reviews.approve(proposal.id, expected_world_version=1)
    assert approved.proposal.status == "approved"
    assert approved.world.version == 2
    assert approved.world.pack["flags"]["gate"] is True
    with pytest.raises(RuntimeError, match="PROPOSAL_ALREADY_REVIEWED"):
        reviews.reject(proposal.id)


def test_rag_retrieval_is_world_scoped_and_returns_evidence(uow_factory, monkeypatch):
    from core.application.embedding_service import deterministic_embedding

    monkeypatch.setenv("RAG_RUNTIME_MODE", "deterministic")
    worlds = WorldApplicationService(uow_factory)
    documents = DocumentApplicationService(uow_factory)
    worlds.create("moon", "Moon", {})
    worlds.create("secret", "Secret", {})
    for world_id, checksum, content in [
        ("moon", "b" * 64, "銀色檔案盒只能在月蝕鐘響後開啟"),
        ("secret", "c" * 64, "這是另一個世界不可洩漏的秘密"),
    ]:
        registered = documents.register(
            world_id=world_id,
            filename=f"{world_id}.md",
            object_key=f"worlds/{world_id}/documents/{checksum}/lore.md",
            content_type="text/markdown",
            checksum=checksum,
            size_bytes=len(content),
        )
        job, _, _ = documents.claim_job(registered.job.id)
        documents.complete_job(
            job.id,
            [{"content": content, "embedding": deterministic_embedding(content)}],
        )

    hits = RagRetrievalService(uow_factory).retrieve(
        world_id="moon", query="銀色檔案盒如何開啟？"
    )

    assert hits[0]["filename"] == "moon.md"
    assert "月蝕鐘響" in hits[0]["excerpt"]
    assert all(hit["document_id"] for hit in hits)
    assert all("不可洩漏" not in hit["excerpt"] for hit in hits)


def test_story_completion_creates_review_proposal(uow_factory):
    worlds = WorldApplicationService(uow_factory)
    stories = StoryApplicationService(uow_factory)
    worlds.create("proposal-world", "Proposal", {})
    session = stories.create_session(
        world_id="proposal-world",
        player_name="Rin",
        persona_id=None,
        runtime_preset_id=None,
    )
    enqueued = stories.enqueue_turn(
        session_id=session.id,
        idempotency_key="proposal-turn",
        player_input="discover",
    )
    stories.claim_job(enqueued.job.id, execution_id="proposal-execution")
    completed = stories.complete_job(
        enqueued.job.id,
        {
            "narrative": "A discovery.",
            "world_patch": {"discoveries": {"gate": "open"}},
            "proposal_reasoning": "Observed in story.",
        },
        execution_id="proposal-execution",
    )

    with uow_factory() as uow:
        proposals = list(uow.session.query(ReviewProposalRecord))
    assert completed.result["review_proposal_id"] == proposals[0].id
    assert proposals[0].status == "pending"


def test_reconciler_requeues_expired_jobs_and_exhausts_attempts(
    uow_factory, monkeypatch
):
    from datetime import datetime, timedelta, timezone

    monkeypatch.setenv("JOB_MAX_ATTEMPTS", "3")
    worlds = WorldApplicationService(uow_factory)
    stories = StoryApplicationService(uow_factory)
    worlds.create("recover", "Recover", {})
    session = stories.create_session(
        world_id="recover", player_name="Rin", persona_id=None, runtime_preset_id=None
    )
    first = stories.enqueue_turn(
        session_id=session.id, idempotency_key="recover-1", player_input="go"
    )
    stories.claim_job(first.job.id, execution_id="old-worker")
    with uow_factory() as uow:
        job = uow.session.get(JobRecord, first.job.id)
        job.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    result = JobMaintenanceService(uow_factory).reconcile()
    assert result.requeued == [(first.job.id, "story_turn")]
    assert stories.get_job(first.job.id).status == "queued"

    with uow_factory() as uow:
        job = uow.session.get(JobRecord, first.job.id)
        job.status = "running"
        job.attempt_count = 3
        job.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    exhausted = JobMaintenanceService(uow_factory).reconcile()
    assert exhausted.failed == [first.job.id]
    assert stories.get_job(first.job.id).error_code == "JOB_RETRY_EXHAUSTED"
    assert [
        event.event_type for event in JobEventService(uow_factory).list(first.job.id)
    ][-2:] == ["lease_reconciled", "retry_exhausted"]


def test_job_event_details_are_sanitized(uow_factory):
    from core.application.job_events import append_job_event

    worlds = WorldApplicationService(uow_factory)
    stories = StoryApplicationService(uow_factory)
    worlds.create("audit", "Audit", {})
    session = stories.create_session(
        world_id="audit", player_name="Rin", persona_id=None, runtime_preset_id=None
    )
    enqueued = stories.enqueue_turn(
        session_id=session.id, idempotency_key="audit-1", player_input="go"
    )
    with uow_factory() as uow:
        job = uow.session.get(JobRecord, enqueued.job.id)
        append_job_event(
            uow.session,
            job,
            "diagnostic",
            actor="admin",
            details={"password": "unsafe", "nested": {"api_key": "unsafe"}},
        )

    diagnostic = JobEventService(uow_factory).list(enqueued.job.id)[-1]
    assert diagnostic.details == {
        "password": "[redacted]",
        "nested": {"api_key": "[redacted]"},
    }


def test_superseded_worker_cannot_emit_terminal_event(uow_factory):
    worlds = WorldApplicationService(uow_factory)
    stories = StoryApplicationService(uow_factory)
    worlds.create("fencing", "Fencing", {})
    session = stories.create_session(
        world_id="fencing", player_name="Rin", persona_id=None, runtime_preset_id=None
    )
    enqueued = stories.enqueue_turn(
        session_id=session.id, idempotency_key="fence-1", player_input="go"
    )
    stories.claim_job(enqueued.job.id, execution_id="current-worker")

    with pytest.raises(RuntimeError, match="JOB_EXECUTION_SUPERSEDED"):
        stories.complete_job(
            enqueued.job.id,
            {"narrative": "stale result"},
            execution_id="stale-worker",
        )

    with uow_factory() as uow:
        event_types = [
            event.event_type
            for event in uow.session.query(JobEventRecord)
            .filter(JobEventRecord.job_id == enqueued.job.id)
            .order_by(JobEventRecord.occurred_at, JobEventRecord.id)
        ]
    assert event_types == ["queued", "claimed"]
