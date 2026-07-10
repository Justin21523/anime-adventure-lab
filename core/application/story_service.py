from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import uuid
from typing import Any

from sqlalchemy import func, select

from core.persistence.models import (
    JobRecord,
    StorySessionRecord,
    StoryTurnRecord,
    WorldRecord,
    ReviewProposalRecord,
)
from core.persistence.unit_of_work import UnitOfWork
from core.application.job_events import append_job_event


ACTIVE_JOB_STATUSES = ("queued", "running")


def _patch_depth(value: Any, depth: int = 0) -> int:
    if not isinstance(value, dict) or not value:
        return depth
    return max(_patch_depth(child, depth + 1) for child in value.values())


def _safe_world_patch(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not value or "world_id" in value:
        return None
    if _patch_depth(value) > 5:
        return None
    encoded = json.dumps(value, ensure_ascii=False)
    return value if len(encoded.encode("utf-8")) <= 16 * 1024 else None


def _deep_merge(base: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in delta.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class StoryTurnInProgressError(RuntimeError):
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        super().__init__(f"STORY_TURN_IN_PROGRESS:{job_id}")


@dataclass(frozen=True)
class EnqueuedStoryTurn:
    job: JobRecord
    turn: StoryTurnRecord
    replayed: bool


@dataclass(frozen=True)
class ClaimedStoryJob:
    job: JobRecord
    claimed: bool


class StoryApplicationService:
    def __init__(self, uow_factory=UnitOfWork) -> None:
        self.uow_factory = uow_factory

    def create_session(
        self,
        *,
        world_id: str,
        player_name: str,
        persona_id: str | None,
        runtime_preset_id: str | None,
    ) -> StorySessionRecord:
        with self.uow_factory() as uow:
            assert uow.session is not None
            if uow.session.get(WorldRecord, world_id) is None:
                raise LookupError(f"World not found: {world_id}")
            record = StorySessionRecord(
                world_id=world_id,
                player_name=player_name,
                persona_id=persona_id,
                runtime_preset_id=runtime_preset_id,
                state={"turn_count": 0, "flags": {}, "inventory": []},
            )
            uow.session.add(record)
            uow.session.flush()
            return record

    def list_sessions(self) -> list[StorySessionRecord]:
        with self.uow_factory() as uow:
            assert uow.session is not None
            statement = select(StorySessionRecord).order_by(
                StorySessionRecord.updated_at.desc()
            )
            return list(uow.session.scalars(statement))

    def get_session(self, session_id: str) -> StorySessionRecord | None:
        with self.uow_factory() as uow:
            assert uow.session is not None
            return uow.session.get(StorySessionRecord, session_id)

    def enqueue_turn(
        self,
        *,
        session_id: str,
        idempotency_key: str,
        player_input: str,
        choice_id: str | None = None,
        use_agent: bool = False,
        rag_mode: str = "auto",
        include_image: bool = False,
        request_id: str | None = None,
    ) -> EnqueuedStoryTurn:
        """Create the turn and durable job in one transaction.

        A repeated idempotency key returns the original job. A different key is
        rejected while a turn for the same session is queued/running so story
        state can only advance serially.
        """
        with self.uow_factory() as uow:
            assert uow.session is not None
            existing = uow.session.scalar(
                select(JobRecord).where(
                    JobRecord.kind == "story_turn",
                    JobRecord.session_id == session_id,
                    JobRecord.idempotency_key == idempotency_key,
                )
            )
            if existing is not None:
                turn = uow.session.get(StoryTurnRecord, existing.turn_id)
                if turn is None:
                    raise RuntimeError("STORY_TURN_JOB_CORRUPT")
                return EnqueuedStoryTurn(existing, turn, True)

            session = uow.session.scalar(
                select(StorySessionRecord)
                .where(StorySessionRecord.id == session_id)
                .with_for_update()
            )
            if session is None:
                raise LookupError(f"Story session not found: {session_id}")

            active = uow.session.scalar(
                select(JobRecord).where(
                    JobRecord.kind == "story_turn",
                    JobRecord.session_id == session_id,
                    JobRecord.status.in_(ACTIVE_JOB_STATUSES),
                )
            )
            if active is not None:
                raise StoryTurnInProgressError(active.id)

            last_turn = uow.session.scalar(
                select(func.max(StoryTurnRecord.turn_number)).where(
                    StoryTurnRecord.session_id == session_id
                )
            )
            turn = StoryTurnRecord(
                session_id=session_id,
                turn_number=int(last_turn or 0) + 1,
                idempotency_key=idempotency_key,
                player_input=player_input,
                status="queued",
                trace={"choice_id": choice_id, "rag_mode": rag_mode},
            )
            uow.session.add(turn)
            uow.session.flush()

            job = JobRecord(
                kind="story_turn",
                status="queued",
                progress=0,
                session_id=session_id,
                turn_id=turn.id,
                idempotency_key=idempotency_key,
                payload={
                    "session_id": session_id,
                    "turn_id": turn.id,
                    "player_input": player_input,
                    "choice_id": choice_id,
                    "use_agent": use_agent,
                    "rag_mode": rag_mode,
                    "include_image": include_image,
                },
                request_id=request_id,
            )
            uow.session.add(job)
            uow.session.flush()
            append_job_event(
                uow.session,
                job,
                "queued",
                actor="api",
                to_status="queued",
                details={"kind": job.kind},
            )
            return EnqueuedStoryTurn(job, turn, False)

    def get_job(self, job_id: str) -> JobRecord | None:
        with self.uow_factory() as uow:
            assert uow.session is not None
            return uow.session.get(JobRecord, job_id)

    def list_turns(self, session_id: str) -> list[StoryTurnRecord]:
        with self.uow_factory() as uow:
            assert uow.session is not None
            if uow.session.get(StorySessionRecord, session_id) is None:
                raise LookupError(f"Story session not found: {session_id}")
            statement = (
                select(StoryTurnRecord)
                .where(StoryTurnRecord.session_id == session_id)
                .order_by(StoryTurnRecord.turn_number)
            )
            return list(uow.session.scalars(statement))

    def claim_job(
        self, job_id: str, execution_id: str | None = None
    ) -> ClaimedStoryJob:
        with self.uow_factory() as uow:
            assert uow.session is not None
            job = uow.session.scalar(
                select(JobRecord).where(JobRecord.id == job_id).with_for_update()
            )
            if job is None:
                raise LookupError(f"Job not found: {job_id}")
            if job.status == "completed":
                return ClaimedStoryJob(job, False)
            if job.status == "running":
                now = datetime.now(timezone.utc)
                lease = job.lease_expires_at
                if lease is not None and lease.tzinfo is None:
                    lease = lease.replace(tzinfo=timezone.utc)
                if execution_id and job.execution_id == execution_id:
                    return ClaimedStoryJob(job, True)
                if lease is not None and lease > now:
                    return ClaimedStoryJob(job, False)
            if job.status != "queued":
                if job.status != "running":
                    raise RuntimeError(f"JOB_NOT_CLAIMABLE:{job.status}")
            now = datetime.now(timezone.utc)
            previous_status = job.status
            job.status = "running"
            job.progress = 5
            job.attempt_count += 1
            job.execution_id = execution_id or str(uuid.uuid4())
            job.lease_expires_at = now + timedelta(minutes=35)
            job.started_at = now
            job.finished_at = None
            if job.turn_id:
                turn = uow.session.get(StoryTurnRecord, job.turn_id)
                if turn is not None:
                    turn.status = "running"
            append_job_event(
                uow.session,
                job,
                "claimed",
                actor="worker",
                from_status=previous_status,
                to_status="running",
                details={"lease_minutes": 35},
            )
            uow.session.flush()
            return ClaimedStoryJob(job, True)

    def complete_job(
        self, job_id: str, result: dict[str, Any], execution_id: str | None = None
    ) -> JobRecord:
        with self.uow_factory() as uow:
            assert uow.session is not None
            job = uow.session.scalar(
                select(JobRecord).where(JobRecord.id == job_id).with_for_update()
            )
            if job is None:
                raise LookupError(f"Job not found: {job_id}")
            if job.status == "completed":
                return job
            if job.status != "running":
                raise RuntimeError(f"JOB_NOT_RUNNING:{job.status}")
            if execution_id and job.execution_id != execution_id:
                raise RuntimeError("JOB_EXECUTION_SUPERSEDED")
            turn = uow.session.get(StoryTurnRecord, job.turn_id)
            session = uow.session.get(StorySessionRecord, job.session_id)
            if turn is None or session is None:
                raise RuntimeError("STORY_TURN_JOB_CORRUPT")

            previous_status = job.status
            turn.status = "completed"
            turn.narrative = str(result.get("narrative") or "")
            turn.choices = list(result.get("choices") or [])
            turn.citations = list(result.get("citations") or [])
            turn.state_delta = dict(result.get("state_delta") or {})
            turn.trace = {**turn.trace, **dict(result.get("trace") or {})}
            turn.updated_at = datetime.now(timezone.utc)

            state = _deep_merge(dict(session.state or {}), turn.state_delta)
            state["turn_count"] = max(
                int(state.get("turn_count") or 0), turn.turn_number
            )
            state["last_narrative"] = turn.narrative[:500]
            state["last_turn_status"] = "completed"
            session.state = state
            session.version += 1
            session.updated_at = datetime.now(timezone.utc)

            job.status = "completed"
            job.progress = 100
            persisted_result = dict(result)
            patch = _safe_world_patch(result.get("world_patch"))
            if patch:
                proposal = ReviewProposalRecord(
                    world_id=session.world_id,
                    session_id=session.id,
                    status="pending",
                    patch=patch,
                    reasoning=str(result.get("proposal_reasoning") or "")[:4000]
                    or None,
                )
                uow.session.add(proposal)
                uow.session.flush()
                persisted_result["review_proposal_id"] = proposal.id
            elif result.get("world_patch"):
                turn.trace = {**turn.trace, "proposal_error": "WORLD_PATCH_REJECTED"}
                persisted_result["proposal_error"] = "WORLD_PATCH_REJECTED"
            job.result = persisted_result
            job.error_code = None
            job.error_message = None
            job.lease_expires_at = None
            job.finished_at = datetime.now(timezone.utc)
            job.updated_at = datetime.now(timezone.utc)
            append_job_event(
                uow.session,
                job,
                "completed",
                actor="worker",
                from_status=previous_status,
                to_status="completed",
                details={
                    "rag_hit_count": len(turn.citations),
                    "review_proposal_created": bool(
                        persisted_result.get("review_proposal_id")
                    ),
                },
            )
            uow.session.flush()
            return job

    def fail_job(
        self,
        job_id: str,
        *,
        error_code: str,
        message: str,
        execution_id: str | None = None,
    ) -> JobRecord:
        with self.uow_factory() as uow:
            assert uow.session is not None
            job = uow.session.scalar(
                select(JobRecord).where(JobRecord.id == job_id).with_for_update()
            )
            if job is None:
                raise LookupError(f"Job not found: {job_id}")
            if job.status == "completed":
                return job
            if execution_id and job.execution_id != execution_id:
                raise RuntimeError("JOB_EXECUTION_SUPERSEDED")
            previous_status = job.status
            job.status = "failed"
            job.error_code = error_code[:120]
            job.error_message = message[:4000]
            job.lease_expires_at = None
            job.finished_at = datetime.now(timezone.utc)
            job.updated_at = datetime.now(timezone.utc)
            if job.turn_id:
                turn = uow.session.get(StoryTurnRecord, job.turn_id)
                if turn is not None:
                    turn.status = "failed"
                    turn.updated_at = datetime.now(timezone.utc)
            session = (
                uow.session.get(StorySessionRecord, job.session_id)
                if job.session_id
                else None
            )
            if session is not None:
                session.state = {
                    **dict(session.state or {}),
                    "last_turn_status": "failed",
                }
            append_job_event(
                uow.session,
                job,
                "failed",
                actor="worker",
                from_status=previous_status,
                to_status="failed",
                details={"error_code": job.error_code},
            )
            uow.session.flush()
            return job
