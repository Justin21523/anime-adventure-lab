from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from core.persistence.models import (
    ReviewProposalRecord,
    StorySessionRecord,
    WorldRecord,
)
from core.persistence.unit_of_work import UnitOfWork


def _merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


@dataclass(frozen=True)
class ApprovedProposal:
    proposal: ReviewProposalRecord
    world: WorldRecord


class ReviewApplicationService:
    def __init__(self, uow_factory=UnitOfWork) -> None:
        self.uow_factory = uow_factory

    def create(
        self,
        *,
        world_id: str,
        session_id: str | None,
        patch: dict[str, Any],
        reasoning: str | None,
    ) -> ReviewProposalRecord:
        with self.uow_factory() as uow:
            assert uow.session is not None
            if uow.session.get(WorldRecord, world_id) is None:
                raise LookupError(f"World not found: {world_id}")
            if session_id:
                session = uow.session.get(StorySessionRecord, session_id)
                if session is None or session.world_id != world_id:
                    raise LookupError(f"Story session not found in world: {session_id}")
            proposal = ReviewProposalRecord(
                world_id=world_id,
                session_id=session_id,
                status="pending",
                patch=patch,
                reasoning=reasoning,
            )
            uow.session.add(proposal)
            uow.session.flush()
            return proposal

    def list(
        self,
        *,
        status: str | None = None,
        world_id: str | None = None,
        session_id: str | None = None,
    ) -> list[ReviewProposalRecord]:
        with self.uow_factory() as uow:
            assert uow.session is not None
            statement = select(ReviewProposalRecord).order_by(
                ReviewProposalRecord.created_at.desc()
            )
            if status:
                statement = statement.where(ReviewProposalRecord.status == status)
            if world_id:
                statement = statement.where(ReviewProposalRecord.world_id == world_id)
            if session_id:
                statement = statement.where(
                    ReviewProposalRecord.session_id == session_id
                )
            return list(uow.session.scalars(statement))

    def approve(
        self, proposal_id: str, expected_world_version: int
    ) -> ApprovedProposal:
        with self.uow_factory() as uow:
            assert uow.session is not None
            proposal = uow.session.scalar(
                select(ReviewProposalRecord)
                .where(ReviewProposalRecord.id == proposal_id)
                .with_for_update()
            )
            if proposal is None:
                raise LookupError(f"Review proposal not found: {proposal_id}")
            if proposal.status != "pending":
                raise RuntimeError(f"PROPOSAL_ALREADY_REVIEWED:{proposal.status}")
            world = uow.session.scalar(
                select(WorldRecord)
                .where(WorldRecord.id == proposal.world_id)
                .with_for_update()
            )
            if world is None:
                raise RuntimeError("PROPOSAL_WORLD_MISSING")
            if world.version != expected_world_version:
                raise RuntimeError("WORLD_VERSION_CONFLICT")
            world.pack = _merge(dict(world.pack or {}), dict(proposal.patch or {}))
            world.name = str(world.pack.get("name") or world.name)
            world.version += 1
            now = datetime.now(timezone.utc)
            world.updated_at = now
            proposal.status = "approved"
            proposal.reviewed_at = now
            uow.session.flush()
            return ApprovedProposal(proposal, world)

    def reject(self, proposal_id: str) -> ReviewProposalRecord:
        with self.uow_factory() as uow:
            assert uow.session is not None
            proposal = uow.session.scalar(
                select(ReviewProposalRecord)
                .where(ReviewProposalRecord.id == proposal_id)
                .with_for_update()
            )
            if proposal is None:
                raise LookupError(f"Review proposal not found: {proposal_id}")
            if proposal.status != "pending":
                raise RuntimeError(f"PROPOSAL_ALREADY_REVIEWED:{proposal.status}")
            proposal.status = "rejected"
            proposal.reviewed_at = datetime.now(timezone.utc)
            uow.session.flush()
            return proposal
