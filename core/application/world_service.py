from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from core.persistence.models import WorldRecord
from core.persistence.unit_of_work import UnitOfWork


class WorldApplicationService:
    def __init__(self, uow_factory=UnitOfWork) -> None:
        self.uow_factory = uow_factory

    def create(self, world_id: str, name: str, pack: dict[str, Any]) -> WorldRecord:
        with self.uow_factory() as uow:
            assert uow.session is not None
            if uow.session.get(WorldRecord, world_id) is not None:
                raise ValueError(f"World already exists: {world_id}")
            record = WorldRecord(id=world_id, name=name, pack=pack)
            uow.session.add(record)
            uow.session.flush()
            return record

    def list(self) -> list[WorldRecord]:
        with self.uow_factory() as uow:
            assert uow.session is not None
            return list(
                uow.session.scalars(
                    select(WorldRecord).order_by(WorldRecord.updated_at.desc())
                )
            )

    def get(self, world_id: str) -> WorldRecord | None:
        with self.uow_factory() as uow:
            assert uow.session is not None
            return uow.session.get(WorldRecord, world_id)

    def update(
        self, world_id: str, expected_version: int, pack: dict[str, Any]
    ) -> WorldRecord:
        with self.uow_factory() as uow:
            assert uow.session is not None
            record = uow.session.get(WorldRecord, world_id)
            if record is None:
                raise LookupError(world_id)
            if record.version != expected_version:
                raise RuntimeError("WORLD_VERSION_CONFLICT")
            record.pack = pack
            record.name = str(pack.get("name") or record.name)
            record.version += 1
            record.updated_at = datetime.now(timezone.utc)
            uow.session.flush()
            return record
