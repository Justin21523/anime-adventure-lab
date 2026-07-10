from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.persistence.models import StorySessionRecord, WorldRecord
from core.persistence.unit_of_work import UnitOfWork


@dataclass
class LegacyImportReport:
    dry_run: bool
    worlds_imported: int = 0
    sessions_imported: int = 0
    skipped: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object")
    return payload


def import_legacy_data(
    *,
    worldpacks_dir: Path,
    sessions_dir: Path,
    dry_run: bool = True,
    uow_factory=UnitOfWork,
) -> LegacyImportReport:
    """Import legacy JSON once; existing IDs are skipped, never overwritten."""

    report = LegacyImportReport(dry_run=dry_run)
    with uow_factory() as uow:
        assert uow.session is not None

        for path in (
            sorted(worldpacks_dir.glob("*.json")) if worldpacks_dir.exists() else []
        ):
            try:
                payload = _read_object(path)
                world_id = str(payload.get("world_id") or path.stem).strip()
                if not world_id:
                    raise ValueError("world_id is missing")
                if uow.session.get(WorldRecord, world_id) is not None:
                    report.skipped.append({"path": str(path), "reason": "world_exists"})
                    continue
                report.worlds_imported += 1
                if not dry_run:
                    uow.session.add(
                        WorldRecord(
                            id=world_id,
                            name=str(payload.get("name") or world_id),
                            pack=payload,
                        )
                    )
            except Exception as exc:
                report.skipped.append({"path": str(path), "reason": str(exc)})

        if not dry_run:
            uow.session.flush()

        for path in (
            sorted(sessions_dir.glob("*.json")) if sessions_dir.exists() else []
        ):
            if path.name.endswith(".context.json"):
                continue
            try:
                payload = _read_object(path)
                session_id = str(payload.get("session_id") or path.stem).strip()
                world_id = str(payload.get("world_id") or "default").strip()
                if uow.session.get(StorySessionRecord, session_id) is not None:
                    report.skipped.append(
                        {"path": str(path), "reason": "session_exists"}
                    )
                    continue
                if uow.session.get(WorldRecord, world_id) is None:
                    report.skipped.append(
                        {"path": str(path), "reason": f"missing_world:{world_id}"}
                    )
                    continue
                report.sessions_imported += 1
                if not dry_run:
                    state = (
                        payload.get("current_state")
                        if isinstance(payload.get("current_state"), dict)
                        else {}
                    )
                    state = {**state, "legacy_history": payload.get("history", [])}
                    uow.session.add(
                        StorySessionRecord(
                            id=session_id,
                            world_id=world_id,
                            player_name=str(payload.get("player_name") or "Player"),
                            persona_id=payload.get("persona_id"),
                            runtime_preset_id=state.get("story_context", {}).get(
                                "runtime_preset_id"
                            ),
                            state=state,
                        )
                    )
            except Exception as exc:
                report.skipped.append({"path": str(path), "reason": str(exc)})

        if dry_run:
            uow.rollback()

    return report
