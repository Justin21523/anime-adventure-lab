#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sqlalchemy import delete, select

from core.application import (
    DocumentApplicationService,
    StoryApplicationService,
    WorldApplicationService,
)
from core.application.embedding_service import deterministic_embedding
from core.persistence.models import (
    ArtifactRecord,
    DocumentChunkRecord,
    DocumentRecord,
    JobRecord,
    ReviewProposalRecord,
    StorySessionRecord,
    StoryTurnRecord,
    WorldRecord,
)
from core.persistence.unit_of_work import UnitOfWork
from core.storage.object_store import get_object_store


WORLD_ID = "moon-archive"
LORE_PATH = Path(__file__).resolve().parents[1] / "docs/demo-data/moon-archive-lore.md"


def reset_demo() -> dict[str, int]:
    deleted: dict[str, int] = {}
    object_keys: list[str] = []
    with UnitOfWork() as uow:
        assert uow.session is not None
        session_ids = list(
            uow.session.scalars(
                select(StorySessionRecord.id).where(
                    StorySessionRecord.world_id == WORLD_ID
                )
            )
        )
        document_ids = list(
            uow.session.scalars(
                select(DocumentRecord.id).where(DocumentRecord.world_id == WORLD_ID)
            )
        )
        object_keys = list(
            uow.session.scalars(
                select(DocumentRecord.object_key).where(
                    DocumentRecord.world_id == WORLD_ID
                )
            )
        )
        statements = [
            (
                "artifacts",
                (
                    delete(ArtifactRecord).where(
                        ArtifactRecord.session_id.in_(session_ids)
                    )
                    if session_ids
                    else None
                ),
            ),
            (
                "jobs",
                (
                    delete(JobRecord).where(
                        (JobRecord.session_id.in_(session_ids))
                        | (JobRecord.document_id.in_(document_ids))
                    )
                    if session_ids or document_ids
                    else None
                ),
            ),
            (
                "turns",
                (
                    delete(StoryTurnRecord).where(
                        StoryTurnRecord.session_id.in_(session_ids)
                    )
                    if session_ids
                    else None
                ),
            ),
            (
                "proposals",
                delete(ReviewProposalRecord).where(
                    ReviewProposalRecord.world_id == WORLD_ID
                ),
            ),
            (
                "chunks",
                delete(DocumentChunkRecord).where(
                    DocumentChunkRecord.world_id == WORLD_ID
                ),
            ),
            (
                "documents",
                delete(DocumentRecord).where(DocumentRecord.world_id == WORLD_ID),
            ),
            (
                "sessions",
                delete(StorySessionRecord).where(
                    StorySessionRecord.world_id == WORLD_ID
                ),
            ),
            ("worlds", delete(WorldRecord).where(WorldRecord.id == WORLD_ID)),
        ]
        for name, statement in statements:
            deleted[name] = (
                int(uow.session.execute(statement).rowcount or 0)
                if statement is not None
                else 0
            )
    try:
        store = get_object_store()
        for key in object_keys:
            store.delete("uploads", key)
    except Exception:
        pass
    return deleted


def seed_demo() -> dict[str, str]:
    worlds = WorldApplicationService()
    stories = StoryApplicationService()
    documents = DocumentApplicationService()
    world = worlds.get(WORLD_ID)
    if world is None:
        world = worlds.create(
            WORLD_ID,
            "月蝕檔案館",
            {
                "name": "月蝕檔案館",
                "setting": "mystery fantasy",
                "description": "保存失落記憶、且所有 AI 變更都需人工核准的月下檔案館。",
                "rules": {"human_review_required": True},
            },
        )
    sessions = [item for item in stories.list_sessions() if item.world_id == WORLD_ID]
    session = (
        sessions[0]
        if sessions
        else stories.create_session(
            world_id=WORLD_ID,
            player_name="凜",
            persona_id="wise_sage",
            runtime_preset_id="portfolio-demo",
        )
    )

    payload = LORE_PATH.read_bytes()
    checksum = hashlib.sha256(payload).hexdigest()
    object_key = f"worlds/{WORLD_ID}/documents/{checksum}/{LORE_PATH.name}"
    store = get_object_store()
    store.put_bytes("uploads", object_key, payload, "text/markdown")
    registered = documents.register(
        world_id=WORLD_ID,
        filename=LORE_PATH.name,
        object_key=object_key,
        content_type="text/markdown",
        checksum=checksum,
        size_bytes=len(payload),
        request_id="demo-seed",
    )
    if registered.document.status != "ready":
        job, _, claimed = documents.claim_job(
            registered.job.id, execution_id="demo-seed"
        )
        if claimed:
            text = payload.decode("utf-8")
            documents.complete_job(
                job.id,
                [
                    {
                        "content": text,
                        "embedding": deterministic_embedding(text),
                        "metadata": {"seed": True},
                    }
                ],
                execution_id="demo-seed",
            )
    return {"world_id": world.id, "session_id": session.id, "document": LORE_PATH.name}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seed the isolated SagaForge portfolio demo"
    )
    parser.add_argument(
        "--apply", action="store_true", help="Apply changes; default is dry-run"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete only the moon-archive demo namespace first",
    )
    args = parser.parse_args()
    if not args.apply:
        print(
            json.dumps(
                {"dry_run": True, "world_id": WORLD_ID, "reset": args.reset},
                ensure_ascii=False,
            )
        )
        return 0
    result: dict[str, object] = {"dry_run": False}
    if args.reset:
        result["deleted"] = reset_demo()
    result["seeded"] = seed_demo()
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
