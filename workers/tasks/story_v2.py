from __future__ import annotations

import os
import time
from typing import Any

from core.application.story_processor import (
    DeterministicStoryTurnProcessor,
    LLMStoryTurnProcessor,
    StoryTurnContext,
)
from core.application.story_service import StoryApplicationService
from core.application.rag_retrieval_service import RagRetrievalService
from core.application.job_service import JobMaintenanceService
from core.application.world_service import WorldApplicationService
from core.llm.runtime import get_runtime_llm
from workers.celery_app import celery_app


def _processor() -> Any:
    mode = os.getenv("STORY_RUNTIME_MODE", "llm").strip().lower()
    if mode in {"mock", "deterministic"}:
        return DeterministicStoryTurnProcessor()
    return LLMStoryTurnProcessor(get_runtime_llm())


@celery_app.task(bind=True, name="story_turn_v2_task")
def story_turn_v2_task(self, job_id: str):
    """Run one durable v2 Story turn and atomically persist its outcome."""
    stories = StoryApplicationService()
    execution_id = str(self.request.id or "")
    try:
        claim = stories.claim_job(job_id, execution_id=execution_id)
        job = claim.job
        if not claim.claimed:
            return job.result or {"job_id": job.id, "status": job.status}

        delay = max(0, int(os.getenv("DEMO_JOB_DELAY_MS", "0"))) / 1000
        if delay:
            time.sleep(delay / 3)

        payload = dict(job.payload or {})
        session = stories.get_session(str(job.session_id))
        if session is None:
            raise RuntimeError("STORY_SESSION_MISSING")
        world = WorldApplicationService().get(session.world_id)
        if world is None:
            raise RuntimeError("STORY_WORLD_MISSING")

        rag_mode = str(payload.get("rag_mode") or "auto")
        citations: list[dict[str, Any]] = []
        rag_error: str | None = None
        if rag_mode != "off":
            try:
                citations = RagRetrievalService().retrieve(
                    world_id=session.world_id,
                    query=str(payload.get("player_input") or ""),
                )
            except Exception as exc:  # noqa: BLE001
                rag_error = f"{type(exc).__name__}: {exc}"
                if rag_mode == "on":
                    raise RuntimeError(f"RAG_RETRIEVAL_FAILED:{rag_error}") from exc
        JobMaintenanceService().update_progress(job_id, 45, execution_id=execution_id)
        if delay:
            time.sleep(delay / 3)

        result = _processor().process(
            StoryTurnContext(
                session_id=session.id,
                player_name=session.player_name,
                world=dict(world.pack or {}),
                state=dict(session.state or {}),
                player_input=str(payload.get("player_input") or ""),
                choice_id=payload.get("choice_id"),
                citations=citations,
            )
        )
        result["trace"] = {
            **dict(result.get("trace") or {}),
            "rag": {
                "mode": rag_mode,
                "hit_count": len(citations),
                "degraded": bool(rag_error),
                "error": rag_error,
            },
        }
        JobMaintenanceService().update_progress(job_id, 80, execution_id=execution_id)
        if delay:
            time.sleep(delay / 3)
        completed = stories.complete_job(job_id, result, execution_id=execution_id)
        return completed.result
    except Exception as exc:  # noqa: BLE001
        try:
            stories.fail_job(
                job_id,
                error_code=type(exc).__name__,
                message=str(exc),
                execution_id=execution_id,
            )
        except Exception:
            pass
        raise
