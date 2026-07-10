from __future__ import annotations

from core.application.job_service import JobMaintenanceService
from workers.celery_app import celery_app


def dispatch_v2_job(job_id: str, kind: str) -> None:
    if kind == "story_turn":
        from workers.tasks.story_v2 import story_turn_v2_task

        story_turn_v2_task.delay(job_id)
    elif kind == "document_index":
        from workers.tasks.rag_v2 import document_index_v2_task

        document_index_v2_task.delay(job_id)
    else:
        raise RuntimeError(f"UNSUPPORTED_JOB_KIND:{kind}")


@celery_app.task(name="reconcile_v2_jobs_task")
def reconcile_v2_jobs_task():
    service = JobMaintenanceService()
    result = service.reconcile()
    dispatched: list[str] = []
    deferred: list[str] = []
    for job_id, kind in result.requeued:
        try:
            dispatch_v2_job(job_id, kind)
            service.mark_dispatched(job_id)
            dispatched.append(job_id)
        except Exception:  # noqa: BLE001
            service.mark_dispatched(job_id, deferred=True)
            deferred.append(job_id)
    return {
        "dispatched": dispatched,
        "deferred": deferred,
        "failed": result.failed,
    }
