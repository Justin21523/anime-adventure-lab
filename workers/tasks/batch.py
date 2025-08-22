from workers.celery_app import celery_app
@celery_app.task
def batch_job(job_id: str) -> dict: return {"job_id": job_id, "status": "queued"}
