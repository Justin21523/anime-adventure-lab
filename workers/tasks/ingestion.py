from workers.celery_app import celery_app
@celery_app.task
def ingest(world_id: str, path: str) -> dict: return {"world_id": world_id, "path": path}
