from workers.celery_app import celery_app
@celery_app.task
def lora_train(run_id: str, config_path: str) -> dict: return {"run_id": run_id, "config": config_path, "ok": True}
