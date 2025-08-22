from workers.celery_app import celery_app
@celery_app.task
def gen_task(prompt: str) -> dict: return {"image_path": "/tmp/fake.png", "prompt": prompt}
