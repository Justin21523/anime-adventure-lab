# workers/celery_app.py
import os
from celery import Celery

# Redis broker (fallback to memory if Redis unavailable)
REDIS_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")

# Create Celery app
celery_app = Celery(
    "charaforge_workers",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["workers.tasks.batch", "workers.tasks.t2i", "workers.tasks.training"],
)

# Configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    result_expires=3600,  # 1 hour
    worker_prefetch_multiplier=1,  # Important for GPU tasks
    worker_max_tasks_per_child=10,  # Prevent memory leaks
)
