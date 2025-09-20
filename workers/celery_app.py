# workers/celery_app.py
import os
import os
import redis
from kombu import Queue
from core.shared_cache import get_shared_cache
from celery import Celery

# Shared cache setup
get_shared_cache()

# Redis connection
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.from_url(REDIS_URL)

# Celery app configuration
celery_app = Celery(
    "multi_modal_lab",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["backend.jobs.tasks"],
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,
    # Custom queues for different task types
    task_routes={
        "backend.jobs.tasks.batch_caption_task": {"queue": "vision"},
        "backend.jobs.tasks.batch_vqa_task": {"queue": "vision"},
        "backend.jobs.tasks.batch_chat_task": {"queue": "text"},
        "backend.jobs.tasks.train_lora_task": {"queue": "training"},
    },
    task_default_queue="default",
    task_queues=[
        Queue("default"),
        Queue("vision"),
        Queue("text"),
        Queue("training"),
    ],
)
