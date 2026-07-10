# workers/celery_app.py
import os
import json as _json
from datetime import datetime as _dt
import redis
from kombu import Queue
from core.shared_cache import get_shared_cache
from celery import Celery

# Shared cache setup
get_shared_cache()

# Redis connection
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.from_url(REDIS_URL)

# The default worker is intentionally CPU-safe and only registers the durable
# v2 tasks.  The legacy task set imports PyTorch and model-specific adapters;
# operators must opt in when running a separately provisioned AI worker.
WORKER_PROFILE = os.getenv("WORKER_PROFILE", "core").strip().lower()
CORE_TASK_MODULES = [
    "workers.tasks.story_v2",
    "workers.tasks.rag_v2",
    "workers.tasks.maintenance_v2",
]
EXPERIMENTAL_TASK_MODULES = [
    "workers.tasks",
    "workers.tasks.batch",
    "workers.tasks.t2i",
    "workers.tasks.story",
    "workers.tasks.datasets",
    "workers.tasks.rag",
    "workers.tasks.training",
]
task_modules = CORE_TASK_MODULES.copy()
if WORKER_PROFILE in {"experimental", "ai", "full"}:
    os.environ.setdefault("ENABLE_EXPERIMENTAL_WORKER_TASKS", "1")
    task_modules.extend(EXPERIMENTAL_TASK_MODULES)

# Celery app configuration
celery_app = Celery(
    "multi_modal_lab",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=task_modules,
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
        # NOTE: task names use explicit `name=` in decorators; route by that.
        "batch_caption_task": {"queue": "vision"},
        "batch_vqa_task": {"queue": "vision"},
        "batch_chat_task": {"queue": "text"},
        "story_turn_task": {"queue": "text"},
        "story_turn_v2_task": {"queue": "text"},
        "train_lora_task": {"queue": "training"},
        "generate_image_async": {"queue": "vision"},
        "story_scene_image_task": {"queue": "vision"},
        "dataset_caption_task": {"queue": "vision"},
        "rag_rebuild_task": {"queue": "training"},
        "rag_upload_task": {"queue": "training"},
        "document_index_v2_task": {"queue": "training"},
    },
    task_default_queue="default",
    task_queues=[
        Queue("default"),
        Queue("vision"),
        Queue("text"),
        Queue("training"),
        Queue("postprocess"),
    ],
    worker_send_task_events=True,
    task_send_sent_event=True,
    task_acks_late=True,
    beat_schedule={
        "reconcile-v2-jobs": {
            "task": "reconcile_v2_jobs_task",
            "schedule": 30.0,
        }
    },
)


# ---- TaskProgress (absorbed from charaforge-T2I-Lab) ----
#
# Publishes progress events to Redis PubSub so the API WebSocket endpoint
# can relay them to the client in real time.


class TaskProgress:
    """Task progress tracker with Redis PubSub publishing."""

    def __init__(self, task, total_steps: int = 100):
        self.task = task
        self.total_steps = total_steps
        self.current_step = 0

    def _publish_ws(self, state: str, payload: dict) -> None:
        """Publish progress events to Redis PubSub for WebSocket consumers."""
        try:
            import redis as _redis
        except Exception:
            return

        job_id = getattr(getattr(self.task, "request", None), "id", None)
        if not job_id:
            return

        url = os.getenv("REDIS_URL") or os.getenv("CELERY_BROKER_URL") or REDIS_URL
        if not url or not str(url).startswith("redis"):
            return

        channel = f"anime_adventure:train:{job_id}"
        topic = (
            "training.progress"
            if state == "PROGRESS"
            else "training.complete" if state == "SUCCESS" else "training.failure"
        )

        try:
            client = _redis.Redis.from_url(
                url,
                socket_timeout=1,
                socket_connect_timeout=1,
                retry_on_timeout=False,
                decode_responses=True,
            )
            client.publish(
                channel,
                _json.dumps(
                    {
                        "topic": topic,
                        "job_id": job_id,
                        "state": state,
                        "progress": payload,
                        "timestamp": _dt.now().isoformat(),
                    },
                    ensure_ascii=False,
                ),
            )
        except Exception:
            return

    def update(self, step: int, message: str = "", **extra_data):
        self.current_step = step
        progress_percent = (step / self.total_steps) * 100
        state_data = {
            "current": step,
            "total": self.total_steps,
            "percent": round(progress_percent, 1),
            "message": message,
            **extra_data,
        }
        self.task.update_state(state="PROGRESS", meta=state_data)
        self._publish_ws("PROGRESS", state_data)
        return state_data

    def complete(self, result: dict):
        final_data = {
            "current": self.total_steps,
            "total": self.total_steps,
            "percent": 100.0,
            "message": "Complete",
            **result,
        }
        self.task.update_state(state="SUCCESS", meta=final_data)
        self._publish_ws("SUCCESS", final_data)
        return final_data

    def fail(self, error_message: str, **extra_data):
        error_data = {
            "current": self.current_step,
            "total": self.total_steps,
            "percent": (self.current_step / self.total_steps) * 100,
            "message": f"Failed: {error_message}",
            "error": error_message,
            **extra_data,
        }
        self.task.update_state(state="FAILURE", meta=error_data)
        self._publish_ws("FAILURE", error_data)
        return error_data
