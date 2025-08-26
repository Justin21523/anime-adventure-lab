# workers/utils.py
import os
import redis
import json
from typing import Optional, Dict, Any


def get_redis_client():
    """Get Redis client (with fallback)"""
    try:
        client = redis.Redis.from_url(
            os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
            decode_responses=True,
        )
        client.ping()  # Test connection
        return client
    except:
        return None


def store_job_result(job_id: str, result: Dict[str, Any]):
    """Store job result (Redis or file fallback)"""
    client = get_redis_client()

    if client:
        client.setex(f"job_result:{job_id}", 3600, json.dumps(result))
    else:
        # File fallback
        result_file = Path(BATCH_DIR) / f"{job_id}_result.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)


def get_job_result(job_id: str) -> Optional[Dict[str, Any]]:
    """Get job result"""
    client = get_redis_client()

    if client:
        result_data = client.get(f"job_result:{job_id}")
        if result_data:
            return json.loads(result_data)
    else:
        # File fallback
        result_file = Path(BATCH_DIR) / f"{job_id}_result.json"
        if result_file.exists():
            with open(result_file, "r") as f:
                return json.load(f)

    return None
