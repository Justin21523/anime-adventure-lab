# core/performance/cache_manager.py
import json
import hashlib
import pickle
from pathlib import Path
from typing import Any, Optional, Dict, List
from dataclasses import dataclass, asdict
import time
import redis
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    redis_url: str = "redis://localhost:6379/1"
    disk_cache_dir: str = "../ai_warehouse/cache/pipeline_cache"
    embedding_cache_ttl: int = 3600 * 24 * 7  # 1 week
    image_cache_ttl: int = 3600 * 24 * 3  # 3 days
    max_disk_cache_gb: float = 5.0
    enable_kv_cache: bool = True
    prefetch_queue_size: int = 3


class CacheManager:
    """Unified caching for embeddings, images, and KV states"""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.disk_cache_dir = Path(config.disk_cache_dir)
        self.disk_cache_dir.mkdir(parents=True, exist_ok=True)

        # Redis connection for fast access
        try:
            self.redis_client = redis.from_url(config.redis_url)
            self.redis_client.ping()
            self.redis_available = True
        except:
            logger.warning("Redis not available, using disk cache only")
            self.redis_available = False
            self.redis_client = None

    def _hash_key(self, data: Any) -> str:
        """Create hash key from data"""
        if isinstance(data, str):
            content = data.encode("utf-8")
        elif isinstance(data, dict):
            content = json.dumps(data, sort_keys=True).encode("utf-8")
        else:
            content = str(data).encode("utf-8")

        return hashlib.sha256(content).hexdigest()[:16]

    def get_embedding_cache(self, text: str, model_name: str) -> Optional[List[float]]:
        """Get cached embedding"""
        cache_key = f"emb:{model_name}:{self._hash_key(text)}"

        # Try Redis first
        if self.redis_available:
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")

        # Fallback to disk
        cache_file = self.disk_cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    if (
                        time.time() - data["timestamp"]
                        < self.config.embedding_cache_ttl
                    ):
                        return data["embedding"]
            except Exception as e:
                logger.warning(f"Disk cache read failed: {e}")

        return None

    def set_embedding_cache(self, text: str, model_name: str, embedding: List[float]):
        """Cache embedding"""
        cache_key = f"emb:{model_name}:{self._hash_key(text)}"

        # Store in Redis
        if self.redis_available:
            try:
                self.redis_client.setex(
                    cache_key, self.config.embedding_cache_ttl, json.dumps(embedding)
                )
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")

        # Store on disk
        cache_file = self.disk_cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(
                    {
                        "embedding": embedding,
                        "timestamp": time.time(),
                        "text_preview": text[:100],
                    },
                    f,
                )
        except Exception as e:
            logger.warning(f"Disk cache write failed: {e}")

    def get_image_cache(self, prompt_hash: str, model_config: Dict) -> Optional[str]:
        """Get cached image path"""
        config_hash = self._hash_key(model_config)
        cache_key = f"img:{prompt_hash}:{config_hash}"

        if self.redis_available:
            try:
                return self.redis_client.get(cache_key)
            except:
                pass

        return None

    def set_image_cache(self, prompt_hash: str, model_config: Dict, image_path: str):
        """Cache image generation result"""
        config_hash = self._hash_key(model_config)
        cache_key = f"img:{prompt_hash}:{config_hash}"

        if self.redis_available:
            try:
                self.redis_client.setex(
                    cache_key, self.config.image_cache_ttl, image_path
                )
            except Exception as e:
                logger.warning(f"Image cache set failed: {e}")

    def get_kv_cache(self, conversation_id: str, turn_id: int) -> Optional[Any]:
        """Get cached KV states for LLM"""
        if not self.config.enable_kv_cache:
            return None

        cache_key = f"kv:{conversation_id}:{turn_id}"
        cache_file = self.disk_cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"KV cache read failed: {e}")

        return None

    def set_kv_cache(self, conversation_id: str, turn_id: int, kv_states: Any):
        """Cache KV states"""
        if not self.config.enable_kv_cache:
            return

        cache_key = f"kv:{conversation_id}:{turn_id}"
        cache_file = self.disk_cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(kv_states, f)
        except Exception as e:
            logger.warning(f"KV cache write failed: {e}")

    def cleanup_expired(self):
        """Clean up expired cache files"""
        current_time = time.time()
        cleaned_count = 0

        for cache_file in self.disk_cache_dir.glob("*.json"):
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    if (
                        current_time - data.get("timestamp", 0)
                        > self.config.embedding_cache_ttl
                    ):
                        cache_file.unlink()
                        cleaned_count += 1
            except:
                # Remove corrupted files
                cache_file.unlink()
                cleaned_count += 1

        logger.info(f"Cleaned up {cleaned_count} expired cache files")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            "disk_cache_files": len(list(self.disk_cache_dir.glob("*"))),
            "disk_cache_size_mb": sum(
                f.stat().st_size for f in self.disk_cache_dir.glob("*")
            )
            / 1024**2,
            "redis_available": self.redis_available,
        }

        if self.redis_available:
            try:
                info = self.redis_client.info()
                stats["redis_memory_mb"] = info.get("used_memory", 0) / 1024**2
                stats["redis_keys"] = info.get("db1", {}).get("keys", 0)
            except:
                pass

        return stats
