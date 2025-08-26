# core/performance/cache.py
"""Cache management utilities"""
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from ..shared_cache import get_shared_cache


class CacheManager:
    """Disk and memory cache management"""

    class Config:
        def __init__(self):
            self.max_disk_cache_mb = 5000
            self.max_memory_cache_mb = 1000
            self.ttl_seconds = 3600
            self.cleanup_interval = 300

    def __init__(self, config: Optional[Config] = None):
        self.config = config or self.Config()
        self.cache = get_shared_cache()
        self.cache_dir = Path(self.cache.cache_root) / "disk_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}
        self.last_cleanup = time.time()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            # Calculate disk cache size
            disk_size_bytes = sum(
                f.stat().st_size for f in self.cache_dir.rglob("*") if f.is_file()
            )
            disk_size_mb = disk_size_bytes / (1024 * 1024)

            # Count files
            file_count = len(list(self.cache_dir.rglob("*")))

            # Memory cache size (approximate)
            memory_size_mb = len(str(self.memory_cache)) / (
                1024 * 1024
            )  # Rough estimate

            # Check Redis availability (mock)
            redis_available = False  # Would check actual Redis connection

            return {
                "disk_cache_size_mb": round(disk_size_mb, 2),
                "disk_cache_files": file_count,
                "memory_cache_size_mb": round(memory_size_mb, 2),
                "memory_cache_items": len(self.memory_cache),
                "redis_available": redis_available,
                "max_disk_cache_mb": self.config.max_disk_cache_mb,
                "max_memory_cache_mb": self.config.max_memory_cache_mb,
            }

        except Exception as e:
            return {"error": str(e)}

    def cleanup_expired(self) -> None:
        """Clean up expired cache entries"""
        try:
            current_time = time.time()

            # Skip if cleaned up recently
            if current_time - self.last_cleanup < self.config.cleanup_interval:
                return

            # Clean memory cache
            expired_keys = []
            for key, data in self.memory_cache.items():
                if current_time - data.get("timestamp", 0) > self.config.ttl_seconds:
                    expired_keys.append(key)

            for key in expired_keys:
                del self.memory_cache[key]

            # Clean disk cache (simple age-based cleanup)
            for cache_file in self.cache_dir.rglob("*.json"):
                if current_time - cache_file.stat().st_mtime > self.config.ttl_seconds:
                    cache_file.unlink(missing_ok=True)

            self.last_cleanup = current_time

        except Exception as e:
            print(f"Cache cleanup error: {e}")
