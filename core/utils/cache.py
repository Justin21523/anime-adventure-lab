# core/utils/cache.py
"""
Cache Management Utilities
File system caching, model caching, and data persistence
"""

import os
import json
import pickle
import hashlib
import logging
import shutil
from pathlib import Path
from typing import Any, Optional, Dict, Union, Callable
import time
from functools import wraps

from ..config import get_config
from ..exceptions import CacheError

logger = logging.getLogger(__name__)


class CacheManager:
    """Unified cache management system"""

    def __init__(self):
        self.config = get_config()
        self.cache_root = Path(os.getenv("AI_CACHE_ROOT", "/tmp/ai_cache"))
        self.cache_root.mkdir(parents=True, exist_ok=True)

        # Cache directories
        self.file_cache_dir = self.cache_root / "file_cache"
        self.model_cache_dir = self.cache_root / "model_cache"
        self.result_cache_dir = self.cache_root / "results"

        for cache_dir in [
            self.file_cache_dir,
            self.model_cache_dir,
            self.result_cache_dir,
        ]:
            cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache settings
        self.max_cache_size_gb = getattr(self.config, "max_cache_size_gb", 50)
        self.cache_ttl_hours = getattr(self.config, "cache_ttl_hours", 24)

    def _generate_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        cache_data = {"args": args, "kwargs": sorted(kwargs.items())}
        cache_string = json.dumps(cache_data, sort_keys=True, default=str)
        return hashlib.md5(cache_string.encode()).hexdigest()

    def _get_cache_path(self, key: str, cache_type: str = "result") -> Path:
        """Get cache file path"""
        if cache_type == "file":
            return self.file_cache_dir / f"{key}.cache"
        elif cache_type == "model":
            return self.model_cache_dir / f"{key}.cache"
        else:
            return self.result_cache_dir / f"{key}.cache"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file is valid and not expired"""
        if not cache_path.exists():
            return False

        # Check TTL
        file_age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        return file_age_hours < self.cache_ttl_hours

    def get(self, key: str, cache_type: str = "result") -> Optional[Any]:
        """Get item from cache"""
        try:
            cache_path = self._get_cache_path(key, cache_type)

            if not self._is_cache_valid(cache_path):
                return None

            with open(cache_path, "rb") as f:
                data = pickle.load(f)

            logger.debug(f"📋 Cache hit: {key}")
            return data

        except Exception as e:
            logger.debug(f"📋 Cache miss: {key} ({e})")
            return None

    def set(self, key: str, value: Any, cache_type: str = "result") -> bool:
        """Set item in cache"""
        try:
            cache_path = self._get_cache_path(key, cache_type)

            with open(cache_path, "wb") as f:
                pickle.dump(value, f)

            logger.debug(f"💾 Cached: {key}")
            return True

        except Exception as e:
            logger.warning(f"⚠️ Failed to cache {key}: {e}")
            return False

    def delete(self, key: str, cache_type: str = "result") -> bool:
        """Delete item from cache"""
        try:
            cache_path = self._get_cache_path(key, cache_type)
            if cache_path.exists():
                cache_path.unlink()
                logger.debug(f"🗑️ Deleted cache: {key}")
                return True
            return False

        except Exception as e:
            logger.warning(f"⚠️ Failed to delete cache {key}: {e}")
            return False

    def clear_expired(self) -> Dict[str, int]:
        """Clear expired cache entries"""
        cleared_counts = {"file": 0, "model": 0, "result": 0}

        for cache_type, cache_dir in [
            ("file", self.file_cache_dir),
            ("model", self.model_cache_dir),
            ("result", self.result_cache_dir),
        ]:
            for cache_file in cache_dir.glob("*.cache"):
                if not self._is_cache_valid(cache_file):
                    try:
                        cache_file.unlink()
                        cleared_counts[cache_type] += 1
                    except Exception as e:
                        logger.warning(
                            f"⚠️ Failed to delete expired cache {cache_file}: {e}"
                        )

        total_cleared = sum(cleared_counts.values())
        if total_cleared > 0:
            logger.info(f"🧹 Cleared {total_cleared} expired cache entries")

        return cleared_counts

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            "cache_root": str(self.cache_root),
            "total_size_mb": 0,
            "file_cache": {"count": 0, "size_mb": 0},
            "model_cache": {"count": 0, "size_mb": 0},
            "result_cache": {"count": 0, "size_mb": 0},
        }

        for cache_type, cache_dir in [
            ("file_cache", self.file_cache_dir),
            ("model_cache", self.model_cache_dir),
            ("result_cache", self.result_cache_dir),
        ]:
            cache_files = list(cache_dir.glob("*.cache"))
            total_size = sum(f.stat().st_size for f in cache_files if f.exists())

            stats[cache_type]["count"] = len(cache_files)
            stats[cache_type]["size_mb"] = total_size / 1024**2
            stats["total_size_mb"] += stats[cache_type]["size_mb"]

        return stats

    def cleanup_cache(self, target_size_gb: Optional[float] = None) -> Dict[str, Any]:
        """Cleanup cache to target size"""
        if target_size_gb is None:
            target_size_gb = self.max_cache_size_gb * 0.8  # Clean to 80% of max

        target_size_mb = target_size_gb * 1024
        stats = self.get_cache_stats()

        if stats["total_size_mb"] <= target_size_mb:
            return {"cleaned": False, "reason": "under_target_size", "stats": stats}

        # Clear expired entries first
        self.clear_expired()

        # Get updated stats
        stats = self.get_cache_stats()
        if stats["total_size_mb"] <= target_size_mb:
            return {"cleaned": True, "method": "expired_only", "stats": stats}

        # If still over limit, remove oldest files
        all_cache_files = []
        for cache_dir in [
            self.file_cache_dir,
            self.model_cache_dir,
            self.result_cache_dir,
        ]:
            for cache_file in cache_dir.glob("*.cache"):
                all_cache_files.append((cache_file, cache_file.stat().st_mtime))

        # Sort by modification time (oldest first)
        all_cache_files.sort(key=lambda x: x[1])

        removed_count = 0
        removed_size_mb = 0

        for cache_file, _ in all_cache_files:
            if stats["total_size_mb"] - removed_size_mb <= target_size_mb:
                break

            try:
                file_size_mb = cache_file.stat().st_size / 1024**2
                cache_file.unlink()
                removed_count += 1
                removed_size_mb += file_size_mb
            except Exception as e:
                logger.warning(f"⚠️ Failed to remove cache file {cache_file}: {e}")

        final_stats = self.get_cache_stats()
        logger.info(
            f"🧹 Cache cleanup: removed {removed_count} files ({removed_size_mb:.1f}MB)"
        )

        return {
            "cleaned": True,
            "method": "lru_cleanup",
            "removed_count": removed_count,
            "removed_size_mb": removed_size_mb,
            "stats": final_stats,
        }

    def cache_function(
        self, ttl_hours: Optional[float] = None, cache_type: str = "result"
    ):
        """Decorator for function result caching"""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_cache_key(func.__name__, *args, **kwargs)

                # Try to get from cache
                cached_result = self.get(cache_key, cache_type)
                if cached_result is not None:
                    return cached_result

                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, cache_type)

                return result

            return wrapper

        return decorator


# Global instances
_cache_manager = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager
