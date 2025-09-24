# core/safety/rate_limiter.py
"""
Rate Limiting System
Implements request rate limiting, resource usage control and abuse prevention
"""

import time
import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import hashlib
import redis
import json

from ..config import get_config
from ..exceptions import RateLimitError

logger = logging.getLogger(__name__)


@dataclass
class RateLimit:
    """Rate limit configuration"""

    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_size: int = 10
    resource_quota: Dict[str, int] = field(
        default_factory=dict
    )  # e.g., {"gpu_seconds": 3600}


@dataclass
class RequestRecord:
    """Individual request record"""

    timestamp: float
    client_id: str
    endpoint: str
    resource_cost: Dict[str, float] = field(default_factory=dict)
    success: bool = True


class InMemoryRateLimiter:
    """In-memory rate limiter for single-instance deployments"""

    def __init__(self):
        self.config = get_config()
        self.redis = None  # Redis 連接 (optional)
        self.memory_store = {}  # 記憶體存儲作為備用方案

        self.request_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=10000)
        )
        self.resource_usage: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.blocked_clients: Dict[str, float] = {}  # client_id -> unblock_time

    def _cleanup_old_records(self, client_id: str, current_time: float):
        """Remove old records outside time windows"""
        history = self.request_history[client_id]
        # Keep only last 24 hours
        cutoff_time = current_time - 86400
        while history and history[0].timestamp < cutoff_time:
            history.popleft()

    def _get_request_counts(
        self, client_id: str, current_time: float
    ) -> Dict[str, int]:
        """Get request counts for different time windows"""
        history = self.request_history[client_id]

        counts = {"minute": 0, "hour": 0, "day": 0, "burst": 0}

        # Time thresholds
        minute_ago = current_time - 60
        hour_ago = current_time - 3600
        day_ago = current_time - 86400
        burst_window = current_time - 10  # 10 seconds for burst detection

        for record in history:
            if record.timestamp >= day_ago:
                counts["day"] += 1
            if record.timestamp >= hour_ago:
                counts["hour"] += 1
            if record.timestamp >= minute_ago:
                counts["minute"] += 1
            if record.timestamp >= burst_window:
                counts["burst"] += 1

        return counts

    def check_rate_limit(
        self, client_id: str, endpoint: str, rate_limit: RateLimit
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is within rate limits"""
        current_time = time.time()

        # Check if client is blocked
        if client_id in self.blocked_clients:
            if current_time < self.blocked_clients[client_id]:
                return False, {
                    "error": "Client temporarily blocked",
                    "unblock_time": self.blocked_clients[client_id],
                    "reason": "abuse_detected",
                }
            else:
                del self.blocked_clients[client_id]

        # Cleanup old records
        self._cleanup_old_records(client_id, current_time)

        # Get current counts
        counts = self._get_request_counts(client_id, current_time)

        # Check limits
        if counts["day"] >= rate_limit.requests_per_day:
            return False, {
                "error": "Daily rate limit exceeded",
                "limit": rate_limit.requests_per_day,
                "current": counts["day"],
                "reset_time": current_time + 86400 - (current_time % 86400),
            }

        if counts["hour"] >= rate_limit.requests_per_hour:
            return False, {
                "error": "Hourly rate limit exceeded",
                "limit": rate_limit.requests_per_hour,
                "current": counts["hour"],
                "reset_time": current_time + 3600 - (current_time % 3600),
            }

        if counts["minute"] >= rate_limit.requests_per_minute:
            return False, {
                "error": "Per-minute rate limit exceeded",
                "limit": rate_limit.requests_per_minute,
                "current": counts["minute"],
                "reset_time": current_time + 60 - (current_time % 60),
            }

        if counts["burst"] >= rate_limit.burst_size:
            return False, {
                "error": "Burst limit exceeded",
                "limit": rate_limit.burst_size,
                "current": counts["burst"],
                "reset_time": current_time + 10,
            }

        return True, {
            "allowed": True,
            "remaining": {
                "minute": rate_limit.requests_per_minute - counts["minute"],
                "hour": rate_limit.requests_per_hour - counts["hour"],
                "day": rate_limit.requests_per_day - counts["day"],
            },
        }

    def record_request(
        self,
        client_id: str,
        endpoint: str,
        resource_cost: Optional[Dict[str, float]] = None,
        success: bool = True,
    ):
        """Record a completed request"""
        current_time = time.time()

        record = RequestRecord(
            timestamp=current_time,
            client_id=client_id,
            endpoint=endpoint,
            resource_cost=resource_cost or {},
            success=success,
        )

        self.request_history[client_id].append(record)

        # Update resource usage
        if resource_cost:
            for resource, cost in resource_cost.items():
                self.resource_usage[client_id][resource] += cost

    def check_resource_quota(
        self, client_id: str, resource_cost: Dict[str, float], rate_limit: RateLimit
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if request would exceed resource quotas"""
        current_usage = self.resource_usage[client_id]

        for resource, cost in resource_cost.items():
            if resource in rate_limit.resource_quota:
                quota = rate_limit.resource_quota[resource]
                if current_usage[resource] + cost > quota:
                    return False, {
                        "error": f"Resource quota exceeded for {resource}",
                        "quota": quota,
                        "current_usage": current_usage[resource],
                        "requested": cost,
                    }

        return True, {"allowed": True}


class RedisRateLimiter:
    """Redis-based rate limiter for distributed deployments"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.config = get_config()
        self.redis = None  # Redis 連接 (optional)
        self.memory_store = {}  # 記憶體存儲作為備用方案

        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("✅ Connected to Redis for rate limiting")
        except Exception as e:
            logger.warning(
                f"⚠️ Failed to connect to Redis: {e}, falling back to in-memory limiter"
            )
            self.redis_client = None

    def _get_key(self, client_id: str, window: str) -> str:
        """Generate Redis key for rate limiting"""
        return f"ratelimit:{client_id}:{window}"

    def check_rate_limit(
        self, client_id: str, endpoint: str, rate_limit: RateLimit
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limits using Redis sliding window"""
        if not self.redis_client:
            # Fallback to in-memory limiter
            return InMemoryRateLimiter().check_rate_limit(
                client_id, endpoint, rate_limit
            )
        current_time = time.time()

        try:
            # Use Redis pipeline for atomic operations
            pipe = self.redis_client.pipeline()

            # Check different time windows
            windows = {
                "minute": (60, rate_limit.requests_per_minute),
                "hour": (3600, rate_limit.requests_per_hour),
                "day": (86400, rate_limit.requests_per_day),
                "burst": (10, rate_limit.burst_size),
            }

            for window, (duration, limit) in windows.items():
                key = self._get_key(client_id, window)

                # Remove old entries
                pipe.zremrangebyscore(key, 0, current_time - duration)

                # Count current entries
                pipe.zcard(key)

                # Set expiry
                pipe.expire(key, duration)

            results = pipe.execute()

            # Process results
            counts = {}
            for i, (window, (duration, limit)) in enumerate(windows.items()):
                count = results[i * 3 + 1]  # Get count result
                counts[window] = count

                if count >= limit:
                    return False, {
                        "error": f"{window.capitalize()} rate limit exceeded",
                        "limit": limit,
                        "current": count,
                        "reset_time": (
                            current_time + duration - (current_time % duration)
                            if window != "burst"
                            else current_time + 10
                        ),
                    }

            return True, {
                "allowed": True,
                "remaining": {
                    window: limit - counts[window]
                    for window, (_, limit) in windows.items()
                },
            }

        except Exception as e:
            logger.error(f"❌ Redis rate limit check failed: {e}")
            # Fallback to allowing request
            return True, {"allowed": True, "error": "Rate limiter unavailable"}

    def get_client_id(self, request) -> str:
        """Get client ID from request - 簡化版本"""
        try:
            # 從請求中提取客戶端 ID
            if hasattr(request, "client") and hasattr(request.client, "host"):
                return request.client.host
            elif hasattr(request, "headers"):
                return request.headers.get("x-forwarded-for", "unknown")
            else:
                return "test_client"
        except Exception:
            return "default_client"

    def get_rate_limit_for_endpoint(self, endpoint: str) -> Dict[str, int]:
        """Get rate limit configuration for specific endpoint"""
        # 返回預設限制
        return {
            "requests_per_minute": 60,
            "requests_per_hour": 1000,
            "requests_per_day": 10000,
        }

    def record_request(
        self,
        client_id: str,
        endpoint: str,
        resource_cost: Optional[Dict[str, float]] = None,
        success: bool = True,
    ):
        """Record request in Redis"""
        if not self.redis_client:
            return

        current_time = time.time()
        request_id = hashlib.md5(
            f"{client_id}:{endpoint}:{current_time}".encode()
        ).hexdigest()

        try:
            pipe = self.redis_client.pipeline()

            # Add to different time windows
            for window in ["minute", "hour", "day", "burst"]:
                key = self._get_key(client_id, window)
                pipe.zadd(key, {request_id: current_time})

            # Store request details
            details_key = f"request_details:{client_id}:{request_id}"
            request_data = {
                "endpoint": endpoint,
                "timestamp": current_time,
                "resource_cost": json.dumps(resource_cost or {}),
                "success": success,
            }
            pipe.hset(details_key, mapping=request_data)
            pipe.expire(details_key, 86400)  # Keep for 24 hours

            pipe.execute()

        except Exception as e:
            logger.error(f"❌ Failed to record request in Redis: {e}")


class RateLimiterManager:
    """Main rate limiter manager"""

    def __init__(self):
        self.config = get_config()

        # Initialize appropriate limiter
        if hasattr(self.config, "redis") and self.config.cache.redis_enable:
            self.limiter = RedisRateLimiter(self.config.cache.redis_url)
        else:
            self.limiter = InMemoryRateLimiter()

        # Define endpoint-specific rate limits
        self.endpoint_limits = self._load_endpoint_limits()

    def _load_endpoint_limits(self) -> Dict[str, RateLimit]:
        """Load rate limits for different endpoints"""
        return {
            # High-cost generation endpoints
            "/api/v1/txt2img": RateLimit(
                requests_per_minute=10,
                requests_per_hour=100,
                requests_per_day=500,
                burst_size=3,
                resource_quota={"gpu_seconds": 3600, "vram_mb": 8000},
            ),
            "/api/v1/img2img": RateLimit(
                requests_per_minute=8,
                requests_per_hour=80,
                requests_per_day=400,
                burst_size=2,
                resource_quota={"gpu_seconds": 3600, "vram_mb": 8000},
            ),
            "/api/v1/controlnet/*": RateLimit(
                requests_per_minute=5,
                requests_per_hour=50,
                requests_per_day=200,
                burst_size=2,
                resource_quota={"gpu_seconds": 7200, "vram_mb": 10000},
            ),
            # Medium-cost endpoints
            "/api/v1/caption": RateLimit(
                requests_per_minute=30,
                requests_per_hour=500,
                requests_per_day=2000,
                burst_size=10,
            ),
            "/api/v1/vqa": RateLimit(
                requests_per_minute=20,
                requests_per_hour=300,
                requests_per_day=1500,
                burst_size=5,
            ),
            # Low-cost endpoints
            "/api/v1/chat": RateLimit(
                requests_per_minute=60,
                requests_per_hour=1000,
                requests_per_day=5000,
                burst_size=20,
            ),
            "/api/v1/health": RateLimit(
                requests_per_minute=120,
                requests_per_hour=2000,
                requests_per_day=10000,
                burst_size=50,
            ),
            # Default for unlisted endpoints
            "default": RateLimit(
                requests_per_minute=30,
                requests_per_hour=300,
                requests_per_day=1000,
                burst_size=10,
            ),
        }

    def get_rate_limit_for_endpoint(self, endpoint: str) -> RateLimit:
        """Get rate limit configuration for specific endpoint"""
        # Try exact match first
        if endpoint in self.endpoint_limits:
            return self.endpoint_limits[endpoint]

        # Try pattern matching
        for pattern, limit in self.endpoint_limits.items():
            if pattern.endswith("*") and endpoint.startswith(pattern[:-1]):
                return limit

        # Return default
        return self.endpoint_limits["default"]

    def get_client_id(self, request) -> str:
        """Extract client ID from request"""
        # Try different methods to identify client

        # 1. API key (if using authentication)
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api_key:{hashlib.md5(api_key.encode()).hexdigest()[:16]}"

        # 2. User ID (if authenticated)
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return f"user:{user_id}"

        # 3. IP address (fallback)
        client_ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        if not client_ip:
            client_ip = request.headers.get("X-Real-IP", "")
        if not client_ip:
            client_ip = getattr(request.client, "host", "unknown")

        return f"ip:{client_ip}"

    async def check_rate_limit(
        self, request, endpoint: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is within rate limits"""
        client_id = self.get_client_id(request)
        rate_limit = self.get_rate_limit_for_endpoint(endpoint)

        return self.limiter.check_rate_limit(client_id, endpoint, rate_limit)

    def record_request(
        self,
        request,
        endpoint: str,
        resource_cost: Optional[Dict[str, float]] = None,
        success: bool = True,
    ):
        """Record completed request"""
        client_id = self.get_client_id(request)
        self.limiter.record_request(client_id, endpoint, resource_cost, success)

    def estimate_resource_cost(
        self, endpoint: str, params: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate resource cost for a request"""
        cost = {}

        if endpoint in ["/api/v1/txt2img", "/api/v1/img2img"]:
            # Estimate based on image resolution and steps
            width = params.get("width", 512)
            height = params.get("height", 512)
            steps = params.get("num_inference_steps", 20)

            # Base cost calculation
            pixel_count = width * height
            base_cost = pixel_count / (512 * 512)  # Normalize to 512x512
            step_multiplier = steps / 20  # Normalize to 20 steps

            cost["gpu_seconds"] = (
                base_cost * step_multiplier * 15
            )  # Estimated 15s for 512x512@20steps
            cost["vram_mb"] = base_cost * 4000  # Estimated 4GB for 512x512

        elif endpoint.startswith("/api/v1/controlnet"):
            # ControlNet adds overhead
            width = params.get("width", 512)
            height = params.get("height", 512)
            steps = params.get("num_inference_steps", 20)

            pixel_count = width * height
            base_cost = pixel_count / (512 * 512)
            step_multiplier = steps / 20

            cost["gpu_seconds"] = (
                base_cost * step_multiplier * 25
            )  # Higher cost for ControlNet
            cost["vram_mb"] = base_cost * 6000  # Higher VRAM usage

        elif endpoint in ["/api/v1/caption", "/api/v1/vqa"]:
            # Vision-language models
            cost["gpu_seconds"] = 2.0
            cost["vram_mb"] = 1000

        elif endpoint == "/api/v1/chat":
            # Text generation
            max_tokens = params.get("max_tokens", 150)
            cost["gpu_seconds"] = max_tokens / 100  # Rough estimate
            cost["vram_mb"] = 500

        return cost


# Global instance
_rate_limiter = None


def get_rate_limiter() -> RateLimiterManager:
    """Get global rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiterManager()
    return _rate_limiter
