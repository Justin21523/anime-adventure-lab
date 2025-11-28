"""
Performance Router
Provides system/request profiling, recent latency summaries, and thresholds.
"""

import logging
from typing import Optional, Any, Dict
from datetime import datetime

from fastapi import APIRouter, HTTPException

from core.performance.monitor import get_performance_monitor
from schemas.monitoring import PerformanceReport

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/performance", tags=["performance"])
perf_monitor = get_performance_monitor()


@router.get("/stats")
async def get_performance_stats(endpoint: Optional[str] = None, window_seconds: int = 3600):
    """Return aggregated performance stats for recent completed requests."""
    try:
        stats = perf_monitor.profiler.get_performance_stats(
            endpoint=endpoint, time_window=window_seconds
        )
        return {"timestamp": datetime.utcnow().isoformat(), "stats": stats}
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to get performance stats: %s", e)
        raise HTTPException(500, f"Performance stats failed: {str(e)}") from e


@router.get("/active")
async def list_active_requests():
    """List currently active profiled requests."""
    try:
        active = perf_monitor.profiler.get_active_requests()
        return {"active_count": len(active), "active": [r.__dict__ for r in active]}
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to list active requests: %s", e)
        raise HTTPException(500, f"Active request list failed: {str(e)}") from e


@router.get("/history")
async def recent_history(limit: int = 100):
    """Return recent completed requests with durations and memory peaks."""
    try:
        completed = perf_monitor.profiler.get_completed_requests(limit=limit)
        data = [
            {
                "request_id": r.request_id,
                "endpoint": r.endpoint,
                "duration": r.duration,
                "status": r.status,
                "error": r.error,
                "peak_memory_mb": r.peak_memory_mb,
                "peak_gpu_memory_mb": r.peak_gpu_memory_mb,
                "start_time": r.start_time,
                "end_time": r.end_time,
            }
            for r in completed
        ]
        return {"count": len(data), "requests": data}
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to read performance history: %s", e)
        raise HTTPException(500, f"Performance history failed: {str(e)}") from e


@router.post("/thresholds")
async def update_thresholds(payload: Dict[str, Any]):
    """Update performance alert thresholds (cpu_percent, memory_percent, etc.)."""
    try:
        perf_monitor.thresholds.update(payload)
        return {"success": True, "thresholds": perf_monitor.thresholds}
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to update thresholds: %s", e)
        raise HTTPException(500, f"Update thresholds failed: {str(e)}") from e


@router.get("/report", response_model=PerformanceReport)
async def get_performance_report(hours: int = 24):
    """Return the same performance report as monitoring endpoint for convenience."""
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        report = await metrics_collector.generate_performance_report(  # type: ignore
            start_time=start_time, end_time=end_time
        )
        return PerformanceReport(
            time_range=f"{start_time.isoformat()} to {end_time.isoformat()}",
            total_requests=report["total_requests"],
            avg_response_time=report["avg_response_time"],
            error_rate=report["error_rate"],
            peak_cpu_usage=report["peak_cpu_usage"],
            peak_memory_usage=report["peak_memory_usage"],
            top_endpoints=report["top_endpoints"],
            slowest_endpoints=report["slowest_endpoints"],
        )
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to generate performance report: %s", e)
        raise HTTPException(500, f"Performance report failed: {str(e)}") from e

