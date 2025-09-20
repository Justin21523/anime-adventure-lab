# ===================================
# schemas/monitoring.py
"""
System Monitoring API Schemas
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
from .batch import BatchJobResponse
from enum import Enum


class SystemMetrics(BaseModel):
    """System resource metrics"""

    timestamp: datetime = Field(..., description="Metrics timestamp")
    cpu_percent: float = Field(..., description="CPU usage percentage")
    memory_used_gb: float = Field(..., description="Memory used in GB")
    memory_total_gb: float = Field(..., description="Total memory in GB")
    memory_percent: float = Field(..., description="Memory usage percentage")
    disk_used_gb: float = Field(..., description="Disk used in GB")
    disk_total_gb: float = Field(..., description="Total disk in GB")
    disk_percent: float = Field(..., description="Disk usage percentage")
    gpu_metrics: Dict[str, Any] = Field(
        default_factory=dict, description="GPU metrics by device"
    )


class TaskMetrics(BaseModel):
    """Task execution metrics"""

    timestamp: datetime = Field(..., description="Metrics timestamp")
    total_tasks: int = Field(..., description="Total tasks executed")
    completed_tasks: int = Field(..., description="Successfully completed tasks")
    failed_tasks: int = Field(..., description="Failed tasks")
    pending_tasks: int = Field(..., description="Pending tasks")
    avg_execution_time: float = Field(..., description="Average execution time in ms")
    tasks_per_minute: float = Field(..., description="Tasks processed per minute")
    error_rate: float = Field(..., description="Task error rate (0.0-1.0)")


class WorkerStatus(BaseModel):
    """Celery worker status"""

    name: str = Field(..., description="Worker name")
    status: str = Field(..., description="Worker status (online/offline)")
    active_tasks: int = Field(..., description="Number of active tasks")
    processed_tasks: int = Field(..., description="Total processed tasks")
    last_heartbeat: datetime = Field(..., description="Last heartbeat timestamp")
    queue_size: int = Field(..., description="Current queue size")


class AlertLevel(str, Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertStatus(BaseModel):
    """System alert information"""

    id: str = Field(..., description="Alert identifier")
    level: AlertLevel = Field(..., description="Alert severity level")
    message: str = Field(..., description="Alert message")
    timestamp: datetime = Field(..., description="Alert timestamp")
    resolved: bool = Field(default=False, description="Whether alert is resolved")


class PerformanceReport(BaseModel):
    """Performance analysis report"""

    time_range: str = Field(..., description="Report time range")
    total_requests: int = Field(..., description="Total API requests")
    avg_response_time: float = Field(..., description="Average response time in ms")
    error_rate: float = Field(..., description="API error rate")
    peak_cpu_usage: float = Field(..., description="Peak CPU usage")
    peak_memory_usage: float = Field(..., description="Peak memory usage")
    top_endpoints: List[Dict[str, Any]] = Field(..., description="Most used endpoints")
    slowest_endpoints: List[Dict[str, Any]] = Field(
        ..., description="Slowest endpoints"
    )


class QueueStats(BaseModel):
    """Queue statistics"""

    queue_name: str = Field(..., description="Queue name")
    active_tasks: int = Field(..., description="Active tasks in queue")
    scheduled_tasks: int = Field(..., description="Scheduled tasks in queue")
    priority: int = Field(..., description="Queue priority")
    estimated_wait_time: Optional[int] = Field(
        None, description="Estimated wait time in seconds"
    )


class MonitoringDashboardData(BaseModel):
    """Complete dashboard data"""

    system_metrics: SystemMetrics = Field(..., description="Current system metrics")
    task_metrics: TaskMetrics = Field(..., description="Task execution metrics")
    worker_status: List[WorkerStatus] = Field(..., description="Worker status list")
    active_alerts: List[AlertStatus] = Field(..., description="Active system alerts")
    queue_stats: List[QueueStats] = Field(..., description="Queue statistics")
    recent_jobs: List[BatchJobResponse] = Field(..., description="Recent batch jobs")
