# backend/schemas/__init__.py
from .base import *
from .caption import *
from .vqa import *
from .chat import *
from .batch import *  # NEW
from .monitoring import *  # NEW

__all__ = [
    # Base schemas
    "BaseResponse",
    "ErrorResponse",
    "HealthResponse",
    # Feature schemas
    "CaptionRequest",
    "CaptionResponse",
    "VQARequest",
    "VQAResponse",
    "ChatRequest",
    "ChatResponse",
    "ChatMessage",
    # Batch schemas (NEW)
    "BatchJobRequest",
    "BatchJobResponse",
    "BatchJobList",
    "BatchStatus",
    "TaskProgress",
    # Monitoring schemas (NEW)
    "SystemMetrics",
    "TaskMetrics",
    "WorkerStatus",
    "AlertStatus",
    "AlertLevel",
    "PerformanceReport",
    "QueueStats",
    "MonitoringDashboardData",
]
