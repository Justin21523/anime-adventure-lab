# core/utils/logging.py
"""
Structured Logging Utilities
JSON logging, performance tracking, and log management
"""

import logging
import logging.handlers
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path
import traceback
from datetime import datetime

from ..config import get_config


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add extra fields
        if hasattr(record, "extra_data"):
            log_entry.update(record.extra_data)

        return json.dumps(log_entry, ensure_ascii=False)


class PerformanceLogger:
    """Performance tracking logger"""

    def __init__(self, logger_name: str = "performance"):
        self.logger = logging.getLogger(logger_name)
        self.start_times: Dict[str, float] = {}

    def start_operation(self, operation_id: str, operation_type: str, **metadata):
        """Start tracking an operation"""
        self.start_times[operation_id] = time.time()

        self.logger.info(
            f"Operation started: {operation_type}",
            extra={
                "extra_data": {
                    "operation_id": operation_id,
                    "operation_type": operation_type,
                    "event": "operation_start",
                    **metadata,
                }
            },
        )

    def end_operation(self, operation_id: str, success: bool = True, **metadata):
        """End tracking an operation"""
        if operation_id not in self.start_times:
            self.logger.warning(f"Operation {operation_id} not found in start times")
            return

        duration = time.time() - self.start_times.pop(operation_id)

        self.logger.info(
            f"Operation completed: {operation_id}",
            extra={
                "extra_data": {
                    "operation_id": operation_id,
                    "event": "operation_end",
                    "duration_seconds": duration,
                    "success": success,
                    **metadata,
                }
            },
        )

        return duration


def setup_structured_logging(config=None):
    """Setup structured logging configuration"""
    if config is None:
        config = get_config()

    # Create logs directory
    log_dir = Path(getattr(config, "log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, getattr(config, "log_level", "INFO").upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler with structured format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    use_structured = getattr(config, "structured_logging", False)
    if use_structured:
        console_formatter = StructuredFormatter()
    else:
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "app.log", maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
    )
    file_handler.setLevel(
        getattr(logging, getattr(config, "file_log_level", "DEBUG").upper())
    )
    file_handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(file_handler)

    # Error file handler
    error_handler = logging.handlers.RotatingFileHandler(
        log_dir / "error.log", maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(error_handler)


def get_logger(
    name: str, extra_data: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """Get logger with optional extra data"""
    logger = logging.getLogger(name)

    if extra_data:
        # Create adapter to inject extra data
        class LoggerAdapter(logging.LoggerAdapter):
            def process(self, msg, kwargs):
                if "extra" not in kwargs:
                    kwargs["extra"] = {}
                if "extra_data" not in kwargs["extra"]:
                    kwargs["extra"]["extra_data"] = {}
                kwargs["extra"]["extra_data"].update(self.extra)
                return msg, kwargs

        return LoggerAdapter(logger, extra_data)

    return logger


# Global instances
_model_manager = None
_cache_manager = None


def get_model_manager() -> ModelManager:
    """Get global model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager
