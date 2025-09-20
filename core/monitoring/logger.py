# core/monitoring/logger.py
import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path
import os


class StructuredLogger:
    """Structured logger for consistent log formatting"""

    def __init__(self, name: str = "multi_modal_lab"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Setup log directory
        AI_CACHE_ROOT = os.getenv("AI_CACHE_ROOT", "/tmp/ai_cache")
        log_dir = Path(AI_CACHE_ROOT) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Console handler with JSON formatter
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(console_handler)

        # File handler for persistent logs
        file_handler = logging.FileHandler(log_dir / "app.log")
        file_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(file_handler)

        # Error file handler
        error_handler = logging.FileHandler(log_dir / "error.log")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(error_handler)

    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info level message"""
        self._log(logging.INFO, message, extra)

    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning level message"""
        self._log(logging.WARNING, message, extra)

    def error(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log error level message"""
        self._log(logging.ERROR, message, extra)

    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug level message"""
        self._log(logging.DEBUG, message, extra)

    def _log(self, level: int, message: str, extra: Optional[Dict[str, Any]] = None):
        """Internal logging method"""
        log_data = {
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "level": logging.getLevelName(level),
        }

        if extra:
            log_data.update(extra)

        self.logger.log(level, json.dumps(log_data))


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record):
        """Format log record as JSON"""
        try:
            # Try to parse message as JSON (for structured logs)
            log_data = json.loads(record.getMessage())
        except json.JSONDecodeError:
            # Fallback for plain text messages
            log_data = {
                "message": record.getMessage(),
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
            }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)


# Global logger instance
structured_logger = StructuredLogger()
