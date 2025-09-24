# core/exceptions.py
"""
Unified Exception Classes
Standardized error handling across all modules
"""
import logging
import asyncio
import hashlib
import json
import time
from typing import Any, Optional, Dict, List
from functools import wraps, lru_cache
from dataclasses import dataclass
import torch
import redis
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
from core.config import get_config
from core.shared_cache import get_shared_cache

logger = logging.getLogger(__name__)


class MultiModalLabError(Exception):
    """Base exception for Multi-Modal Lab"""

    def __init__(self, message: str, error_code: str = "UNKNOWN", details: dict = None):  # type: ignore
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
        }


class ModelError(MultiModalLabError):
    """Model-related errors"""

    def __init__(
        self, message: str, model_name: str = "", error_code: str = "MODEL_ERROR"
    ):
        super().__init__(message, error_code, {"model_name": model_name})


class ModelNotFoundError(ModelError):
    """Model not found or not loaded"""

    def __init__(self, model_name: str):
        super().__init__(
            f"Model not found: {model_name}", model_name, "MODEL_NOT_FOUND"
        )


class ModelLoadError(ModelError):
    """Model loading failed"""

    def __init__(self, model_name: str, reason: str = ""):
        message = f"Failed to load model: {model_name}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, model_name, "MODEL_LOAD_ERROR")


class CUDAOutOfMemoryError(ModelError):
    """CUDA out of memory error"""

    def __init__(self, model_name: str, allocated_gb: float = 0):
        message = f"CUDA OOM when loading {model_name}"
        if allocated_gb > 0:
            message += f" (allocated: {allocated_gb:.1f}GB)"
        super().__init__(message, model_name, "CUDA_OOM")


# === Performance Errors ===
class PerformanceError(MultiModalLabError):
    """Performance monitoring and system metric errors"""

    def __init__(
        self, message: str, metric_type: str = "", error_code: str = "PERFORMANCE_ERROR"
    ):
        super().__init__(message, error_code, {"metric_type": metric_type})


class SystemMetricsError(PerformanceError):
    """System metrics collection failed"""

    def __init__(self, message: str):
        super().__init__(message, "system_metrics", "SYSTEM_METRICS_ERROR")


class ProfilingError(PerformanceError):
    """Request profiling failed"""

    def __init__(self, message: str):
        super().__init__(message, "profiling", "PROFILING_ERROR")


class ResourceLimitExceeded(PerformanceError):
    """Resource limits exceeded"""

    def __init__(self, resource_type: str, limit: str, current: str):
        message = f"{resource_type} limit exceeded: {current} > {limit}"
        super().__init__(message, resource_type, "RESOURCE_LIMIT_EXCEEDED")


# API
class APIError(MultiModalLabError):
    """API 相關錯誤"""

    def __init__(
        self, message: str, status_code: int = 500, error_code: str = "API_ERROR"
    ):
        super().__init__(message, error_code, {"status_code": status_code})
        self.status_code = status_code


# Validation Errors
class ValidationError(MultiModalLabError):
    """驗證錯誤"""

    def __init__(self, field: str, message: str, error_type: str = "validation_error"):
        super().__init__(
            f"Validation error in field '{field}': {message}",
            "VALIDATION_ERROR",
            {"field": field, "error_type": error_type},
        )


# =============================================================================
# Processing Related Exceptions
# =============================================================================


class ProcessingError(MultiModalLabError):
    """處理過程錯誤"""

    def __init__(self, message: str, process_type: str = "unknown"):
        super().__init__(message, "PROCESSING_ERROR", {"process_type": process_type})


class ConversionError(ProcessingError):
    """格式轉換錯誤"""

    def __init__(self, message: str, source_format: str = "", target_format: str = ""):
        super().__init__(f"Conversion error: {message}", "conversion")
        self.details.update(
            {"source_format": source_format, "target_format": target_format}
        )


class AuthenticationError(APIError):
    """認證錯誤"""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, 401, "AUTHENTICATION_ERROR")


class AuthorizationError(APIError):
    """授權錯誤"""

    def __init__(self, message: str = "Access denied"):
        super().__init__(message, 403, "AUTHORIZATION_ERROR")


class ContextLengthExceededError(MultiModalLabError):
    """Context length exceeded error"""

    def __init__(self, current_length: int, max_length: int, model_name: str = ""):
        message = f"Context length exceeded: {current_length} > {max_length}"
        if model_name:
            message += f" for model {model_name}"
        super().__init__(
            message,
            "CONTEXT_LENGTH_EXCEEDED",
            {
                "current_length": current_length,
                "max_length": max_length,
                "model_name": model_name,
            },
        )


class VLMError(MultiModalLabError):
    """Vision-Language Model errors"""

    def __init__(self, message: str, error_code: str = "VLM_ERROR"):
        super().__init__(message, error_code)


class ImageProcessingError(VLMError):
    """Image processing failed"""

    def __init__(self, reason: str):
        super().__init__(f"Image processing failed: {reason}", "IMAGE_PROCESSING_ERROR")


class RAGError(MultiModalLabError):
    """RAG system errors"""

    def __init__(self, message: str, error_code: str = "RAG_ERROR"):
        super().__init__(message, error_code)


class EmbeddingError(RAGError):
    """Embedding generation failed"""

    def __init__(self, text: str, reason: str = ""):
        message = f"Failed to generate embedding for text: {text[:50]}..."
        if reason:
            message += f" - {reason}"
        super().__init__(message, "EMBEDDING_ERROR")


class DocumentIndexError(RAGError):
    """Document indexing failed"""

    def __init__(self, doc_id: str, reason: str = ""):
        message = f"Failed to index document: {doc_id}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, "DOC_INDEX_ERROR")


# Text Processing Errors
class TextProcessingError(MultiModalLabError):
    """Text processing failed"""

    def __init__(self, operation: str, reason: str = ""):
        message = f"Text processing failed: {operation}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, "TEXT_PROCESSING_ERROR")


class ChineseProcessingError(TextProcessingError):
    """Chinese text processing failed"""

    def __init__(self, text: str, reason: str = ""):
        super().__init__(f"Chinese processing for text: {text[:50]}...", reason)


# Session Management Errors
class SessionNotFoundError(MultiModalLabError):
    """Session not found in chat manager"""

    def __init__(self, session_id: str):
        super().__init__(
            f"Session not found: {session_id}",
            "SESSION_NOT_FOUND",
            {"session_id": session_id},
        )


class SessionExpiredError(MultiModalLabError):
    """Session has expired"""

    def __init__(self, session_id: str, expired_at: str = ""):
        message = f"Session expired: {session_id}"
        if expired_at:
            message += f" (expired at: {expired_at})"
        super().__init__(message, "SESSION_EXPIRED", {"session_id": session_id})


class ConfigurationError(MultiModalLabError):
    """Configuration error"""

    def __init__(self, config_key: str, reason: str = ""):
        message = f"Configuration error: {config_key}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, "CONFIG_ERROR", {"config_key": config_key})


# Agent and Tool Errors
class AgentError(MultiModalLabError):
    """Agent execution failed"""

    def __init__(self, agent_name: str, reason: str = ""):
        message = f"Agent error: {agent_name}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, "AGENT_ERROR", {"agent_name": agent_name})


class ToolExecutionError(AgentError):
    """Tool execution failed"""

    def __init__(self, tool_name: str, reason: str = ""):
        super().__init__(f"Tool execution failed: {tool_name}", reason)


class ToolNotFoundError(AgentError):
    """Tool not found in registry"""

    def __init__(self, tool_name: str):
        super().__init__(f"Tool not found: {tool_name}")


# Batch Processing Errors
class BatchProcessingError(MultiModalLabError):
    """Batch processing failed"""

    def __init__(self, batch_id: str, reason: str = ""):
        message = f"Batch processing failed: {batch_id}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, "BATCH_ERROR", {"batch_id": batch_id})


class QueueFullError(BatchProcessingError):
    """Task queue is full"""

    def __init__(self, queue_name: str, max_size: int):
        super().__init__(
            f"Queue full: {queue_name} (max: {max_size})",
            f"Queue {queue_name} has reached maximum capacity",
        )


# Story Engine Errors
class StoryEngineError(MultiModalLabError):
    """Story engine errors"""

    def __init__(self, message: str, error_code: str = "STORY_ERROR"):
        super().__init__(message, error_code)


class CharacterError(StoryEngineError):
    """Character-related errors"""

    def __init__(self, character_name: str, reason: str = ""):
        message = f"Character error: {character_name}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, "CHARACTER_ERROR")


class NarrativeError(StoryEngineError):
    """Narrative generation errors"""

    def __init__(self, reason: str):
        super().__init__(f"Narrative generation failed: {reason}", "NARRATIVE_ERROR")


# Export and Format Errors
class ExportError(MultiModalLabError):
    """Export operation failed"""

    def __init__(self, format_type: str, reason: str = ""):
        message = f"Export failed: {format_type}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, "EXPORT_ERROR", {"format": format_type})


class FormatNotSupportedError(ExportError):
    """Unsupported format"""

    def __init__(self, format_type: str, supported_formats: List[str] = None):  # type: ignore
        message = f"Format not supported: {format_type}"
        if supported_formats:
            message += f" (supported: {', '.join(supported_formats)})"
        super().__init__(format_type, message)


# Monitoring and Metrics Errors
class MonitoringError(MultiModalLabError):
    """Monitoring system errors"""

    def __init__(self, component: str, reason: str = ""):
        message = f"Monitoring error: {component}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, "MONITORING_ERROR", {"component": component})


class MetricsCollectionError(MonitoringError):
    """Metrics collection failed"""

    def __init__(self, metric_name: str, reason: str = ""):
        super().__init__(f"Metrics collection failed: {metric_name}", reason)


# Cache and Storage Errors


class StorageError(MultiModalLabError):
    """Storage operation failed"""

    def __init__(self, path: str, operation: str, reason: str = ""):
        message = f"Storage error: {operation} on {path}"
        if reason:
            message += f" - {reason}"
        super().__init__(
            message, "STORAGE_ERROR", {"path": path, "operation": operation}
        )


class CacheError(MultiModalLabError):
    """Cache operation failed"""

    def __init__(self, operation: str, reason: str = ""):
        message = f"Cache error: {operation}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, "CACHE_ERROR", {"operation": operation})


class DatabaseError(StorageError):
    """資料庫錯誤"""

    def __init__(self, message: str, operation: str = "unknown"):
        super().__init__(f"Database error: {message}", "database")
        self.details["operation"] = operation


# =============================================================================
# Configuration Related Exceptions
# =============================================================================


class ConfigError(MultiModalLabError):
    """配置錯誤"""

    def __init__(self, message: str, config_key: str = ""):
        super().__init__(message, "CONFIG_ERROR", {"config_key": config_key})


class EnvironmentError(ConfigError):
    """環境配置錯誤"""

    def __init__(self, message: str, env_var: str = ""):
        super().__init__(f"Environment error: {message}", "environment")
        self.details["env_var"] = env_var


# Security and Safety Errors
class SecurityError(MultiModalLabError):
    """Security violation"""

    def __init__(self, violation_type: str, details: str = ""):
        message = f"Security violation: {violation_type}"
        if details:
            message += f" - {details}"
        super().__init__(message, "SECURITY_ERROR", {"violation": violation_type})


class SafetyError(MultiModalLabError):
    """Safety filter errors"""

    def __init__(
        self, message: str, content_type: str = "", error_code: str = "SAFETY_ERROR"
    ):
        super().__init__(message, error_code, {"content_type": content_type})


class ContentFilterError(SafetyError):
    """Content filtered by safety system"""

    def __init__(self, message: str, filter_type: str = "nsfw"):
        super().__init__(f"Content filtered: {message}", "content_filter")
        self.details["filter_type"] = filter_type


class RateLimitError(SecurityError):
    """Rate limit exceeded"""

    def __init__(self, resource: str, limit: int):
        super().__init__(
            f"Rate limit exceeded: {resource}", f"Exceeded limit of {limit} requests"
        )


class GameEngineError(StoryEngineError):
    """Game engine specific errors"""

    pass


class GameError(MultiModalLabError):
    """Text adventure game errors"""

    def __init__(
        self, message: str, session_id: str = "", error_code: str = "GAME_ERROR"
    ):
        super().__init__(message, error_code, {"session_id": session_id})


class InvalidChoiceError(GameError):
    """Invalid player choice"""

    def __init__(self, choice_id: str, session_id: str = ""):
        super().__init__(f"Invalid choice: {choice_id}", session_id, "INVALID_CHOICE")


class T2IError(MultiModalLabError):
    """Text-to-Image errors"""

    def __init__(self, message: str, error_code: str = "T2I_ERROR"):
        super().__init__(message, error_code)


class LoRAError(T2IError):
    """LoRA-related errors"""

    def __init__(self, message: str, lora_id: str = "", error_code: str = "LORA_ERROR"):
        super().__init__(message, error_code)
        self.lora_id = lora_id


class LoRANotFoundError(LoRAError):
    """LoRA model not found"""

    def __init__(self, lora_id: str):
        super().__init__(f"LoRA not found: {lora_id}", lora_id, "LORA_NOT_FOUND")


class LoRALoadError(LoRAError):
    """LoRA loading failed"""

    def __init__(self, lora_id: str, reason: str = ""):
        message = f"Failed to load LoRA: {lora_id}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, lora_id, "LORA_LOAD_ERROR")


class ContentBlockedError(SafetyError):
    """Content blocked by safety filter"""

    def __init__(self, reason: str, content_type: str = ""):
        super().__init__(f"Content blocked: {reason}", content_type, "CONTENT_BLOCKED")


class ResourceError(MultiModalLabError):
    """Resource management errors"""

    def __init__(
        self, resource_type: str, message: str, error_code: str = "RESOURCE_ERROR"
    ):
        super().__init__(
            f"{resource_type}: {message}", error_code, {"resource_type": resource_type}
        )


# Error handler decorators
def handle_cuda_oom(func):
    """Decorator to handle CUDA OOM with graceful fallback"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Try to determine model name from args
                model_name = ""
                if hasattr(args[0], "model_name"):
                    model_name = args[0].model_name

                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                raise CUDAOutOfMemoryError(model_name) from e
            raise
        except Exception as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                model_name = ""
                if hasattr(args[0], "model_name"):
                    model_name = args[0].model_name

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                raise CUDAOutOfMemoryError(model_name) from e
            raise

    return wrapper


def handle_model_error(func):
    """Decorator to standardize model error handling"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            model_name = ""
            if hasattr(args[0], "model_name"):
                model_name = args[0].model_name
            raise ModelNotFoundError(model_name) from e
        except (ImportError, AttributeError) as e:
            model_name = ""
            if hasattr(args[0], "model_name"):
                model_name = args[0].model_name
            raise ModelLoadError(model_name, str(e)) from e
        except Exception as e:
            if "out of memory" in str(e).lower():
                model_name = ""
                if hasattr(args[0], "model_name"):
                    model_name = args[0].model_name

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                raise CUDAOutOfMemoryError(model_name) from e
            raise

    return wrapper


def handle_validation_error(func):
    """Decorator to handle validation errors"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            # Convert ValueError to ValidationError for consistency
            raise ValidationError("input", str(e), "Value error") from e
        except TypeError as e:
            raise ValidationError("type", str(e), "Type error") from e

    return wrapper


def handle_api_errors(func):
    """處理 API 錯誤的通用裝飾器"""

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ValidationError:
            raise  # Re-raise validation errors as-is
        except ModelError:
            raise  # Re-raise model errors as-is
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise APIError(f"Internal server error: {str(e)}") from e

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValidationError:
            raise  # Re-raise validation errors as-is
        except ModelError:
            raise  # Re-raise model errors as-is
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise APIError(f"Internal server error: {str(e)}") from e

    # Return appropriate wrapper based on function type
    import asyncio

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper
