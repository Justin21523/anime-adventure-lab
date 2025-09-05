# core/exceptions.py
"""
Unified Exception Classes
Standardized error handling across all modules
"""
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


class MultiModalLabError(Exception):
    """Base exception for Multi-Modal Lab"""

    def __init__(self, message: str, error_code: str = "UNKNOWN", details: dict = None):  # type: ignore
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


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


class GameError(MultiModalLabError):
    """Text adventure game errors"""

    def __init__(
        self, message: str, session_id: str = "", error_code: str = "GAME_ERROR"
    ):
        super().__init__(message, error_code, {"session_id": session_id})


class SessionNotFoundError(GameError):
    """Game session not found"""

    def __init__(self, session_id: str):
        super().__init__(
            f"Game session not found: {session_id}", session_id, "SESSION_NOT_FOUND"
        )


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


class SafetyError(MultiModalLabError):
    """Safety filter errors"""

    def __init__(
        self, message: str, content_type: str = "", error_code: str = "SAFETY_ERROR"
    ):
        super().__init__(message, error_code, {"content_type": content_type})


class ContentBlockedError(SafetyError):
    """Content blocked by safety filter"""

    def __init__(self, reason: str, content_type: str = ""):
        super().__init__(f"Content blocked: {reason}", content_type, "CONTENT_BLOCKED")


class ValidationError(MultiModalLabError):
    """Input validation errors"""

    def __init__(self, field: str, value: Any, reason: str = ""):
        message = f"Validation failed for field '{field}': {value}"
        if reason:
            message += f" - {reason}"
        super().__init__(
            message, "VALIDATION_ERROR", {"field": field, "value": str(value)}
        )


class ResourceError(MultiModalLabError):
    """Resource management errors"""

    def __init__(
        self, resource_type: str, message: str, error_code: str = "RESOURCE_ERROR"
    ):
        super().__init__(
            f"{resource_type}: {message}", error_code, {"resource_type": resource_type}
        )


class CacheError(ResourceError):
    """Cache-related errors"""

    def __init__(self, message: str):
        super().__init__("Cache", message, "CACHE_ERROR")


class StorageError(ResourceError):
    """Storage-related errors"""

    def __init__(self, message: str):
        super().__init__("Storage", message, "STORAGE_ERROR")


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
