"""
Core Multi-Modal Lab Components

Lightweight init only. Heavy dependencies (t2i, rag, pydantic_settings, redis, etc.)
are imported lazily via __getattr__ to keep agent/tests usable even when optional
packages are missing.
"""

__version__ = "0.1.0"
__author__ = "Multi-Modal Lab Team"


def __getattr__(name):
    """Lazy import heavy modules on demand."""
    if name in {"get_t2i_pipeline", "save_image_to_cache", "LoRAManager"}:
        from .t2i import get_t2i_pipeline, save_image_to_cache, LoRAManager

        return {
            "get_t2i_pipeline": get_t2i_pipeline,
            "save_image_to_cache": save_image_to_cache,
            "LoRAManager": LoRAManager,
        }[name]

    if name == "DocumentMemory":
        from .rag import DocumentMemory

        return DocumentMemory

    if name in {"get_config", "ModelConfig", "APIConfig", "SafetyConfig"}:
        from .config import get_config, ModelConfig, APIConfig, SafetyConfig

        return {
            "get_config": get_config,
            "ModelConfig": ModelConfig,
            "APIConfig": APIConfig,
            "SafetyConfig": SafetyConfig,
        }[name]

    if name in {
        "ValidationError",
        "SafetyError",
        "RateLimitError",
        "ImageProcessingError",
        "ModelLoadError",
        "ConfigurationError",
    }:
        from .exceptions import (
            ValidationError,
            SafetyError,
            RateLimitError,
            ImageProcessingError,
            ModelLoadError,
            ConfigurationError,
        )

        return {
            "ValidationError": ValidationError,
            "SafetyError": SafetyError,
            "RateLimitError": RateLimitError,
            "ImageProcessingError": ImageProcessingError,
            "ModelLoadError": ModelLoadError,
            "ConfigurationError": ConfigurationError,
        }[name]

    if name == "get_shared_cache":
        from .shared_cache import get_shared_cache

        return get_shared_cache

    if name in {"get_content_filter", "get_input_validator", "get_rate_limiter"}:
        from .safety import get_content_filter, get_input_validator, get_rate_limiter

        return {
            "get_content_filter": get_content_filter,
            "get_input_validator": get_input_validator,
            "get_rate_limiter": get_rate_limiter,
        }[name]

    if name in {"ImageProcessor", "CacheManager"}:
        from .utils.image import ImageProcessor
        from .utils.cache import CacheManager

        return {"ImageProcessor": ImageProcessor, "CacheManager": CacheManager}[name]

    raise AttributeError(f"module 'core' has no attribute '{name}'")


__all__ = [
    "get_t2i_pipeline",
    "save_image_to_cache",
    "LoRAManager",
    "DocumentMemory",
    "get_config",
    "ModelConfig",
    "APIConfig",
    "SafetyConfig",
    "ValidationError",
    "SafetyError",
    "RateLimitError",
    "ImageProcessingError",
    "ModelLoadError",
    "ConfigurationError",
    "get_shared_cache",
    "get_content_filter",
    "get_input_validator",
    "get_rate_limiter",
    "ImageProcessor",
    "CacheManager",
    "__version__",
    "__author__",
]
