# core/t2i/__init__.py
"""
Text-to-Image core functionality
Exports all main classes for the T2I module
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _optional_import(module_name: str, attr_name: str):
    try:
        module = __import__(f"{__name__}.{module_name}", fromlist=[attr_name])
        return getattr(module, attr_name)
    except Exception as exc:  # noqa: BLE001
        logger.debug("core.t2i optional import failed: %s.%s (%s)", module_name, attr_name, exc)
        return None


# Core engine and pipeline (optional in lite/test environments)
T2IEngine = _optional_import("engine", "T2IEngine")  # type: ignore[assignment]
PipelineManager = _optional_import("pipeline", "PipelineManager")  # type: ignore[assignment]

# Component managers
LoRAManager = _optional_import("lora_manager", "LoRAManager")  # type: ignore[assignment]
ControlNetManager = _optional_import("controlnet", "ControlNetManager")  # type: ignore[assignment]
ModelConfigManager = _optional_import("model_config", "ModelConfigManager")  # type: ignore[assignment]

# Utilities
MemoryOptimizer = _optional_import("memory_utils", "MemoryOptimizer")  # type: ignore[assignment]
PromptProcessor = _optional_import("prompt_utils", "PromptProcessor")  # type: ignore[assignment]

# Public API exports
__all__ = [
    # Main engine
    "T2IEngine",
    # Pipeline management
    "PipelineManager",
    # Component managers
    "LoRAManager",
    "ControlNetManager",
    "ModelConfigManager",
    # Utilities
    "MemoryOptimizer",
    "PromptProcessor",
]

# Version info
__version__ = "1.0.0"


# Module-level convenience functions
def get_t2i_pipeline(
    model_id: str = "runwayml/stable-diffusion-v1-5",
    cache_root: str = "./cache",
    device: str = "auto",
):
    """
    Convenience function to get a T2I pipeline quickly

    Args:
        model_id: Model to load
        cache_root: Cache directory
        device: Device to use

    Returns:
        Configured PipelineManager instance
    """
    return PipelineManager(cache_root, device)


def save_image_to_cache(image, cache_root: str = "./cache", filename: str = None):  # type: ignore
    """
    Convenience function to save image to cache

    Args:
        image: PIL Image to save
        cache_root: Cache directory
        filename: Optional filename

    Returns:
        Path to saved image
    """
    from pathlib import Path
    from datetime import datetime

    from core.shared_cache import get_shared_cache

    cache_path = Path(get_shared_cache().get_path("OUTPUT_DIR")) / "t2i"
    cache_path.mkdir(parents=True, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_{timestamp}.png"

    output_path = cache_path / filename
    image.save(output_path)

    return str(output_path)
