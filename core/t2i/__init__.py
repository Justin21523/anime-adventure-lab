# core/t2i/__init__.py
"""
Text-to-Image core functionality
Exports all main classes for the T2I module
"""

# Core engine and pipeline
from .engine import T2IEngine
from .pipeline import PipelineManager

# Component managers
from .lora_manager import LoRAManager
from .controlnet import ControlNetManager
from .model_config import ModelConfigManager

# Utilities
from .memory_utils import MemoryOptimizer
from .prompt_utils import PromptProcessor

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

    cache_path = Path(cache_root) / "outputs" / "t2i"
    cache_path.mkdir(parents=True, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_{timestamp}.png"

    output_path = cache_path / filename
    image.save(output_path)

    return str(output_path)
