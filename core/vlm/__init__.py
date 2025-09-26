# core/vlm/__init__.py
"""
Vision-Language Model (VLM) Module

This module provides comprehensive vision-language capabilities including:
- Image captioning using BLIP-2 and similar models
- Visual Question Answering (VQA) using LLaVA, Qwen-VL
- Advanced image preprocessing and quality assessment
- Chinese language support for questions and responses
- Memory-efficient model management with quantization support
- Batch processing capabilities
- Safety filtering and content validation

Key Components:
- VLMEngine: Main interface for all VLM operations
- ModelManager: Handles model loading, unloading, and memory optimization
- CaptionPipeline: Specialized image captioning with multiple strategies
- VQAPipeline: Visual question answering with conversation support
- VLMImageProcessor: Advanced image preprocessing and quality assessment
- VLMTextProcessor: Text preprocessing and response postprocessing

Usage Examples:
    # Basic captioning
    from core.vlm import get_vlm_engine
    engine = get_vlm_engine()
    result = engine.caption("image.jpg", max_length=50)

    # Visual Question Answering
    result = engine.vqa("image.jpg", "這張圖片中有什麼？", max_length=100)

    # Batch processing
    results = engine.batch_process(images, questions)
"""

from .engine import VLMEngine, get_vlm_engine
from .model_manager import VLMModelManager
from .caption_pipeline import CaptionPipeline
from .vqa_pipeline import VQAPipeline
from .processors import (
    VLMImageProcessor,
    VLMTextProcessor,
    TextProcessor,
    PromptTemplate,
    ModelCompatibility,
    OutputFormatter,
)

__all__ = [
    # Main engine
    "VLMEngine",
    "get_vlm_engine",
    # Core components
    "VLMModelManager",
    "CaptionPipeline",
    "VQAPipeline",
    # Processors
    "VLMImageProcessor",
    "VLMTextProcessor",
    "TextProcessor",
    "PromptTemplate",
    "ModelCompatibility",
    "OutputFormatter",
]


# Version info
__version__ = "1.0.0"
__author__ = "Multi-Modal Lab Team"

# Model defaults - can be overridden via config
DEFAULT_MODELS = {
    "caption": "Salesforce/blip2-opt-2.7b",
    "vqa": "llava-hf/llava-1.5-7b-hf",
    "vqa_chinese": "Qwen/Qwen-VL-Chat",  # Better Chinese support
}

# Supported image formats
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

# Quality thresholds
QUALITY_THRESHOLDS = {
    "min_resolution": (224, 224),
    "max_resolution": (2048, 2048),
    "min_quality_score": 0.3,
    "blur_threshold": 50,
    "brightness_range": (30, 230),
    "contrast_min": 15,
}

# Language support configuration
LANGUAGE_CONFIG = {
    "supported_languages": ["zh-TW", "zh-CN", "en", "ja", "ko"],
    "default_language": "zh-TW",
    "enable_auto_translation": True,
    "chinese_variants": {"traditional": "zh-TW", "simplified": "zh-CN"},
}

# Safety configuration
SAFETY_CONFIG = {
    "enable_content_filter": True,
    "enable_face_blur": False,  # Optional privacy protection
    "max_question_length": 200,
    "max_response_length": 500,
    "blocked_keywords": ["violence", "暴力", "blood", "血腥", "nude", "裸體"],
}


def get_model_info() -> dict:
    """Get information about available VLM models"""
    return {
        "default_models": DEFAULT_MODELS,
        "supported_formats": list(SUPPORTED_IMAGE_FORMATS),
        "quality_thresholds": QUALITY_THRESHOLDS,
        "language_config": LANGUAGE_CONFIG,
        "safety_config": SAFETY_CONFIG,
    }


def validate_image_format(filename: str) -> bool:
    """Validate if image format is supported"""
    import os

    _, ext = os.path.splitext(filename.lower())
    return ext in SUPPORTED_IMAGE_FORMATS


def create_vlm_config(
    caption_model: str = None,  # type: ignore
    vqa_model: str = None,  # type: ignore
    enable_chinese: bool = True,
    enable_safety: bool = True,
    max_vram_usage: float = 0.8,
    use_quantization: bool = False,
) -> dict:
    """Create VLM configuration dictionary"""
    config = {
        "models": {
            "caption_model": caption_model or DEFAULT_MODELS["caption"],
            "vqa_model": vqa_model
            or (
                DEFAULT_MODELS["vqa_chinese"]
                if enable_chinese
                else DEFAULT_MODELS["vqa"]
            ),
        },
        "performance": {
            "max_vram_usage": max_vram_usage,
            "use_8bit": use_quantization,
            "enable_attention_slicing": True,
            "enable_vae_slicing": True,
            "low_cpu_mem_usage": True,
        },
        "processing": {
            "max_image_size": 1024,
            "min_image_size": 224,
            "enable_auto_enhance": True,
            "quality_threshold": 0.7,
        },
        "language": {
            "enable_chinese_support": enable_chinese,
            "default_language": "zh-TW" if enable_chinese else "en",
            "max_question_length": 200,
            "max_response_length": 500,
        },
        "safety": {
            "enable_content_filter": enable_safety,
            "enable_nsfw_filter": enable_safety,
            "blocked_keywords": (
                SAFETY_CONFIG["blocked_keywords"] if enable_safety else []
            ),
        },
    }
    return config


# Utility functions for common operations
def quick_caption(image_path: str, **kwargs) -> str:
    """Quick caption generation for single image"""
    engine = get_vlm_engine()
    result = engine.caption(image_path, **kwargs)
    return result.get("caption", "無法生成描述")


def quick_vqa(image_path: str, question: str, **kwargs) -> str:
    """Quick VQA for single image and question"""
    engine = get_vlm_engine()
    result = engine.vqa(image_path, question, **kwargs)
    return result.get("answer", "無法回答問題")


# Module initialization checks
def _check_dependencies():
    """Check if required dependencies are available"""
    required_packages = ["torch", "transformers", "PIL", "cv2", "numpy"]

    missing_packages = []
    for package in required_packages:
        try:
            if package == "cv2":
                import cv2
            elif package == "PIL":
                from PIL import Image
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        raise ImportError(
            f"Missing required packages: {', '.join(missing_packages)}. "
            f"Please install them using: pip install {' '.join(missing_packages)}"
        )


# Initialize module
try:
    _check_dependencies()

    # Log module initialization
    import logging

    logger = logging.getLogger(__name__)
    logger.info(f"VLM module initialized (v{__version__})")
    logger.info(f"Default models: {DEFAULT_MODELS}")
    logger.info(f"Supported formats: {list(SUPPORTED_IMAGE_FORMATS)}")

except Exception as e:
    import warnings

    warnings.warn(f"VLM module initialization warning: {e}", UserWarning)

# Export configuration for other modules
MODULE_CONFIG = {
    "name": "vlm",
    "version": __version__,
    "description": "Vision-Language Model processing with Chinese support",
    "capabilities": [
        "image_captioning",
        "visual_question_answering",
        "batch_processing",
        "chinese_language_support",
        "quality_assessment",
        "safety_filtering",
    ],
    "models": DEFAULT_MODELS,
    "requirements": [
        "torch>=1.13.0",
        "transformers>=4.21.0",
        "Pillow>=8.0.0",
        "opencv-python>=4.5.0",
        "numpy>=1.21.0",
    ],
}
