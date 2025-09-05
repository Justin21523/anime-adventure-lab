# core/vlm/engine.py
"""
Vision-Language Model Engine
Handles caption generation and visual question answering
"""

import base64
import io
import torch
import logging
from PIL import Image
from typing import Union, Optional, Dict, Any, List

from ..exceptions import (
    VLMError,
    ImageProcessingError,
    ModelLoadError,
    handle_cuda_oom,
    handle_model_error,
)
from ..config import get_config
from ..shared_cache import get_shared_cache
from ..utils.image import ImageProcessor
from .model_manager import VLMModelManager
from .caption_pipeline import CaptionPipeline
from .vqa_pipeline import VQAPipeline

logger = logging.getLogger(__name__)


class VLMEngine:
    """Vision-Language Model engine for caption and VQA"""

    def __init__(self):
        self.config = get_config()
        self.cache = get_shared_cache()
        self.image_processor = ImageProcessor()

        # Initialize managers and pipelines
        self.model_manager = VLMModelManager(self.config, self.cache)
        self.caption_pipeline = CaptionPipeline(
            self.model_manager, self.image_processor
        )
        self.vqa_pipeline = VQAPipeline(self.model_manager, self.image_processor)

    def load_caption_model(self, model_name: Optional[str] = None) -> None:
        """Load image captioning model"""
        self.model_manager.load_caption_model(model_name)

    def load_vqa_model(self, model_name: Optional[str] = None) -> None:
        """Load VQA model"""
        self.model_manager.load_vqa_model(model_name)

    @handle_cuda_oom
    @handle_model_error
    def caption(
        self,
        image: Union[str, bytes, Image.Image],
        max_length: int = 50,
        num_beams: int = 3,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate image caption"""
        return self.caption_pipeline.generate_caption(
            image=image, max_length=max_length, num_beams=num_beams, **kwargs
        )

    @handle_cuda_oom
    @handle_model_error
    def vqa(
        self,
        image: Union[str, bytes, Image.Image],
        question: str,
        max_length: int = 100,
        **kwargs,
    ) -> Dict[str, Any]:
        """Answer question about image"""
        return self.vqa_pipeline.answer_question(
            image=image, question=question, max_length=max_length, **kwargs
        )

    def unload_models(self) -> None:
        """Unload all VLM models to free memory"""
        self.model_manager.unload_all_models()

    def get_status(self) -> Dict[str, Any]:
        """Get VLM engine status"""
        return self.model_manager.get_status()


# Global VLM engine instance
_vlm_engine: Optional[VLMEngine] = None


def get_vlm_engine() -> VLMEngine:
    """Get global VLM engine instance"""
    global _vlm_engine
    if _vlm_engine is None:
        _vlm_engine = VLMEngine()
    return _vlm_engine
