# core/vlm/model_manager.py
"""
VLM Model Manager - Handles model loading, unloading and memory management
"""

import torch
import logging
from typing import Optional, Dict, Any, Tuple
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    Blip2Processor,
    Blip2ForConditionalGeneration,
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    AutoProcessor,
    AutoModelForVision2Seq,
)

from ..exceptions import ModelLoadError, handle_cuda_oom, handle_model_error

logger = logging.getLogger(__name__)


class VLMModelManager:
    """Manages VLM model lifecycle and memory"""

    def __init__(self, config, cache):
        self.config = config
        self.cache = cache

        # Model instances
        self._caption_model = None
        self._caption_processor = None
        self._vqa_model = None
        self._vqa_processor = None

        # Model states
        self._caption_loaded = False
        self._vqa_loaded = False

    @handle_cuda_oom
    @handle_model_error
    def load_caption_model(self, model_name: Optional[str] = None) -> None:
        """Load image captioning model (BLIP-2)"""
        if self._caption_loaded:
            logger.info("Caption model already loaded")
            return

        model_name = model_name or self.config.model.caption_model

        try:
            logger.info(f"Loading caption model: {model_name}")

            # Setup low VRAM configuration
            device_map = self._get_device_map()
            torch_dtype = self._get_torch_dtype()

            cache_dir = self.cache.cache_root / "hf"

            # Load processor and model based on model type
            if "blip2" in model_name.lower():
                self._caption_processor = Blip2Processor.from_pretrained(
                    model_name, cache_dir=cache_dir
                )
                self._caption_model = Blip2ForConditionalGeneration.from_pretrained(
                    model_name,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    cache_dir=cache_dir,
                    low_cpu_mem_usage=True,
                    load_in_8bit=(
                        self.config.performance.use_8bit
                        if hasattr(self.config.performance, "use_8bit")
                        else False
                    ),
                )
            elif "blip" in model_name.lower():
                self._caption_processor = BlipProcessor.from_pretrained(
                    model_name, cache_dir=cache_dir
                )
                self._caption_model = BlipForConditionalGeneration.from_pretrained(
                    model_name,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    cache_dir=cache_dir,
                    low_cpu_mem_usage=True,
                )
            else:
                # Fallback to auto
                self._caption_processor = AutoProcessor.from_pretrained(
                    model_name, cache_dir=cache_dir
                )
                self._caption_model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    cache_dir=cache_dir,
                    low_cpu_mem_usage=True,
                )

            # Enable memory optimizations
            if hasattr(self._caption_model, "enable_vae_slicing"):
                self._caption_model.enable_vae_slicing()  # type: ignore
            if hasattr(self._caption_model, "enable_attention_slicing"):
                self._caption_model.enable_attention_slicing()  # type: ignore

            self._caption_loaded = True
            logger.info(f"Caption model loaded successfully: {model_name}")

        except Exception as e:
            logger.error(f"Failed to load caption model: {e}")
            raise ModelLoadError(f"Caption model loading failed: {str(e)}")

    @handle_cuda_oom
    @handle_model_error
    def load_vqa_model(self, model_name: Optional[str] = None) -> None:
        """Load VQA model (LLaVA/Qwen-VL)"""
        if self._vqa_loaded:
            logger.info("VQA model already loaded")
            return

        model_name = model_name or self.config.model.vqa_model

        try:
            logger.info(f"Loading VQA model: {model_name}")

            device_map = self._get_device_map()
            torch_dtype = self._get_torch_dtype()
            cache_dir = self.cache.cache_root / "hf"

            # Load processor and model based on model type
            if "llava" in model_name.lower():
                self._vqa_processor = LlavaNextProcessor.from_pretrained(
                    model_name, cache_dir=cache_dir
                )
                self._vqa_model = LlavaNextForConditionalGeneration.from_pretrained(
                    model_name,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    cache_dir=cache_dir,
                    low_cpu_mem_usage=True,
                    load_in_8bit=(
                        self.config.performance.use_8bit
                        if hasattr(self.config.performance, "use_8bit")
                        else False
                    ),
                )
            else:
                # Fallback to auto
                self._vqa_processor = AutoProcessor.from_pretrained(
                    model_name, cache_dir=cache_dir
                )
                self._vqa_model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    cache_dir=cache_dir,
                    low_cpu_mem_usage=True,
                )

            # Enable memory optimizations
            if hasattr(self._vqa_model, "enable_vae_slicing"):
                self._vqa_model.enable_vae_slicing()  # type: ignore
            if hasattr(self._vqa_model, "enable_attention_slicing"):
                self._vqa_model.enable_attention_slicing()  # type: ignore

            self._vqa_loaded = True
            logger.info(f"VQA model loaded successfully: {model_name}")

        except Exception as e:
            logger.error(f"Failed to load VQA model: {e}")
            raise ModelLoadError(f"VQA model loading failed: {str(e)}")

    def unload_caption_model(self) -> None:
        """Unload caption model"""
        if self._caption_model is not None:
            del self._caption_model
            del self._caption_processor
            self._caption_model = None
            self._caption_processor = None
            self._caption_loaded = False
            logger.info("Caption model unloaded")

    def unload_vqa_model(self) -> None:
        """Unload VQA model"""
        if self._vqa_model is not None:
            del self._vqa_model
            del self._vqa_processor
            self._vqa_model = None
            self._vqa_processor = None
            self._vqa_loaded = False
            logger.info("VQA model unloaded")

    def unload_all_models(self) -> None:
        """Unload all VLM models to free memory"""
        self.unload_caption_model()
        self.unload_vqa_model()

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("All VLM models unloaded")

    def get_models(
        self,
    ) -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[Any]]:
        """Get current model instances"""
        return (
            self._caption_model,
            self._caption_processor,
            self._vqa_model,
            self._vqa_processor,
        )

    def get_status(self) -> Dict[str, Any]:
        """Get model manager status"""
        status = {
            "caption_model_loaded": self._caption_loaded,
            "vqa_model_loaded": self._vqa_loaded,
            "caption_model": (
                self.config.model.caption_model if self._caption_loaded else None
            ),
            "vqa_model": self.config.model.vqa_model if self._vqa_loaded else None,
        }

        # Add GPU memory info if available
        if torch.cuda.is_available():
            status["gpu_memory"] = {
                "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "cached_gb": torch.cuda.memory_reserved() / 1e9,
            }

        return status

    def _get_device_map(self) -> str:
        """Get optimal device mapping"""
        if torch.cuda.is_available():
            return "auto"
        return "cpu"

    def _get_torch_dtype(self):
        """Get optimal torch dtype"""
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
