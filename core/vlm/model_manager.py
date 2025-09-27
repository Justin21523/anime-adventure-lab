# core/vlm/model_manager.py
"""
VLM Model Manager - Handles model loading, unloading and memory management
"""

import torch
import logging
import gc
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    Blip2Processor,
    Blip2ForConditionalGeneration,
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    AutoProcessor,
    AutoModelForVision2Seq,
    BitsAndBytesConfig,
)

from ..config import get_config
from ..shared_cache import get_shared_cache
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

        self._caption_model_name = None
        self._vqa_model_name = None

        # Model states
        self._caption_loaded = False
        self._vqa_loaded = False

        # Default models
        self.default_caption_model = "Salesforce/blip2-opt-2.7b"
        self.default_vqa_model = "llava-hf/llava-1.5-7b-hf"

    def _get_device_map(self):
        """Get optimal device mapping based on available hardware"""
        if self.config.models.device == "auto":
            return "auto" if torch.cuda.is_available() else None
        else:
            return self.config.models.device

    def _get_torch_dtype(self) -> torch.dtype:
        """Get optimal torch dtype for current hardware"""
        if self.config.models.precision == "fp16":
            return torch.float16
        elif self.config.models.precision == "bf16":
            return torch.bfloat16
        else:
            return torch.float32

    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization config for low VRAM scenarios"""
        if not torch.cuda.is_available():
            return None

        if (
            hasattr(self.config.performance, "use_8bit")
            and self.config.performance.use_8bit
        ):
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )

        if (
            hasattr(self.config.performance, "use_4bit")
            and self.config.performance.use_4bit
        ):
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

        return None

    @handle_cuda_oom
    @handle_model_error
    def load_caption_model(self, model_name: Optional[str] = None) -> None:
        """Load image captioning model (BLIP-2)"""
        if self._caption_loaded:
            logger.info("Caption model already loaded")
            return

        model_name = model_name or self.config.model.caption_model
        self._caption_model_name = model_name

        try:
            logger.info(f"Loading caption model: {model_name}")

            # Setup configuration
            device_map = self._get_device_map()
            torch_dtype = self._get_torch_dtype()
            quantization_config = self._get_quantization_config()
            cache_dir = str(self.cache.cache_root / "hf")

            # Model loading kwargs
            model_kwargs = {
                "device_map": device_map,
                "torch_dtype": torch_dtype,
                "cache_dir": cache_dir,
                "low_cpu_mem_usage": True,
            }

            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config

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
            self._enable_memory_optimizations(self._caption_model)

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
        self._vqa_model_name = model_name

        try:
            logger.info(f"Loading VQA model: {model_name}")

            device_map = self._get_device_map()
            torch_dtype = self._get_torch_dtype()
            quantization_config = self._get_quantization_config()
            cache_dir = str(self.cache.cache_root / "hf")

            model_kwargs = {
                "device_map": device_map,
                "torch_dtype": torch_dtype,
                "cache_dir": cache_dir,
                "low_cpu_mem_usage": True,
            }

            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config

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
            elif "qwen" in model_name.lower():
                self._vqa_processor = AutoProcessor.from_pretrained(
                    model_name, cache_dir=cache_dir, trust_remote_code=True
                )
                self._vqa_model = AutoModelForVision2Seq.from_pretrained(
                    model_name, trust_remote_code=True, **model_kwargs
                )
            else:
                # Generic fallback
                self._vqa_processor = AutoProcessor.from_pretrained(
                    model_name, cache_dir=cache_dir
                )
                self._vqa_model = AutoModelForVision2Seq.from_pretrained(
                    model_name, **model_kwargs
                )

            # Enable memory optimizations
            self._enable_memory_optimizations(self._vqa_model)

            self._vqa_loaded = True
            logger.info(f"VQA model loaded successfully: {model_name}")

        except Exception as e:
            logger.error(f"Failed to load VQA model: {e}")
            raise ModelLoadError(f"VQA model loading failed: {str(e)}")

    def _enable_memory_optimizations(self, model):
        """Enable memory optimizations for the model"""
        try:
            if hasattr(model, "enable_vae_slicing"):
                model.enable_vae_slicing()
            if hasattr(model, "enable_attention_slicing"):
                model.enable_attention_slicing()
            if hasattr(model, "enable_xformers_memory_efficient_attention"):
                try:
                    model.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass  # xformers not available
        except Exception as e:
            logger.warning(f"Failed to enable some memory optimizations: {e}")

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

    def _cleanup_caption_model(self):
        """Clean up caption model from memory"""
        if self._caption_model is not None:
            del self._caption_model
            self._caption_model = None
        if self._caption_processor is not None:
            del self._caption_processor
            self._caption_processor = None
        self._caption_loaded = False

    def _cleanup_vqa_model(self):
        """Clean up VQA model from memory"""
        if self._vqa_model is not None:
            del self._vqa_model
            self._vqa_model = None
        if self._vqa_processor is not None:
            del self._vqa_processor
            self._vqa_processor = None
        self._vqa_loaded = False

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
        logger.info("Unloading all VLM models")

        self._cleanup_caption_model()
        self._cleanup_vqa_model()

        # Force garbage collection and CUDA cache cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("All VLM models unloaded")

    def get_status(self) -> Dict[str, Any]:
        """Get VLM model manager status"""
        gpu_memory = {}
        if torch.cuda.is_available():
            gpu_memory = {
                "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                "cached": torch.cuda.memory_reserved() / 1024**3,  # GB
                "total": torch.cuda.get_device_properties(0).total_memory
                / 1024**3,  # GB
            }

        return {
            "caption_loaded": self._caption_loaded,
            "vqa_loaded": self._vqa_loaded,
            "gpu_available": torch.cuda.is_available(),
            "gpu_memory": gpu_memory,
            "loaded_models": {
                "caption": (
                    getattr(self._caption_model, "name_or_path", None)
                    if self._caption_model
                    else None
                ),
                "vqa": (
                    getattr(self._vqa_model, "name_or_path", None)
                    if self._vqa_model
                    else None
                ),
            },
        }
