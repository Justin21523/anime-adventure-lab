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

logger = logging.getLogger(__name__)


class VLMEngine:
    """Vision-Language Model engine for caption and VQA"""

    def __init__(self):
        self.config = get_config()
        self.cache = get_shared_cache()
        self.image_processor = ImageProcessor()

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
            return

        model_name = model_name or self.config.model.caption_model  # type: ignore

        try:
            logger.info(f"Loading caption model: {model_name}")

            # Setup low VRAM configuration
            device_map = "auto" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            # Load processor and model
            if "blip2" in model_name.lower():
                self._caption_processor = Blip2Processor.from_pretrained(
                    model_name, cache_dir=self.cache.cache_root / "hf"  # type: ignore
                )
                self._caption_model = Blip2ForConditionalGeneration.from_pretrained(
                    model_name,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    cache_dir=self.cache.cache_root / "hf",  # type: ignore
                    low_cpu_mem_usage=True,
                )
            else:
                # Fallback to BLIP
                self._caption_processor = BlipProcessor.from_pretrained(
                    model_name, cache_dir=self.cache.cache_root / "hf"  # type: ignore
                )
                self._caption_model = BlipForConditionalGeneration.from_pretrained(
                    model_name,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    cache_dir=self.cache.cache_root / "hf",  # type: ignore
                )

            # Enable optimizations
            if hasattr(self._caption_model, "enable_vae_slicing"):
                self._caption_model.enable_vae_slicing()  # type: ignore

            self._caption_loaded = True
            logger.info("Caption model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load caption model: {e}")
            raise ModelLoadError(model_name, str(e))

    @handle_cuda_oom
    @handle_model_error
    def load_vqa_model(self, model_name: Optional[str] = None) -> None:
        """Load Visual Question Answering model (LLaVA/Qwen-VL)"""
        if self._vqa_loaded:
            return

        model_name = model_name or self.config.model.vqa_model  # type: ignore

        try:
            logger.info(f"Loading VQA model: {model_name}")

            device_map = "auto" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            if "llava" in model_name.lower():
                self._vqa_processor = LlavaNextProcessor.from_pretrained(
                    model_name, cache_dir=self.cache.cache_root / "hf"  # type: ignore
                )
                self._vqa_model = LlavaNextForConditionalGeneration.from_pretrained(
                    model_name,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    cache_dir=self.cache.cache_root / "hf",  # type: ignore
                    low_cpu_mem_usage=True,
                )
            else:
                # Generic vision-to-seq model
                self._vqa_processor = AutoProcessor.from_pretrained(
                    model_name, cache_dir=self.cache.cache_root / "hf"  # type: ignore
                )
                self._vqa_model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    cache_dir=self.cache.cache_root / "hf",  # type: ignore
                )

            self._vqa_loaded = True
            logger.info("VQA model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load VQA model: {e}")
            raise ModelLoadError(model_name, str(e))

    def caption(
        self,
        image: Union[str, bytes, Image.Image],
        max_length: int = 50,
        num_beams: int = 3,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate image caption"""
        if not self._caption_loaded:
            self.load_caption_model()

        try:
            # Process image
            pil_image = self.image_processor.load_image(image)

            # Generate caption
            inputs = self._caption_processor(pil_image, return_tensors="pt")  # type: ignore

            # Move to model device
            device = next(self._caption_model.parameters()).device  # type: ignore
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._caption_model.generate(  # type: ignore
                    **inputs,
                    max_new_tokens=max_length,
                    num_beams=num_beams,
                    do_sample=False,
                    **kwargs,
                )

            caption = self._caption_processor.decode(  # type: ignore
                outputs[0], skip_special_tokens=True
            )

            # Clean up caption (remove prompt prefix if present)
            if caption.startswith("a photo of"):
                caption = caption[len("a photo of") :].strip()

            return {
                "caption": caption,
                "confidence": 0.9,  # Placeholder - real confidence needs additional computation
                "model_used": self._caption_model.config.name_or_path,  # type: ignore
                "parameters": {"max_length": max_length, "num_beams": num_beams},
            }

        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            raise VLMError(f"Caption generation failed: {str(e)}")

    def vqa(
        self,
        image: Union[str, bytes, Image.Image],
        question: str,
        max_length: int = 100,
        **kwargs,
    ) -> Dict[str, Any]:
        """Visual Question Answering"""
        if not self._vqa_loaded:
            self.load_vqa_model()

        if not question.strip():
            raise VLMError("Question cannot be empty")

        try:
            # Process image and question
            pil_image = self.image_processor.load_image(image)

            # Format prompt for LLaVA
            if "llava" in self._vqa_model.config.name_or_path.lower():  # type: ignore
                prompt = f"USER: <image>\n{question}\nASSISTANT:"
            else:
                prompt = question

            inputs = self._vqa_processor(
                text=prompt, images=pil_image, return_tensors="pt"  # type: ignore
            )

            # Move to model device
            device = next(self._vqa_model.parameters()).device  # type: ignore
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._vqa_model.generate(  # type: ignore
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=0.7,
                    **kwargs,
                )

            # Decode answer
            answer = self._vqa_processor.decode(  # type: ignore
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()

            return {
                "question": question,
                "answer": answer,
                "confidence": 0.85,  # Placeholder
                "model_used": self._vqa_model.config.name_or_path,  # type: ignore
                "parameters": {"max_length": max_length},
            }

        except Exception as e:
            logger.error(f"VQA failed: {e}")
            raise VLMError(f"VQA failed: {str(e)}")

    def unload_models(self) -> None:
        """Unload all VLM models to free memory"""
        if self._caption_model is not None:
            del self._caption_model
            del self._caption_processor
            self._caption_model = None
            self._caption_processor = None
            self._caption_loaded = False

        if self._vqa_model is not None:
            del self._vqa_model
            del self._vqa_processor
            self._vqa_model = None
            self._vqa_processor = None
            self._vqa_loaded = False

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("All VLM models unloaded")

    def get_status(self) -> Dict[str, Any]:
        """Get VLM engine status"""
        return {
            "caption_model_loaded": self._caption_loaded,
            "vqa_model_loaded": self._vqa_loaded,
            "caption_model": (
                self.config.model.caption_model if self._caption_loaded else None  # type: ignore
            ),
            "vqa_model": self.config.model.vqa_model if self._vqa_loaded else None,  # type: ignore
        }


# Global VLM engine instance
_vlm_engine: Optional[VLMEngine] = None


def get_vlm_engine() -> VLMEngine:
    """Get global VLM engine instance"""
    global _vlm_engine
    if _vlm_engine is None:
        _vlm_engine = VLMEngine()
    return _vlm_engine
