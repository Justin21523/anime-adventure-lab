# core/vlm/captioner.py
"""
VLM Captioner with multi-model support and low-VRAM optimization
Supports BLIP-2, LLaVA, and Qwen-VL with automatic model selection
"""

import torch
import gc
from typing import Dict, List, Optional, Union, Any
from PIL import Image
from abc import ABC, abstractmethod
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoProcessor,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import logging

logger = logging.getLogger(__name__)


class VLMAdapter(ABC):
    """Abstract base class for VLM models"""

    @abstractmethod
    def caption(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        pass

    @abstractmethod
    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        pass

    @abstractmethod
    def unload(self):
        pass


class BLIP2Captioner(VLMAdapter):
    """BLIP-2 model for image captioning"""

    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        device: str = "auto",
        low_vram: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.low_vram = low_vram
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load BLIP-2 model with low-VRAM optimizations"""
        try:
            # Configure for low VRAM
            if self.low_vram:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                quantization_config = None

            self.processor = BlipProcessor.from_pretrained(self.model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map=self.device if not self.low_vram else "auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )

            if not self.low_vram and self.device != "auto":
                self.model = self.model.to(self.device)

            logger.info(f"BLIP-2 model loaded: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to load BLIP-2 model: {e}")
            raise

    def caption(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        """Generate caption for image"""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Default prompts for different scenarios
        if prompt is None:
            prompt = "a detailed description of this image"

        # Process image and text
        inputs = self.processor(image, prompt, return_tensors="pt")

        # Move to device if not using device_map
        if not self.low_vram and self.device != "auto":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate caption
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=100,
                num_beams=3,
                do_sample=False,
                temperature=0.7,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        return caption.strip()

    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        """Structured analysis of image content"""
        results = {}

        # Multiple analysis prompts
        analysis_prompts = {
            "general": "Describe this image in detail",
            "character": "Describe the character or person in this image, including appearance, clothing, and expression",
            "scene": "Describe the background, setting, and environment in this image",
            "mood": "What is the mood or atmosphere of this image?",
            "objects": "List the main objects and items visible in this image",
        }

        for category, prompt in analysis_prompts.items():
            try:
                results[category] = self.caption(image, prompt)
            except Exception as e:
                logger.warning(f"Analysis failed for {category}: {e}")
                results[category] = ""

        return results

    def unload(self):
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("BLIP-2 model unloaded")


class LLaVACaptioner(VLMAdapter):
    """LLaVA model for image captioning"""

    def __init__(
        self,
        model_name: str = "liuhaotian/llava-v1.6-mistral-7b",
        device: str = "auto",
        low_vram: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.low_vram = low_vram
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load LLaVA model with optimizations"""
        try:
            if self.low_vram:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,  # LLaVA works better with 8-bit
                    load_in_4bit=False,
                )
            else:
                quantization_config = None

            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map=self.device if not self.low_vram else "auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

            logger.info(f"LLaVA model loaded: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to load LLaVA model: {e}")
            raise

    def caption(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        """Generate caption using LLaVA"""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        if prompt is None:
            prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"

        inputs = self.processor(prompt, image, return_tensors="pt")

        if not self.low_vram and self.device != "auto":
            inputs = {
                k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)
            }

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        # Extract assistant response
        if "ASSISTANT:" in response:
            caption = response.split("ASSISTANT:")[-1].strip()
        else:
            caption = response.strip()

        return caption

    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        """Structured analysis using LLaVA"""
        results = {}

        analysis_prompts = {
            "general": "USER: <image>\nProvide a detailed description of this image.\nASSISTANT:",
            "character": "USER: <image>\nDescribe the character in this image, including their appearance, clothing, and facial expression.\nASSISTANT:",
            "scene": "USER: <image>\nDescribe the setting and environment shown in this image.\nASSISTANT:",
            "mood": "USER: <image>\nWhat mood or emotion does this image convey?\nASSISTANT:",
            "style": "USER: <image>\nDescribe the artistic style and visual characteristics of this image.\nASSISTANT:",
        }

        for category, prompt in analysis_prompts.items():
            try:
                results[category] = self.caption(image, prompt)
            except Exception as e:
                logger.warning(f"LLaVA analysis failed for {category}: {e}")
                results[category] = ""

        return results

    def unload(self):
        """Unload LLaVA model"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("LLaVA model unloaded")


class VLMCaptioner:
    """Unified VLM captioner with multiple model support"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.current_model = None
        self.model_cache = {}
        self.default_model = config.get("default_model", "blip2")

    def load_model(self, model_type: str) -> VLMAdapter:
        """Load specific VLM model"""
        if model_type in self.model_cache:
            return self.model_cache[model_type]

        model_configs = self.config.get("models", {})

        if model_type == "blip2":
            model_name = model_configs.get("blip2", "Salesforce/blip2-opt-2.7b")
            adapter = BLIP2Captioner(
                model_name=model_name,
                device=self.config.get("device", "auto"),
                low_vram=self.config.get("low_vram", True),
            )
        elif model_type == "llava":
            model_name = model_configs.get("llava", "liuhaotian/llava-v1.6-mistral-7b")
            adapter = LLaVACaptioner(
                model_name=model_name,
                device=self.config.get("device", "auto"),
                low_vram=self.config.get("low_vram", True),
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.model_cache[model_type] = adapter
        return adapter

    def caption(
        self,
        image: Image.Image,
        model_type: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> str:
        """Generate caption using specified or default model"""
        if model_type is None:
            model_type = self.default_model

        model = self.load_model(model_type)
        return model.caption(image, prompt)

    def analyze(
        self, image: Image.Image, model_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform structured analysis of image"""
        if model_type is None:
            model_type = self.default_model

        model = self.load_model(model_type)
        analysis = model.analyze(image)

        # Add metadata
        analysis["model_type"] = model_type
        analysis["image_size"] = image.size
        analysis["image_mode"] = image.mode

        return analysis

    def unload_model(self, model_type: str):
        """Unload specific model"""
        if model_type in self.model_cache:
            self.model_cache[model_type].unload()
            del self.model_cache[model_type]

    def unload_all(self):
        """Unload all models"""
        for model_type in list(self.model_cache.keys()):
            self.unload_model(model_type)
        gc.collect()
        torch.cuda.empty_cache()


# Example usage
if __name__ == "__main__":
    # Test configuration
    config = {
        "default_model": "blip2",
        "device": "auto",
        "low_vram": True,
        "models": {
            "blip2": "Salesforce/blip2-opt-2.7b",
            "llava": "liuhaotian/llava-v1.6-mistral-7b",
        },
    }

    # Initialize captioner
    captioner = VLMCaptioner(config)

    # Load test image
    image = Image.open("test_image.jpg")

    # Generate caption
    caption = captioner.caption(image)
    print(f"Caption: {caption}")

    # Perform analysis
    analysis = captioner.analyze(image)
    for category, result in analysis.items():
        print(f"{category}: {result}")
