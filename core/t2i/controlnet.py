# core/t2i/controlnet.py

import torch
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, Optional, List
from PIL import Image
from transformers import pipeline as hf_pipeline
from diffusers.pipelines.controlnet.pipeline_controlnet import (
    StableDiffusionControlNetPipeline,
)
from diffusers.models.controlnets.controlnet import ControlNetModel
from controlnet_aux import (
    OpenposeDetector,
    MidasDetector,
    CannyDetector,
    LineartDetector,
)

from ..shared_cache import get_shared_cache

logger = logging.getLogger(__name__)


class ControlNetManager:
    """Manage ControlNet preprocessing and loading"""

    def __init__(self, cache_root: str):
        self.cache = get_shared_cache()
        self.cache_root = Path(cache_root)
        self.controlnet_cache_dir = Path(self.cache.get_path("MODELS_CONTROLNET"))
        self.controlnet_cache_dir.mkdir(parents=True, exist_ok=True)
        self._processors = {}

        # Available ControlNet models
        self.available_controlnets = {
            "pose": {
                "model_id": "lllyasviel/sd-controlnet-openpose",
                "processor": "openpose",
                "description": "Human pose control using OpenPose detection",
            },
            "depth": {
                "model_id": "lllyasviel/sd-controlnet-depth",
                "processor": "midas",
                "description": "Depth-based structure control",
            },
            "canny": {
                "model_id": "lllyasviel/sd-controlnet-canny",
                "processor": "canny",
                "description": "Edge-based control using Canny detection",
            },
            "lineart": {
                "model_id": "lllyasviel/sd-controlnet-mlsd",
                "processor": "lineart",
                "description": "Line art and sketch control",
            },
            "scribble": {
                "model_id": "lllyasviel/sd-controlnet-scribble",
                "processor": "scribble",
                "description": "Scribble and rough sketch control",
            },
        }

        self.loaded_controlnets = {}
        self.loaded_processors = {}
        logger.info("ControlNetManager initialized")

    def list_available_controlnets(self) -> List[Dict]:
        """List all available ControlNet types"""
        return [
            {
                "type": key,
                "model_id": info["model_id"],
                "description": info["description"],
                "loaded": key in self.loaded_controlnets,
            }
            for key, info in self.available_controlnets.items()
        ]

    def load_controlnet(self, controlnet_type: str) -> ControlNetModel:
        """Load ControlNet model"""
        if controlnet_type not in self.available_controlnets:
            raise ValueError(f"Unknown ControlNet type: {controlnet_type}")

        # Return cached model if already loaded
        if controlnet_type in self.loaded_controlnets:
            return self.loaded_controlnets[controlnet_type]

        try:
            model_info = self.available_controlnets[controlnet_type]
            model_id = model_info["model_id"]

            logger.info(f"Loading ControlNet: {controlnet_type} ({model_id})")

            # Load ControlNet model
            controlnet = ControlNetModel.from_pretrained(
                model_id,
                cache_dir=str(self.controlnet_cache_dir),
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
            )

            # Cache the loaded model
            self.loaded_controlnets[controlnet_type] = controlnet

            logger.info(f"ControlNet {controlnet_type} loaded successfully")
            return controlnet

        except Exception as e:
            logger.error(f"Failed to load ControlNet {controlnet_type}: {e}")
            raise RuntimeError(f"ControlNet loading failed: {e}")

    def get_processor(self, controlnet_type: str):
        """Get or create image processor for ControlNet type"""
        if controlnet_type in self.loaded_processors:
            return self.loaded_processors[controlnet_type]

        processor = None

        try:
            if controlnet_type == "pose":
                processor = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
            elif controlnet_type == "depth":
                processor = MidasDetector.from_pretrained("lllyasviel/Annotators")
            elif controlnet_type == "canny":
                processor = CannyDetector()
            elif controlnet_type == "lineart":
                processor = LineartDetector.from_pretrained("lllyasviel/Annotators")
            else:
                raise ValueError(f"No processor available for: {controlnet_type}")

            self.loaded_processors[controlnet_type] = processor
            logger.info(f"Processor loaded for {controlnet_type}")
            return processor

        except Exception as e:
            logger.error(f"Failed to load processor for {controlnet_type}: {e}")
            raise RuntimeError(f"Processor loading failed: {e}")

    def preprocess_control_image(
        self, image: Image.Image, controlnet_type: str, **kwargs
    ) -> Image.Image:
        """Preprocess control image based on ControlNet type"""
        try:
            processor = self.get_processor(controlnet_type)

            if controlnet_type == "pose":
                return processor(image)  # type: ignore
            elif controlnet_type == "depth":
                return processor(image)  # type: ignore
            elif controlnet_type == "canny":
                # Convert PIL to OpenCV format
                opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                low_threshold = kwargs.get("low_threshold", 100)
                high_threshold = kwargs.get("high_threshold", 200)

                canny = cv2.Canny(opencv_image, low_threshold, high_threshold)
                canny_image = Image.fromarray(canny)
                return canny_image
            elif controlnet_type == "lineart":
                return processor(image)  # type: ignore
            else:
                # For scribble or custom types, return image as-is
                return image

        except Exception as e:
            logger.error(f"Image preprocessing failed for {controlnet_type}: {e}")
            raise RuntimeError(f"Preprocessing failed: {e}")

    def create_controlnet_pipeline(self, base_pipeline, controlnet_type: str):
        """Create ControlNet-enabled pipeline from base pipeline"""
        try:
            controlnet_model = self.load_controlnet(controlnet_type)

            # Create ControlNet pipeline
            controlnet_pipeline = StableDiffusionControlNetPipeline(
                vae=base_pipeline.vae,
                text_encoder=base_pipeline.text_encoder,
                tokenizer=base_pipeline.tokenizer,
                unet=base_pipeline.unet,
                controlnet=controlnet_model,
                scheduler=base_pipeline.scheduler,
                safety_checker=base_pipeline.safety_checker,
                feature_extractor=base_pipeline.feature_extractor,
                requires_safety_checker=False,
            )

            return controlnet_pipeline

        except Exception as e:
            logger.error(f"Failed to create ControlNet pipeline: {e}")
            raise RuntimeError(f"ControlNet pipeline creation failed: {e}")

    def unload_controlnet(self, controlnet_type: str) -> bool:
        """Unload specific ControlNet to free memory"""
        try:
            if controlnet_type in self.loaded_controlnets:
                del self.loaded_controlnets[controlnet_type]

            if controlnet_type in self.loaded_processors:
                del self.loaded_processors[controlnet_type]

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"ControlNet {controlnet_type} unloaded")
            return True

        except Exception as e:
            logger.error(f"Failed to unload ControlNet {controlnet_type}: {e}")
            return False

    def get_controlnet_info(self, controlnet_type: str) -> Optional[Dict]:
        """Get information about specific ControlNet"""
        if controlnet_type not in self.available_controlnets:
            return None

        info = self.available_controlnets[controlnet_type].copy()
        info.update(
            {  # type: ignore
                "type": controlnet_type,
                "loaded": controlnet_type in self.loaded_controlnets,
                "processor_loaded": controlnet_type in self.loaded_processors,
            }
        )

        return info
