# core/t2i/controlnet.py
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline as hf_pipeline
from ..shared_cache import get_shared_cache


class ControlNetManager:
    """Manage ControlNet preprocessing and loading"""

    def __init__(self):
        self.cache = get_shared_cache()
        self._processors = {}

    def preprocess_image(self, image: Image.Image, control_type: str) -> Image.Image:
        """Preprocess image for ControlNet"""
        image_array = np.array(image)

        if control_type == "canny":
            # Canny edge detection
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            return Image.fromarray(edges)

        elif control_type == "depth":
            # Use depth estimation model (simplified)
            try:
                if "depth" not in self._processors:
                    self._processors["depth"] = hf_pipeline(
                        "depth-estimation", model="Intel/dpt-large"
                    )
                result = self._processors["depth"](image)
                depth_image = result["depth"]
                return depth_image
            except:
                # Fallback to grayscale
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
                return Image.fromarray(gray)

        elif control_type == "pose":
            # OpenPose (simplified - return original for now)
            return image

        else:
            return image
