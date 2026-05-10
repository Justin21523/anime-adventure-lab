"""Upscale service — Real-ESRGAN wrapper (adapted for anime-adventure-lab).

Original source: sd-multimodal-platform/services/postprocess/upscale_service.py
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Optional Real-ESRGAN support
try:
    from realesrgan import RealESRGANer
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact
    from basicsr.archs.rrdbnet_arch import RRDBNet

    REALESRGAN_AVAILABLE = True
except ImportError:
    RealESRGANer = None  # type: ignore
    SRVGGNetCompact = None  # type: ignore
    RRDBNet = None  # type: ignore
    REALESRGAN_AVAILABLE = False

# Known upscale models
KNOWN_UPSCALE_MODELS: Dict[str, Dict[str, str]] = {
    "realesrgan-x4plus": {
        "scale": "4",
        "model_name": "realesrgan-x4plus",
    },
    "realesrgan-x4plus-anime": {
        "scale": "4",
        "model_name": "realesrgan-x4plus-anime",
    },
    "realesrgan-x2plus": {
        "scale": "2",
        "model_name": "realesrgan-x2plus",
    },
}


class UpscaleService:
    """Image upscaling service with Real-ESRGAN support."""

    def __init__(self, model_root: Optional[str] = None):
        self.model_root = Path(model_root) if model_root else None
        self._enhancer = None

    def _get_model_path(self, model_key: str) -> Path:
        """Resolve model path from known models or explicit path."""
        if self.model_root:
            return self.model_root / model_key
        return Path(model_key)

    def load_model(self, model_key: str, scale: int = 4) -> bool:
        """Load an upscale model. Returns True if successful."""
        if not REALESRGAN_AVAILABLE:
            logger.warning("Real-ESRGAN not installed; upscale unavailable")
            return False

        try:
            from basicsr.utils import load_file_from_url  # type: ignore

            info = KNOWN_UPSCALE_MODELS.get(model_key, {})
            model_name = info.get("model_name", model_key)
            scale = int(info.get("scale", scale))

            # Try to load the model — Real-ESRGAN handles auto-download
            self._enhancer = RealESRGANer(
                scale=scale,
                model_path=str(self._get_model_path(model_name)),
                model=RRDBNet if "rrdb" in model_name.lower() else SRVGGNetCompact,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=not torch.cuda.get_device_properties(0).major < 8,
            )
            logger.info(f"Loaded upscale model: {model_name} (scale={scale})")
            return True
        except Exception as e:
            logger.error(f"Failed to load upscale model {model_key}: {e}")
            return False

    def upscale(self, image: Union[str, Path, np.ndarray, Image.Image], model_key: str = "realesrgan-x4plus", **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Upscale an image.

        Args:
            image: Path to image, numpy array, or PIL Image.
            model_key: Key in KNOWN_UPSCALE_MODELS or explicit path.
            **kwargs: Extra args passed to enhancer.enhance().

        Returns:
            (output_image_array, metadata_dict)
        """
        start = time.time()
        job_id = str(uuid.uuid4())[:8]

        if self._enhancer is None:
            if not self.load_model(model_key):
                raise RuntimeError(f"Upscale model '{model_key}' unavailable (Real-ESRGAN not installed)")

        # Convert input to numpy
        if isinstance(image, (str, Path)):
            img = np.array(Image.open(image).convert("RGB"))
        elif isinstance(image, Image.Image):
            img = np.array(image.convert("RGB"))
        else:
            img = np.asarray(image)

        try:
            output, _ = self._enhancer.enhance(img, **kwargs)
            elapsed = time.time() - start
            return output, {
                "job_id": job_id,
                "model": model_key,
                "input_size": list(img.shape[:2]),
                "output_size": list(output.shape[:2]),
                "elapsed_s": round(elapsed, 3),
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"Upscale failed [{job_id}]: {e}")
            raise

    def unload(self):
        """Release model memory."""
        if self._enhancer:
            del self._enhancer
            self._enhancer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Upscale model unloaded")

    @property
    def is_available(self) -> bool:
        return REALESRGAN_AVAILABLE
