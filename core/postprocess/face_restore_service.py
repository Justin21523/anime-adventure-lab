"""Face restore service — GFPGAN / CodeFormer wrapper (adapted for anime-adventure-lab).

Original source: sd-multimodal-platform/services/postprocess/face_restore_service.py
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Optional GFPGAN support
try:
    from gfpgan import GFPGANer

    GFPGAN_AVAILABLE = True
except ImportError:
    GFPGANer = None  # type: ignore
    GFPGAN_AVAILABLE = False

# Optional CodeFormer support
try:
    from basicsr.utils import imwrite  # type: ignore
    from facelib.utils.face_restoration_helper import FaceRestoreHelper  # type: ignore
    from codeformer import CodeFormer as CodeFormerModel  # type: ignore

    CODEFORMER_AVAILABLE = True
except ImportError:
    CodeFormerModel = None  # type: ignore
    FaceRestoreHelper = None  # type: ignore
    CODEFORMER_AVAILABLE = False


class FaceRestoreService:
    """Face restoration service using GFPGAN or CodeFormer."""

    def __init__(self, model_root: Optional[str] = None):
        self.model_root = Path(model_root) if model_root else None
        self._gfpgan_restorer = None
        self._codeformer_model = None
        self._face_helper = None

    def load_gfpgan(self, model_name: str = "GFPGANv1.4", upscale: bool = True) -> bool:
        """Load GFPGAN restorer."""
        if not GFPGAN_AVAILABLE:
            logger.warning("GFPGAN not installed")
            return False

        try:
            self._gfpgan_restorer = GFPGANer(
                model_path=str(self._get_model_path("GFPGANv1.4")),
                upscale=2 if upscale else 1,
                channel_multiplier=2,
                bg_model=None,
            )
            logger.info(f"Loaded GFPGAN: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load GFPGAN: {e}")
            return False

    def load_codeformer(self, weight: float = 0.5) -> bool:
        """Load CodeFormer model."""
        if not CODEFORMER_AVAILABLE:
            logger.warning("CodeFormer not installed")
            return False

        try:
            self._codeformer_model = CodeFormerModel(
                model_path=str(self._get_model_path("CodeFormer")),
                weight=weight,
                cpu=False,
            )
            logger.info("Loaded CodeFormer")
            return True
        except Exception as e:
            logger.error(f"Failed to load CodeFormer: {e}")
            return False

    def _get_model_path(self, model_name: str) -> Path:
        if self.model_root:
            return self.model_root / model_name
        return Path(model_name)

    def restore_faces(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        method: str = "gfpgan",
        **kwargs,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Restore faces in an image.

        Args:
            image: Input image (path, numpy, or PIL).
            method: "gfpgan" or "codeformer".
            **kwargs: Extra args.

        Returns:
            (restored_image_array, metadata_dict)
        """
        start = time.time()
        job_id = str(uuid.uuid4())[:8]

        if method == "gfpgan":
            if self._gfpgan_restorer is None:
                self.load_gfpgan()
            return self._restore_gfpgan(image, job_id, start, **kwargs)
        elif method == "codeformer":
            if self._codeformer_model is None:
                self.load_codeformer()
            return self._restore_codeformer(image, job_id, start, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _restore_gfpgan(
        self, image, job_id: str, start: float, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        img = self._to_bgr(image)
        _, _, restored = self._gfpgan_restorer.enhance(
            img, has_aligned=False, paste_back=True, **kwargs
        )
        elapsed = time.time() - start
        return restored, {
            "job_id": job_id,
            "method": "gfpgan",
            "elapsed_s": round(elapsed, 3),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _restore_codeformer(
        self, image, job_id: str, start: float, weight: float = 0.5, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Placeholder — full implementation depends on CodeFormer API
        img = self._to_bgr(image)
        elapsed = time.time() - start
        logger.warning("CodeFormer restore stub — return original")
        return img, {
            "job_id": job_id,
            "method": "codeformer",
            "elapsed_s": round(elapsed, 3),
            "timestamp": datetime.utcnow().isoformat(),
            "note": "stub — implement when CodeFormer is available",
        }

    @staticmethod
    def _to_bgr(image) -> np.ndarray:
        if isinstance(image, (str, Path)):
            return cv2.imread(str(image), cv2.IMREAD_COLOR)
        elif isinstance(image, Image.Image):
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return np.asarray(image)

    def unload(self):
        if self._gfpgan_restorer:
            del self._gfpgan_restorer
            self._gfpgan_restorer = None
        if self._codeformer_model:
            del self._codeformer_model
            self._codeformer_model = None
        if self._face_helper:
            del self._face_helper
            self._face_helper = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Face restore models unloaded")

    @property
    def is_gfpgan_available(self) -> bool:
        return GFPGAN_AVAILABLE

    @property
    def is_codeformer_available(self) -> bool:
        return CODEFORMER_AVAILABLE
