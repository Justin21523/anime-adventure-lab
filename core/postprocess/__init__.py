"""Postprocess module (absorbed from sd-multimodal-platform).

Provides image post-processing capabilities:
- Real-ESRGAN upscaling
- GFPGAN / CodeFormer face restoration
- Video processing pipeline
"""

from __future__ import annotations

__all__ = [
    "UpscaleService",
    "FaceRestoreService",
    "PipelineManager",
    "VideoService",
]

# Lazy-load heavy dependencies
_UPSCALE_AVAILABLE = False
_FACE_RESTORE_AVAILABLE = False
_VIDEO_AVAILABLE = False

try:
    from .upscale_service import UpscaleService
    _UPSCALE_AVAILABLE = True
except ImportError:
    UpscaleService = None  # type: ignore

try:
    from .face_restore_service import FaceRestoreService
    _FACE_RESTORE_AVAILABLE = True
except ImportError:
    FaceRestoreService = None  # type: ignore

try:
    from .pipeline_manager import PipelineManager
    _VIDEO_AVAILABLE = True
except ImportError:
    PipelineManager = None  # type: ignore

try:
    from .video_service import VideoService
except ImportError:
    VideoService = None  # type: ignore


def get_available_services() -> dict[str, bool]:
    """Return which postprocess services are available."""
    return {
        "upscale": _UPSCALE_AVAILABLE,
        "face_restore": _FACE_RESTORE_AVAILABLE,
        "video": _VIDEO_AVAILABLE,
    }
