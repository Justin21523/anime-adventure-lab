from __future__ import annotations

import gc
import logging
import os
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from core.runtime.gpu_lock import async_gpu_lock, gpu_lock

logger = logging.getLogger(__name__)


def _is_cuda_device(device: str) -> bool:
    raw = str(device or "").lower()
    return raw.startswith("cuda") or raw == "cuda"


@dataclass
class VRAMStats:
    device: str
    allocated_mb: float = 0.0
    reserved_mb: float = 0.0
    total_mb: float = 0.0
    free_mb: float = 0.0


class VRAMManager:
    """Best-effort VRAM cleanup and stats helpers."""

    def stats(self, device: str = "cuda") -> VRAMStats:
        raw_device = str(device or "cuda")
        stats = VRAMStats(device=raw_device)

        if not torch.cuda.is_available() or not _is_cuda_device(raw_device):
            return stats

        try:
            stats.allocated_mb = float(torch.cuda.memory_allocated() / 1024**2)
            stats.reserved_mb = float(torch.cuda.memory_reserved() / 1024**2)
            props = torch.cuda.get_device_properties(0)
            stats.total_mb = float(props.total_memory / 1024**2)
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                stats.free_mb = float(free_bytes / 1024**2)
                stats.total_mb = float(total_bytes / 1024**2)
            except Exception:
                pass
        except Exception as exc:  # noqa: BLE001
            logger.debug("VRAM stats unavailable: %s", exc)
        return stats

    def cleanup(self) -> None:
        try:
            gc.collect()
        except Exception:
            pass

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass


class ModelRuntime:
    """
    Process-local coordination wrapper that:
    - provides a cross-process GPU lock (file flock)
    - provides VRAM cleanup hooks
    - optionally unloads other model subsystems (best-effort)
    """

    def __init__(self) -> None:
        self.vram = VRAMManager()

    def _lock_timeout_s(self) -> Optional[float]:
        raw = os.getenv("AI_GPU_LOCK_TIMEOUT_S")
        if raw is None or str(raw).strip() == "":
            return None
        try:
            return float(raw)
        except Exception:
            return None

    def unload_best_effort(self) -> Dict[str, Any]:
        """Best-effort unload of known model singletons to free VRAM."""
        summary: Dict[str, Any] = {"unloaded": [], "errors": []}

        try:
            from core.llm.adapter import get_llm_adapter

            llm = get_llm_adapter()
            llm.unload_all()
            summary["unloaded"].append("llm")
        except Exception as exc:  # noqa: BLE001
            summary["errors"].append({"component": "llm", "error": str(exc)})

        try:
            from core.vlm.engine import get_vlm_engine

            vlm = get_vlm_engine()
            vlm.unload_models()
            summary["unloaded"].append("vlm")
        except Exception as exc:  # noqa: BLE001
            summary["errors"].append({"component": "vlm", "error": str(exc)})

        try:
            import core.t2i.engine as t2i_mod

            engine = getattr(t2i_mod, "_t2i_engine", None)
            if engine is not None and getattr(engine, "current_pipeline", None) is not None:
                try:
                    # Async unload isn't accessible here; do minimal cleanup.
                    engine.current_pipeline = None
                    engine.pipeline = None
                    engine.loaded_model = None
                    engine.current_model_id = None
                    summary["unloaded"].append("t2i")
                except Exception as exc:  # noqa: BLE001
                    summary["errors"].append({"component": "t2i", "error": str(exc)})
        except Exception:
            pass

        self.vram.cleanup()
        return summary

    @contextmanager
    def exclusive_gpu(self, *, reason: str = "gpu", device: str = "cuda") -> Any:
        if not torch.cuda.is_available() or not _is_cuda_device(device):
            yield
            return

        with gpu_lock(timeout_s=self._lock_timeout_s(), reason=reason):
            self.vram.cleanup()
            try:
                yield
            finally:
                self.vram.cleanup()

    @asynccontextmanager
    async def exclusive_gpu_async(self, *, reason: str = "gpu", device: str = "cuda") -> Any:
        if not torch.cuda.is_available() or not _is_cuda_device(device):
            yield
            return

        async with async_gpu_lock(timeout_s=self._lock_timeout_s(), reason=reason):
            self.vram.cleanup()
            try:
                yield
            finally:
                self.vram.cleanup()


_model_runtime: Optional[ModelRuntime] = None


def get_model_runtime() -> ModelRuntime:
    global _model_runtime
    if _model_runtime is None:
        _model_runtime = ModelRuntime()
    return _model_runtime

