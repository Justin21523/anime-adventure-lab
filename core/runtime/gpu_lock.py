from __future__ import annotations

import asyncio
import logging
import os
import time
import threading
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import fcntl  # type: ignore
except Exception:  # noqa: BLE001
    fcntl = None  # type: ignore


_local = threading.local()


def _default_lock_path() -> Path:
    override = os.getenv("AI_GPU_LOCK_PATH") or os.getenv("GPU_LOCK_PATH")
    if override:
        return Path(str(override)).expanduser()

    try:
        from core.shared_cache import get_shared_cache

        cache = get_shared_cache()
        return Path(cache.cache_root) / "gpu.lock"
    except Exception:
        return Path("/tmp/anime_adventure_lab_gpu.lock")


class GPUFileLock:
    """Cross-process GPU mutex via filesystem flock (best-effort)."""

    def __init__(
        self,
        *,
        path: Optional[Path] = None,
        timeout_s: Optional[float] = None,
        poll_s: float = 0.2,
        reason: str = "gpu",
    ) -> None:
        self.path = Path(path or _default_lock_path())
        self.timeout_s = timeout_s
        self.poll_s = float(poll_s or 0.2)
        self.reason = str(reason or "gpu")
        self._fh = None
        self._acquired = False

    def acquire(self) -> None:
        depth = int(getattr(_local, "depth", 0) or 0)
        if depth > 0:
            _local.depth = depth + 1
            self._acquired = True
            return

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        self._fh = open(self.path, "a+", encoding="utf-8")  # noqa: SIM115

        if fcntl is None:
            logger.warning("fcntl not available; GPU lock is process-local only")
            _local.depth = 1
            self._acquired = True
            return

        start = time.time()
        while True:
            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                _local.depth = 1
                self._acquired = True
                return
            except BlockingIOError:
                if self.timeout_s is not None and (time.time() - start) >= float(
                    self.timeout_s
                ):
                    raise TimeoutError(
                        f"Timed out waiting for GPU lock ({self.reason})"
                    )
                time.sleep(self.poll_s)

    def release(self) -> None:
        if not self._acquired:
            return

        depth = int(getattr(_local, "depth", 0) or 0)
        if depth > 1:
            _local.depth = depth - 1
            return
        _local.depth = 0

        if self._fh and fcntl is not None:
            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass

        try:
            if self._fh:
                self._fh.close()
        finally:
            self._fh = None
            self._acquired = False


@contextmanager
def gpu_lock(
    *,
    path: Optional[Path] = None,
    timeout_s: Optional[float] = None,
    poll_s: float = 0.2,
    reason: str = "gpu",
):
    lock = GPUFileLock(path=path, timeout_s=timeout_s, poll_s=poll_s, reason=reason)
    lock.acquire()
    try:
        yield lock
    finally:
        lock.release()


@asynccontextmanager
async def async_gpu_lock(
    *,
    path: Optional[Path] = None,
    timeout_s: Optional[float] = None,
    poll_s: float = 0.2,
    reason: str = "gpu",
):
    lock = GPUFileLock(path=path, timeout_s=timeout_s, poll_s=poll_s, reason=reason)
    await asyncio.to_thread(lock.acquire)
    try:
        yield lock
    finally:
        await asyncio.to_thread(lock.release)

