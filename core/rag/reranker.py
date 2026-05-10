from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder  # type: ignore
except Exception:  # noqa: BLE001
    CrossEncoder = None  # type: ignore


def _resolve_device(device: str) -> str:
    raw = (device or "cpu").strip().lower()
    if raw == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    if raw.startswith("cuda"):
        return "cuda"
    return "cpu"


@dataclass
class RerankScoredItem:
    index: int
    score: float


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: Path,
        device: str = "cpu",
        max_seq_length: int = 512,
    ) -> None:
        self.model_name = str(model_name or "").strip()
        self.cache_dir = Path(cache_dir)
        self.device = _resolve_device(device)
        self.max_seq_length = int(max_seq_length or 512)

        self._model: Optional[Any] = None
        self._load_failed: bool = False

    def is_enabled(self) -> bool:
        return bool(self.model_name) and (not self._load_failed) and CrossEncoder is not None

    def _load(self) -> Optional[Any]:
        if self._load_failed:
            return None
        if self._model is not None:
            return self._model
        if CrossEncoder is None:
            self._load_failed = True
            return None

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        try:
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=self.max_seq_length,
                cache_folder=str(self.cache_dir),
            )
            return self._model
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reranker load failed (%s): %s", self.model_name, exc)
            self._load_failed = True
            self._model = None
            return None

    def rerank(self, query: str, passages: List[str]) -> Optional[List[float]]:
        if not passages:
            return []

        model = self._load()
        if model is None:
            return None

        try:
            pairs = [[query, passage] for passage in passages]
            if self.device == "cuda":
                try:
                    from core.runtime import get_model_runtime

                    runtime = get_model_runtime()
                    with runtime.exclusive_gpu(reason="rag.rerank", device="cuda"):
                        scores = model.predict(pairs)
                except Exception:
                    scores = model.predict(pairs)
            else:
                scores = model.predict(pairs)
            return [float(x) for x in scores]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reranker predict failed: %s", exc)
            return None
