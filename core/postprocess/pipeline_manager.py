"""Pipeline manager — orchestrates post-processing steps (adapted for anime-adventure-lab).

Original source: sd-multimodal-platform/services/postprocess/pipeline_manager.py
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class PostProcessStep:
    """A single step in a post-processing pipeline."""

    def __init__(self, name: str, func: Callable, **defaults):
        self.name = name
        self.func = func
        self.defaults = defaults

    def run(self, image: np.ndarray, **kwargs) -> np.ndarray:
        merged = {**self.defaults, **kwargs}
        return self.func(image, **merged)


class PipelineManager:
    """Manages a sequence of post-processing steps."""

    def __init__(self):
        self._steps: List[PostProcessStep] = []

    def add_step(self, name: str, func: Callable, **defaults) -> "PipelineManager":
        self._steps.append(PostProcessStep(name, func, **defaults))
        return self

    def remove_step(self, name: str) -> bool:
        for i, step in enumerate(self._steps):
            if step.name == name:
                self._steps.pop(i)
                return True
        return False

    def run(self, image: np.ndarray, step_overrides: Optional[Dict[str, Dict]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run the full pipeline.

        Returns (output_image, metadata).
        """
        step_overrides = step_overrides or {}
        results = {}
        current = image
        start = time.time()

        for step in self._steps:
            t0 = time.time()
            overrides = step_overrides.get(step.name, {})
            try:
                current = step.run(current, **overrides)
                results[step.name] = {"status": "ok", "elapsed_s": round(time.time() - t0, 3)}
            except Exception as e:
                logger.warning(f"Pipeline step '{step.name}' failed: {e}")
                results[step.name] = {"status": "error", "error": str(e)}

        return current, {
            "steps": results,
            "total_elapsed_s": round(time.time() - start, 3),
        }

    @property
    def step_names(self) -> List[str]:
        return [s.name for s in self._steps]
