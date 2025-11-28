# core/t2i/prompt_utils.py
"""Lightweight prompt processing helpers for T2I."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PromptProcessor:
    """
    Cleans and lightly enhances prompts before they are sent to the T2I engine.
    The class is intentionally lightweight to avoid recursive or expensive
    dependencies during API startup.
    """

    def __init__(self, safety_engine: Any = None, config: Optional[Dict[str, Any]] = None):  # noqa: ANN401
        self.safety_engine = safety_engine
        self.config = config or {}

        self.quality_boosters = [
            "masterpiece",
            "best quality",
            "high quality",
            "detailed",
        ]
        self.negative_defaults = [
            "worst quality",
            "low quality",
            "blurry",
            "bad anatomy",
            "bad hands",
            "text",
            "watermark",
            "signature",
        ]
        self.max_prompt_length = int(self.config.get("max_prompt_length", 500))
        self.processing_stats = {
            "total_processed": 0,
            "enhanced_prompts": 0,
            "filtered_prompts": 0,
        }

    def initialize(self, components: Optional[Dict[str, Any]] = None) -> None:
        """Inject optional component references after construction."""
        components = components or {}
        self.safety_engine = components.get("safety_engine", self.safety_engine)

    def process(self, prompt: str, is_negative: bool = False, enhance: bool = True) -> str:
        """
        Run basic cleaning, optional enhancement, and safety checks.
        Returns a string that is safe to send to the generation backend.
        """
        if not prompt:
            return ""

        self.processing_stats["total_processed"] += 1

        cleaned = self._clean_prompt(prompt)

        if self.safety_engine:
            safety = self.safety_engine.check_prompt_safety(cleaned)
            if not safety.get("is_safe", True):
                self.processing_stats["filtered_prompts"] += 1
                # Prefer cleaned_prompt if provided; otherwise fall back to a safe stub.
                return safety.get(
                    "clean_prompt", "safe content" if not is_negative else "inappropriate content"
                )
            cleaned = safety.get("clean_prompt", cleaned)

        if enhance:
            if is_negative:
                cleaned = self._add_negative_defaults(cleaned)
            else:
                cleaned = self._add_quality_boosters(cleaned)
                self.processing_stats["enhanced_prompts"] += 1

        return self._validate_length(cleaned, self.max_prompt_length)

    def _clean_prompt(self, prompt: str) -> str:
        """Normalize whitespace and strip redundant punctuation."""
        cleaned = re.sub(r"\s+", " ", prompt.strip())
        cleaned = re.sub(r",+", ",", cleaned)
        cleaned = re.sub(r"^,|,$", "", cleaned)
        cleaned = re.sub(r"\(\s*\)", "", cleaned)
        cleaned = re.sub(r"\[\s*\]", "", cleaned)
        return cleaned

    def _add_quality_boosters(self, prompt: str) -> str:
        """Inject a single quality booster if missing."""
        if not prompt:
            return prompt

        tokens = [token.strip() for token in prompt.split(",") if token.strip()]
        for booster in self.quality_boosters:
            if not any(booster.lower() in token.lower() for token in tokens):
                tokens.insert(1 if tokens else 0, booster)
                break
        return ", ".join(tokens)

    def _add_negative_defaults(self, prompt: str) -> str:
        """Ensure common negative terms are present."""
        if not prompt:
            return ", ".join(self.negative_defaults)

        tokens = [token.strip() for token in prompt.split(",") if token.strip()]
        existing = {token.lower() for token in tokens}
        for negative in self.negative_defaults:
            if negative.lower() not in existing:
                tokens.append(negative)
        return ", ".join(tokens)

    def _validate_length(self, prompt: str, max_length: int) -> str:
        """Trim prompt to avoid overly long requests."""
        if len(prompt) <= max_length:
            return prompt

        truncated = prompt[:max_length]
        last_comma = truncated.rfind(",")
        if last_comma > max_length * 0.8:
            truncated = truncated[:last_comma]
        logger.warning("Prompt truncated from %s to %s characters", len(prompt), len(truncated))
        return truncated

    def parse_weighted_prompt(self, prompt: str) -> List[Tuple[str, float]]:
        """Placeholder for future weighted prompt parsing."""
        return [(prompt, 1.0)]

    def suggest_improvements(self, prompt: str) -> List[str]:
        """Return lightweight prompt improvement suggestions."""
        suggestions: List[str] = []
        if len(prompt) < 10:
            suggestions.append("Add more descriptive detail for the scene or subject.")
        if not any(quality in prompt.lower() for quality in self.quality_boosters):
            suggestions.append("Consider a quality booster such as 'high quality' or 'detailed'.")
        if prompt.count(",") < 2:
            suggestions.append("Use comma-separated tags to describe style, lighting, and mood.")
        return suggestions

    def get_processing_stats(self) -> Dict[str, Any]:
        """Return simple counters for observability."""
        total = max(1, self.processing_stats["total_processed"])
        return {
            **self.processing_stats,
            "enhancement_rate": self.processing_stats["enhanced_prompts"] / total,
            "filter_rate": self.processing_stats["filtered_prompts"] / total,
        }

    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.processing_stats = {
            "total_processed": 0,
            "enhanced_prompts": 0,
            "filtered_prompts": 0,
        }

