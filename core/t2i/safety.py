# core/t2i/safety.py
import re
from typing import List, Optional
from PIL import Image


class SafetyFilter:
    """Content safety filtering"""

    def __init__(self):
        self.blocked_terms = [
            # Add sensitive terms here
            "nsfw",
            "explicit",
            "nude",
            "sexual",
        ]

    def filter_prompt(self, prompt: str) -> tuple[str, List[str]]:
        """Filter prompt and return cleaned version + violations"""
        violations = []
        cleaned = prompt

        for term in self.blocked_terms:
            if term.lower() in prompt.lower():
                violations.append(term)
                cleaned = re.sub(term, "[FILTERED]", cleaned, flags=re.IGNORECASE)

        return cleaned, violations

    def is_safe_image(self, image: Image.Image) -> bool:
        """Check if generated image is safe (placeholder)"""
        # TODO: Implement actual NSFW detection
        return True
