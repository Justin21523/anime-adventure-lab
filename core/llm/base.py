# core/llm/base.py
"""Base LLM interface"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class MinimalLLM(ABC):
    """Abstract base class for LLM implementations"""

    def __init__(self):
        self.model_name = ""

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Generate chat response from message history"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if model is available"""
        pass
