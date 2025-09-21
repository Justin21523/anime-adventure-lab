# core/llm/base.py
"""
Abstract LLM Interface
Standardized interface for all LLM implementations
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from ..exceptions import ModelError, ValidationError

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Standardized chat message"""

    role: str  # "system", "user", "assistant"
    content: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMResponse:
    """Standardized LLM response"""

    content: str
    model_name: str
    usage: Dict[str, int]  # tokens, etc.
    metadata: Optional[Dict[str, Any]] = None


class BaseLLM(ABC):
    """Abstract base class for all LLM implementations"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._model_dict = {}
        self._tokenizer = None
        self._loaded = False

    @abstractmethod
    def load_model(self) -> None:
        """Load the model and tokenizer"""
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """Unload model to free memory"""
        pass

    @abstractmethod
    def generate(
        self, prompt: str, max_length: int = 512, temperature: float = 0.7, **kwargs
    ) -> str:
        """Generate text from prompt"""
        pass

    @abstractmethod
    def chat(
        self,
        messages: List[Union[ChatMessage, Dict[str, str]]],
        max_length: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        """Chat completion with message history"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if model is loaded and available"""
        pass

    def format_messages(
        self, messages: List[Union[ChatMessage, Dict[str, str]]]
    ) -> List[ChatMessage]:
        """Normalize messages to ChatMessage objects"""
        formatted = []
        for msg in messages:
            if isinstance(msg, dict):
                formatted.append(
                    ChatMessage(
                        role=msg.get("role", "user"),
                        content=msg.get("content", ""),
                        metadata=msg.get("metadata"),  # type: ignore
                    )
                )
            else:
                formatted.append(msg)
        return formatted

    def extract_json_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response, handle markdown code blocks"""
        response = response.strip()

        # Remove markdown code blocks if present
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]

        if response.endswith("```"):
            response = response[:-3]

        # Find JSON content between braces
        start = response.find("{")
        end = response.rfind("}")

        if start != -1 and end != -1 and end > start:
            json_str = response[start : end + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error: {e}, raw: {response[:100]}...")
                return self._fallback_json_response(response)

        return self._fallback_json_response(response)

    def _fallback_json_response(self, response: str) -> Dict[str, Any]:
        """Fallback JSON response when parsing fails"""
        return {
            "narration": response,
            "dialogues": [],
            "choices": [{"id": "continue", "text": "繼續", "description": "繼續故事"}],
            "scene_change": False,
        }

    def validate_input(self, text: str, max_length: int = 4000) -> None:
        """Validate input text"""
        if not text or not text.strip():
            raise ValidationError("text", text, "Empty or whitespace-only input")

        if len(text) > max_length:
            raise ValidationError("text", len(text), f"Exceeds max length {max_length}")

    def __enter__(self):
        """Context manager entry"""
        if not self.is_available():
            self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - optionally unload"""
        # Keep model loaded by default for performance
        pass
