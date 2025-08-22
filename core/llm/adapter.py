# core/llm/adapter.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import json


class LLMAdapter(ABC):
    """Abstract base class for LLM backends"""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Generate response from messages"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if model is available"""
        pass

    def format_messages(
        self, system: str, user: str, history: List[Dict[str, str]] = None  # type: ignore
    ) -> List[Dict[str, str]]:
        """Format messages for chat completion"""
        messages = [{"role": "system", "content": system}]

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": user})
        return messages

    def extract_json_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response, handle markdown code blocks"""
        response = response.strip()

        # Remove markdown code blocks if present
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
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
                print(f"JSON decode error: {e}")
                print(f"Raw response: {response}")
                # Return fallback structure
                return {
                    "narration": response,
                    "dialogues": [],
                    "choices": [
                        {"id": "continue", "text": "繼續", "description": "繼續故事"}
                    ],
                }

        # Fallback if no valid JSON found
        return {
            "narration": response,
            "dialogues": [],
            "choices": [{"id": "continue", "text": "繼續", "description": "繼續故事"}],
        }
