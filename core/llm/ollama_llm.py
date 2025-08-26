# core/llm/ollama_llm.py
# core/llm/ollama_llm.py
"""Ollama LLM implementation"""
import requests
import json
from typing import List, Dict, Any
from .base import MinimalLLM


class OllamaLLM(MinimalLLM):
    """Ollama API wrapper"""

    def __init__(
        self, model_name: str = "qwen:7b", base_url: str = "http://localhost:11434"
    ):
        super().__init__()
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Generate chat response via Ollama API"""
        try:
            # Format for Ollama chat API
            payload = {"model": self.model_name, "messages": messages, "stream": False}

            response = requests.post(
                f"{self.base_url}/api/chat", json=payload, timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "")
            else:
                # Fallback to mock response
                return f"Mock response (Ollama unavailable): {messages[-1].get('content', '')[:50]}..."

        except Exception as e:
            # Graceful degradation
            return f"Mock response (Ollama error): {messages[-1].get('content', '')[:50]}..."

    def is_available(self) -> bool:
        """Check if Ollama service is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
