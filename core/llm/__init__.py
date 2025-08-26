# core/llm/__init.py
"""
Large Language Model integrations
"""
import requests
from typing import List, Dict, Optional


class OllamaLLM:
    """Ollama local LLM client"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = "llama3.2"  # default model

    def generate(
        self, prompt: str, max_tokens: int = 512, temperature: float = 0.7
    ) -> str:
        """Generate text using Ollama"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                },
                timeout=120,
            )

            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                raise Exception(f"Ollama API error: {response.status_code}")

        except Exception as e:
            print(f"Ollama generation failed: {e}")
            return f"Sorry, I encountered an error: {str(e)}"

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Chat completion using Ollama"""
        # Convert messages to prompt
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        prompt = "\n".join(prompt_parts) + "\nAssistant:"
        return self.generate(prompt)
