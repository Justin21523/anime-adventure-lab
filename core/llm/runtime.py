from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from core.config import get_config

from .base import LLMResponse


class MockRuntimeLLM:
    def chat(self, messages: list[dict[str, str]], **_: Any) -> LLMResponse:
        content = messages[-1].get("content", "") if messages else ""
        return LLMResponse(
            content=f"Mock response to: {content}",
            model_name="mock",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    def list_loaded_models(self) -> list[str]:
        return ["mock"]

    def is_available(self) -> bool:
        return True

    def unload_all(self) -> None:
        return None

    def health_check(self) -> dict[str, Any]:
        return {"available": True, "type": "mock", "model": "mock"}


class RuntimeLLMFacade:
    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        if hasattr(self.backend, "load_model") and not self.backend.is_available():
            self.backend.load_model()
        return self.backend.chat(messages, **kwargs)

    def list_loaded_models(self) -> list[str]:
        if hasattr(self.backend, "is_available") and self.backend.is_available():
            return [str(getattr(self.backend, "model_name", "runtime"))]
        return []

    def is_available(self) -> bool:
        return bool(self.backend.is_available())

    def unload_all(self) -> None:
        if hasattr(self.backend, "unload_model"):
            self.backend.unload_model()

    def health_check(self) -> dict[str, Any]:
        result = {
            "available": bool(self.backend.is_available()),
            "type": type(self.backend).__name__,
            "model": str(getattr(self.backend, "model_name", "runtime")),
        }
        server_url = getattr(self.backend, "server_url", None)
        if server_url:
            result["server_url"] = server_url
        return result


@lru_cache(maxsize=1)
def get_runtime_llm() -> Any:
    """Select mock, remote llama.cpp, or local Transformers at runtime."""
    if os.getenv("LLM_MOCK", "0").strip().lower() in {"1", "true", "yes", "on"}:
        return MockRuntimeLLM()

    backend = os.getenv("LLM_BACKEND", "").strip().lower()
    if backend == "llamacpp" or os.getenv("LLAMA_CPP_SERVER", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        from .llamacpp_server_adapter import LlamaCppServerLLM

        model_name = str(getattr(get_config().model, "chat_model", "runtime-model"))
        return RuntimeLLMFacade(LlamaCppServerLLM(model_name=model_name))

    # The heavy adapter is an explicit local-inference dependency.
    from .adapter import get_llm_adapter

    return get_llm_adapter()
