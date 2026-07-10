"""LLM package with lazy exports.

The API container can use a remote llama.cpp server without importing torch or
transformers. Heavy local-model modules are loaded only when explicitly used.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "BaseLLM": ("core.llm.base", "BaseLLM"),
    "ChatMessage": ("core.llm.base", "ChatMessage"),
    "LLMResponse": ("core.llm.base", "LLMResponse"),
    "LlamaCppServerLLM": ("core.llm.llamacpp_server_adapter", "LlamaCppServerLLM"),
    "get_runtime_llm": ("core.llm.runtime", "get_runtime_llm"),
    "get_llm_adapter": ("core.llm.adapter", "get_llm_adapter"),
    "get_enhanced_llm_adapter": ("core.llm.adapter", "get_enhanced_llm_adapter"),
    "reset_llm_adapter": ("core.llm.adapter", "reset_llm_adapter"),
    "LLMAdapter": ("core.llm.adapter", "LLMAdapter"),
    "EnhancedLLMAdapter": ("core.llm.adapter", "EnhancedLLMAdapter"),
    "EnhancedTransformersLLM": ("core.llm.adapter", "EnhancedTransformersLLM"),
    "QwenLLM": ("core.llm.adapter", "QwenLLM"),
    "LlamaLLM": ("core.llm.adapter", "LlamaLLM"),
    "ModelLoader": ("core.llm.model_loader", "ModelLoader"),
    "ModelLoadConfig": ("core.llm.model_loader", "ModelLoadConfig"),
    "get_model_loader": ("core.llm.model_loader", "get_model_loader"),
    "ChatManager": ("core.llm.chat_manager", "ChatManager"),
    "ChatSession": ("core.llm.chat_manager", "ChatSession"),
    "get_chat_manager": ("core.llm.chat_manager", "get_chat_manager"),
    "ContextManager": ("core.llm.context_manager", "ContextManager"),
    "ContextWindow": ("core.llm.context_manager", "ContextWindow"),
    "TokenUsage": ("core.llm.context_manager", "TokenUsage"),
    "get_context_manager": ("core.llm.context_manager", "get_context_manager"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
