# core/llm/__init__.py
"""
LLM Module
Complete language model integration with advanced features
"""

from .base import BaseLLM, ChatMessage, LLMResponse
from .model_loader import ModelLoader, ModelLoadConfig, get_model_loader
from .chat_manager import ChatManager, ChatSession, get_chat_manager
from .context_manager import (
    ContextManager,
    ContextWindow,
    TokenUsage,
    get_context_manager,
)
from .adapter import (
    EnhancedTransformersLLM,
    QwenLLM,
    LlamaLLM,
    EnhancedLLMAdapter,
    get_llm_adapter,
    get_enhanced_llm_adapter,
)

__all__ = [
    # Base classes
    "BaseLLM",
    "ChatMessage",
    "LLMResponse",
    # Model loading
    "ModelLoader",
    "ModelLoadConfig",
    "get_model_loader",
    # Chat management
    "ChatManager",
    "ChatSession",
    "get_chat_manager",
    # Context management
    "ContextManager",
    "ContextWindow",
    "TokenUsage",
    "get_context_manager",
    # LLM implementations
    "EnhancedTransformersLLM",
    "QwenLLM",
    "LlamaLLM",
    # Main adapter
    "EnhancedLLMAdapter",
    "get_llm_adapter",
    "get_enhanced_llm_adapter",
]
