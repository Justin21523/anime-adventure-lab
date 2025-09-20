# core/llm/__init__.py
"""
LLM Module
Complete language model integration with advanced features
"""
import logging
from typing import Optional
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

logger = logging.getLogger(__name__)

# 模組級別的全域實例
_llm_adapter: Optional["EnhancedTransformersLLM"] = None
_model_loader: Optional["ModelLoader"] = None
_chat_manager: Optional["ChatManager"] = None
_context_manager: Optional["ContextManager"] = None


def get_llm_adapter(
    model_name: str = "microsoft/DialoGPT-medium", **kwargs
) -> "EnhancedTransformersLLM":
    """Get or create LLM adapter instance"""
    global _llm_adapter

    if _llm_adapter is None or _llm_adapter.model_name != model_name:
        from .adapter import EnhancedTransformersLLM
        from .model_loader import ModelLoadConfig

        # Create load config from kwargs
        load_config = ModelLoadConfig(model_name=model_name, **kwargs)

        _llm_adapter = EnhancedTransformersLLM(model_name, load_config)
        logger.info(f"Created new LLM adapter for: {model_name}")

    return _llm_adapter


def get_model_loader() -> "ModelLoader":
    """Get singleton model loader instance"""
    global _model_loader

    if _model_loader is None:
        from .model_loader import ModelLoader

        _model_loader = ModelLoader()
        logger.info("Created ModelLoader instance")

    return _model_loader


def get_chat_manager() -> "ChatManager":
    """Get singleton chat manager instance"""
    global _chat_manager

    if _chat_manager is None:
        from .chat_manager import ChatManager

        _chat_manager = ChatManager()
        logger.info("Created ChatManager instance")

    return _chat_manager


def get_context_manager() -> "ContextManager":
    """Get singleton context manager instance"""
    global _context_manager

    if _context_manager is None:
        from .context_manager import ContextManager

        _context_manager = ContextManager()
        logger.info("Created ContextManager instance")

    return _context_manager


# Export ModelLoadConfig for external use
def create_model_config(**kwargs) -> "ModelLoadConfig":
    """Create a ModelLoadConfig with safe defaults"""
    from .model_loader import ModelLoadConfig

    defaults = {
        "device_map": "auto",
        "torch_dtype": "float16",
        "use_quantization": True,
        "quantization_bits": 4,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    # Override defaults with user kwargs
    config_args = {**defaults, **kwargs}

    # Model name is required
    if "model_name" not in config_args:
        raise ValueError("model_name is required for ModelLoadConfig")

    return ModelLoadConfig(**config_args)


# Expose as module-level alias
ModelLoadConfig = create_model_config


# Clean shutdown
def shutdown_llm_modules():
    """Clean shutdown of all LLM modules"""
    global _llm_adapter, _model_loader, _chat_manager, _context_manager

    logger.info("Shutting down LLM modules...")

    if _llm_adapter and _llm_adapter.is_available():
        _llm_adapter.unload_model()
        _llm_adapter = None

    if _model_loader:
        _model_loader.cleanup()
        _model_loader = None

    if _chat_manager:
        _chat_manager.cleanup()
        _chat_manager = None

    _context_manager = None

    logger.info("LLM modules shutdown complete")


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
