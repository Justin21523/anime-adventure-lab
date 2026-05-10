# core/llm/adapter.py (Enhanced Version)
"""
Enhanced LLM Implementation Adapters
Integrates with model loader, chat manager, and context manager
"""

import torch
import logging
import os
from typing import List, Dict, Any, Optional, Union
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from .base import BaseLLM, ChatMessage, LLMResponse
from .model_loader import get_model_loader, ModelLoadConfig
from .chat_manager import get_chat_manager
from .context_manager import get_context_manager
from ..exceptions import (
    ModelLoadError,
    ModelNotFoundError,
    CUDAOutOfMemoryError,
    ContextLengthExceededError,
    handle_cuda_oom,
    handle_model_error,
)
from ..config import get_config
from ..shared_cache import get_shared_cache

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class EnhancedTransformersLLM(BaseLLM):
    """Enhanced Transformers-based LLM with advanced features"""

    def __init__(self, model_name: str, load_config: Optional[ModelLoadConfig] = None):
        super().__init__(model_name)
        self.load_config = load_config
        self.config = get_config()
        self.cache = get_shared_cache()
        self.model_loader = get_model_loader()
        self.context_manager = get_context_manager()

        # Lazy loading to avoid circular imports
        self._model_loader = None
        # Model components
        self._model_dict: Optional[
            Dict[str, Union[PreTrainedModel, PreTrainedTokenizer]]
        ] = None

        logger.info(f"Initialized EnhancedTransformersLLM: {model_name}")

    @property
    def model(self) -> PreTrainedModel:
        """Get the loaded model"""
        if self._model_dict is None:
            raise ModelNotFoundError(self.model_name)
        return self._model_dict["model"]  # type: ignore

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Get the loaded tokenizer"""
        if self._model_dict is None:
            raise ModelNotFoundError(self.model_name)
        return self._model_dict["tokenizer"]  # type: ignore

    @handle_cuda_oom
    @handle_model_error
    def load_model(self) -> None:
        """Load model using the enhanced model loader"""
        if self._loaded:
            return

        try:
            logger.info(f"Loading enhanced LLM model: {self.model_name}")

            # Use model loader for advanced loading
            self._model_dict = self.model_loader.load_model(
                self.model_name, self.load_config
            )

            self._loaded = True
            logger.info(f"Enhanced LLM model loaded successfully: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to load enhanced LLM {self.model_name}: {e}")
            raise ModelLoadError(self.model_name, str(e))

    def unload_model(self) -> None:
        """Unload model using the model loader"""
        if self._model_dict is not None:
            # Use model loader for proper cleanup
            config_hash = self.load_config.get_cache_key() if self.load_config else None
            self.model_loader.unload_model(self.model_name, config_hash)

            self._model_dict = None
            self._loaded = False

            logger.info(f"Enhanced LLM model unloaded: {self.model_name}")

    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs,
    ) -> str:
        """Generate text with context management"""
        if not self.is_available():
            raise ModelNotFoundError(self.model_name)

        self.validate_input(prompt)

        try:
            # Use context manager for token counting and validation
            messages = [ChatMessage(role="user", content=prompt)]
            prepared_messages, token_usage = self.context_manager.prepare_context(
                messages, self.model_name, max_length
            )

            # Build final prompt
            final_prompt = prepared_messages[0].content

            # Tokenize input (CPU)
            tokenized = self.tokenizer(
                final_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=token_usage.prompt_tokens + 100,  # Safety margin
            )

            input_length = tokenized["input_ids"].shape[1]

            def _do_generate():
                # Move to model device
                device = next(self.model.parameters()).device
                dev_inputs = {k: v.to(device) for k, v in tokenized.items()}

                # Generate with improved parameters
                with torch.no_grad():
                    return self.model.generate(  # type: ignore
                        **dev_inputs,
                        max_new_tokens=max_length,
                        temperature=temperature,
                        do_sample=do_sample,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=kwargs.get("repetition_penalty", 1.1),
                        top_p=kwargs.get("top_p", 0.9),
                        top_k=kwargs.get("top_k", 50),
                        **kwargs,
                    )

            use_gpu_lock = False
            try:
                device_map = str(getattr(self.load_config, "device_map", "") or "").strip().lower()  # type: ignore[union-attr]
                use_gpu_lock = torch.cuda.is_available() and device_map != "cpu"
            except Exception:
                use_gpu_lock = torch.cuda.is_available()

            if use_gpu_lock:
                try:
                    from core.runtime import get_model_runtime

                    runtime = get_model_runtime()
                    with runtime.exclusive_gpu(reason=f"llm.generate:{self.model_name}", device="cuda"):
                        outputs = _do_generate()
                except Exception:
                    outputs = _do_generate()
            else:
                outputs = _do_generate()

            # Decode only the new tokens
            generated_tokens = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            ).strip()

            return generated_text

        except Exception as e:
            logger.error(f"Generation failed for {self.model_name}: {e}")
            raise

    def chat(
        self,
        messages: List[Union[ChatMessage, Dict[str, str]]],
        max_length: int = 512,
        temperature: float = 0.7,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Enhanced chat completion with session management"""
        if not self.is_available():
            raise ModelNotFoundError(self.model_name)

        try:
            # Normalize messages
            normalized_messages = self.format_messages(messages)

            # Use context manager to prepare context
            prepared_messages, token_usage = self.context_manager.prepare_context(
                normalized_messages, self.model_name, max_length
            )

            # Build chat prompt using model-specific formatting
            chat_prompt = self._build_chat_prompt(prepared_messages)

            # Tokenize (CPU)
            tokenized = self.tokenizer(
                chat_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=token_usage.prompt_tokens + 100,
            )

            input_length = tokenized["input_ids"].shape[1]

            def _do_chat_generate():
                device = next(self.model.parameters()).device
                dev_inputs = {k: v.to(device) for k, v in tokenized.items()}
                with torch.no_grad():
                    return self.model.generate(  # type: ignore
                        **dev_inputs,
                        max_new_tokens=max_length,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=kwargs.get("repetition_penalty", 1.1),
                        top_p=kwargs.get("top_p", 0.9),
                        top_k=kwargs.get("top_k", 50),
                        **kwargs,
                    )

            use_gpu_lock = False
            try:
                device_map = str(getattr(self.load_config, "device_map", "") or "").strip().lower()  # type: ignore[union-attr]
                use_gpu_lock = torch.cuda.is_available() and device_map != "cpu"
            except Exception:
                use_gpu_lock = torch.cuda.is_available()

            if use_gpu_lock:
                try:
                    from core.runtime import get_model_runtime

                    runtime = get_model_runtime()
                    with runtime.exclusive_gpu(reason=f"llm.chat:{self.model_name}", device="cuda"):
                        outputs = _do_chat_generate()
                except Exception:
                    outputs = _do_chat_generate()
            else:
                outputs = _do_chat_generate()

            # Decode response
            response_tokens = outputs[0][input_length:]
            response_text = self.tokenizer.decode(
                response_tokens, skip_special_tokens=True
            ).strip()

            # Create response object
            actual_response_tokens = len(response_tokens)
            total_tokens = token_usage.prompt_tokens + actual_response_tokens

            llm_response = LLMResponse(
                content=response_text,
                model_name=self.model_name,
                usage={
                    "prompt_tokens": token_usage.prompt_tokens,
                    "completion_tokens": actual_response_tokens,
                    "total_tokens": total_tokens,
                },
                metadata={
                    "truncated": len(prepared_messages) < len(normalized_messages),
                    "temperature": temperature,
                    "max_length": max_length,
                    "session_id": session_id,
                },
            )

            # Add to session if session_id provided
            if session_id:
                chat_manager = get_chat_manager()
                try:
                    # Add user message
                    if normalized_messages and normalized_messages[-1].role == "user":
                        chat_manager.add_message(
                            session_id, "user", normalized_messages[-1].content
                        )

                    # Add assistant response
                    chat_manager.add_response(session_id, llm_response)
                except Exception as e:
                    logger.warning(f"Failed to update chat session {session_id}: {e}")

            return llm_response

        except Exception as e:
            logger.error(f"Chat completion failed for {self.model_name}: {e}")
            raise

    def _build_chat_prompt(self, messages: List[ChatMessage]) -> str:
        """Build chat prompt - override in subclasses for model-specific formatting"""
        # Generic implementation - concatenate with basic formatting
        prompt_parts = []
        for message in messages:
            if message.role == "system":
                prompt_parts.append(f"System: {message.content}")
            elif message.role == "user":
                prompt_parts.append(f"User: {message.content}")
            elif message.role == "assistant":
                prompt_parts.append(f"Assistant: {message.content}")

        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)

    def is_available(self) -> bool:
        """Check if model is loaded and available"""
        return self._loaded and self._model_dict is not None

    def validate_input(self, prompt: str) -> None:
        """Validate input prompt"""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if len(prompt) > 50000:  # Reasonable character limit
            raise ValueError("Prompt is too long")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        info = {
            "model_name": self.model_name,
            "loaded": self.is_available(),
        }

        if self.is_available():
            info.update(
                {
                    "device": str(next(self.model.parameters()).device),
                    "dtype": str(next(self.model.parameters()).dtype),
                    "num_parameters": sum(p.numel() for p in self.model.parameters()),
                }
            )

        return info


# Mock implementation for testing/development
class MockLLMAdapter(BaseLLM):
    """Mock LLM adapter for testing"""

    def __init__(self, model_name: str = "mock"):
        super().__init__(model_name)
        self._loaded = True  # Mock is always "loaded"

    def load_model(self) -> None:
        """Mock load - always succeeds"""
        self._loaded = True
        logger.info(f"Mock model loaded: {self.model_name}")

    def unload_model(self) -> None:
        """Mock unload"""
        self._loaded = False
        logger.info(f"Mock model unloaded: {self.model_name}")

    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs,
    ) -> str:
        """Generate text with enhanced features"""

        # 特殊處理 mock 模式
        if self.model_name == "mock":
            return f"Mock response to: {prompt[:50]}..."

        if not self.is_available():
            self.load_model()  # Auto-load if needed

        return f"Mock response to: {prompt[:50]}..."

    def chat(
        self,
        messages: List[Union[ChatMessage, Dict[str, str]]],
        max_length: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        """Mock chat"""
        if self.model_name == "mock":
            last_message = (
                messages[-1] if messages else ChatMessage(role="user", content="hello")
            )
            content = (
                last_message.content  # type: ignore
                if hasattr(last_message, "content")
                else str(last_message.get("content", ""))  # type: ignore
            )

            return LLMResponse(
                content=f"Mock response to: {content}",
                model_name=self.model_name,
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 15,
                    "total_tokens": 25,
                },
            )

        if not self.is_available():
            self.load_model()

        last_message = (
            messages[-1] if messages else ChatMessage(role="user", content="hello")
        )
        content = (
            last_message.content  # type: ignore
            if hasattr(last_message, "content")
            else str(last_message.get("content", ""))  # type: ignore
        )

        return LLMResponse(
            content=f"Mock response to: {content}",
            model_name=self.model_name,
            usage={"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
        )

    def is_available(self) -> bool:
        """Check if model is loaded and available"""
        # 特殊處理 mock 模式
        if self.model_name == "mock":
            return True
        return self._loaded and self._model_dict is not None

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        info = {
            "model_name": self.model_name,
            "loaded": self.is_available(),
        }
        return info


class QwenLLM(EnhancedTransformersLLM):
    """Qwen model with specific chat formatting"""

    def _build_chat_prompt(self, messages: List[ChatMessage]) -> str:
        """Qwen-specific chat format"""
        conversation = []

        for msg in messages:
            if msg.role == "system":
                conversation.append(f"<|im_start|>system\n{msg.content}<|im_end|>")
            elif msg.role == "user":
                conversation.append(f"<|im_start|>user\n{msg.content}<|im_end|>")
            elif msg.role == "assistant":
                conversation.append(f"<|im_start|>assistant\n{msg.content}<|im_end|>")

        conversation.append("<|im_start|>assistant\n")
        return "\n".join(conversation)


class LlamaLLM(EnhancedTransformersLLM):
    """Llama model with specific chat formatting"""

    def _build_chat_prompt(self, messages: List[ChatMessage]) -> str:
        """Llama-specific chat format"""
        conversation = []
        system_message = None

        # Extract system message
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            elif msg.role in ["user", "assistant"]:
                conversation.append(msg)

        # Build Llama format
        prompt_parts = []
        if system_message:
            prompt_parts.append(f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n")

        for i, msg in enumerate(conversation):
            if msg.role == "user":
                if i == 0 and system_message:
                    prompt_parts.append(f"{msg.content} [/INST]")
                else:
                    prompt_parts.append(f"<s>[INST] {msg.content} [/INST]")
            elif msg.role == "assistant":
                prompt_parts.append(f" {msg.content} </s>")

        # If last message was user, add opening for assistant
        if conversation and conversation[-1].role == "user":
            prompt_parts.append(" ")

        return "".join(prompt_parts)


class EnhancedLLMAdapter:
    """Enhanced LLM Adapter with full feature integration"""

    def __init__(self):
        self.config = get_config()
        self.cache = get_shared_cache()
        self.model_loader = get_model_loader()
        self.chat_manager = get_chat_manager()
        self.context_manager = get_context_manager()
        self.model_name: str = ""
        self._models: Dict[str, EnhancedTransformersLLM] = {}

    def get_llm(
        self,
        model_name: Optional[str] = None,
        model_type: str = "auto",
        load_config: Optional[ModelLoadConfig] = None,
    ) -> EnhancedTransformersLLM:
        """Get or create enhanced LLM instance"""
        if model_name is None:
            model_name = self.config.get("model.chat_model", "Qwen/Qwen-7B-Chat")

        self.model_name = model_name  # type: ignore

        # Create cache key
        config_key = load_config.get_cache_key() if load_config else "default"
        cache_key = f"{model_name}_{config_key}"

        if cache_key in self._models:
            return self._models[cache_key]

        # Create load config if not provided
        if load_config is None:
            load_config = ModelLoadConfig(
                model_name=model_name,  # type: ignore
                device_map=self.config.model.device_map,
                torch_dtype=self.config.model.torch_dtype,
                use_quantization=self.config.model.use_4bit_loading,
            )

        # Determine model type and create appropriate LLM instance
        if model_type == "auto":
            if "qwen" in model_name.lower():  # type: ignore
                model_type = "qwen"
            elif "llama" in model_name.lower():  # type: ignore
                model_type = "llama"
            else:
                model_type = "generic"

        # Create appropriate LLM instance
        if model_type == "qwen":
            llm = QwenLLM(model_name, load_config)  # type: ignore
        elif model_type == "llama":
            llm = LlamaLLM(model_name, load_config)  # type: ignore
        else:
            llm = EnhancedTransformersLLM(model_name, load_config)  # type: ignore

        self._models[cache_key] = llm
        return llm

    def chat(
        self,
        messages: List[Union[ChatMessage, Dict[str, str]]],
        model_name: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Convenient chat interface with session management"""
        llm = self.get_llm(model_name)
        if not llm.is_available():
            llm.load_model()
        return llm.chat(messages, session_id=session_id, **kwargs)

    def generate(self, prompt: str, model_name: Optional[str] = None, **kwargs) -> str:
        """Convenient generation interface"""
        llm = self.get_llm(model_name)
        if not llm.is_available():
            llm.load_model()
        return llm.generate(prompt, **kwargs)

    def create_chat_session(
        self,
        system_prompt: Optional[str] = None,
        max_history: int = 50,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create new chat session"""
        return self.chat_manager.create_session(
            system_prompt=system_prompt,
            max_history=max_history,
            metadata=metadata,
        )

    def get_chat_session(self, session_id: str) -> Dict[str, Any]:
        """Get chat session information"""
        session = self.chat_manager.get_session(session_id)
        return {
            "session_id": session.session_id,
            "created_at": session.created_at.isoformat(),
            "last_updated": session.last_updated.isoformat(),
            "message_count": session.get_message_count(),
            "system_prompt": session.system_prompt,
            "metadata": session.metadata,
        }

    def list_chat_sessions(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """List chat sessions"""
        return self.chat_manager.list_sessions(limit=limit)

    def delete_chat_session(self, session_id: str) -> bool:
        """Delete chat session"""
        return self.chat_manager.delete_session(session_id)

    def chat_with_session(
        self,
        session_id: str,
        user_message: str,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Chat within an existing session"""
        # Get session context
        session = self.chat_manager.get_session(session_id)

        # Add user message to session
        self.chat_manager.add_message(session_id, "user", user_message)

        # Get conversation context
        messages = self.chat_manager.get_conversation_context(session_id)

        # Generate response
        response = self.chat(
            messages=messages,  # type: ignore
            model_name=model_name,
            session_id=session_id,
            **kwargs,
        )

        return response

    def unload_model(self, model_name: str, config_hash: Optional[str] = None) -> bool:
        """Unload specific model"""
        # Find matching models
        models_to_remove = []

        if config_hash:
            cache_key = f"{model_name}_{config_hash}"
            if cache_key in self._models:
                models_to_remove.append(cache_key)
        else:
            # Remove all instances of this model
            for key in self._models.keys():
                if key.startswith(f"{model_name}_"):
                    models_to_remove.append(key)

        # Unload models
        unloaded = False
        for key in models_to_remove:
            if key in self._models:
                self._models[key].unload_model()
                del self._models[key]
                unloaded = True

        return unloaded

    def unload_all(self) -> int:
        """Unload all models"""
        count = len(self._models)

        # Unload each model properly
        for model in self._models.values():
            try:
                model.unload_model()
            except Exception as e:
                logger.warning(f"Error unloading model: {e}")

        self._models.clear()

        # Use model loader to clean up
        self.model_loader.unload_all()

        logger.info(f"Unloaded {count} LLM models")
        return count

    def list_loaded_models(self) -> List[str]:
        """List currently loaded models"""
        loaded_models = []
        for key, model in self._models.items():
            if model.is_available():
                loaded_models.append(model.model_name)
        return list(set(loaded_models))  # Remove duplicates

    def get_detailed_model_info(self) -> List[Dict[str, Any]]:
        """Get detailed information about loaded models"""
        return self.model_loader.list_loaded_models()

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        return self.model_loader.get_memory_usage()

    def get_context_stats(self) -> Dict[str, Any]:
        """Get context management statistics"""
        return self.context_manager.get_context_stats()  # type: ignore

    def get_chat_stats(self) -> Dict[str, Any]:
        """Get chat session statistics"""
        return self.chat_manager.get_session_stats()

    def cleanup_old_sessions(self, max_age_hours: Optional[int] = None) -> int:
        """Clean up old chat sessions"""
        from datetime import timedelta

        max_age = timedelta(hours=max_age_hours) if max_age_hours else None
        return self.chat_manager.cleanup_old_sessions(max_age)

    def validate_model_context(
        self,
        messages: List[Union[ChatMessage, Dict[str, str]]],
        model_name: Optional[str] = None,
        max_response_length: int = 512,
    ) -> Dict[str, Any]:
        """Validate if messages fit in model context"""
        if model_name is None:
            model_name = self.config.get("model.chat_model", "Qwen/Qwen-7B-Chat")

        normalized_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                normalized_messages.append(ChatMessage(**msg))  # type: ignore
            else:
                normalized_messages.append(msg)

        # Get context validation
        fits = self.context_manager.validate_context_length(
            normalized_messages, model_name, max_response_length  # type: ignore
        )

        # Get token counts
        token_count = self.context_manager.count_messages_tokens(
            normalized_messages, model_name  # type: ignore
        )

        context_window = self.context_manager.get_context_window(model_name)  # type: ignore

        return {
            "fits_in_context": fits,
            "current_tokens": token_count,
            "max_context_tokens": context_window.max_context_length,
            "utilization": token_count / context_window.max_context_length,
            "model_name": model_name,
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "loaded_models": self.list_loaded_models(),
            "detailed_models": self.get_detailed_model_info(),
            "memory_usage": self.get_memory_usage(),
            "context_stats": self.get_context_stats(),
            "chat_stats": self.get_chat_stats(),
            "active_sessions": len(self.chat_manager._sessions),
        }


# Global enhanced adapter instance
_enhanced_llm_adapter: Optional["LLMAdapter"] = None


class LLMAdapter(EnhancedLLMAdapter):
    """Backward-compatible adapter wrapper used by API/workers."""

    def __init__(self, model_name: str, use_mock=None, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.use_mock = use_mock
        self._kwargs = kwargs
        self._llm = self._create_llm()

    def reconfigure(self, model_name: str, use_mock=None, **kwargs) -> None:
        """
        Reconfigure the adapter in-place.

        Why in-place:
        - Many subsystems keep a reference to the global adapter object.
        - Replacing the global singleton would leave stale references behind.
        """
        next_model = str(model_name or "").strip()
        if not next_model:
            return

        next_kwargs = dict(kwargs or {})
        if (
            str(getattr(self, "model_name", "") or "") == next_model
            and getattr(self, "use_mock", None) == use_mock
            and isinstance(getattr(self, "_kwargs", None), dict)
            and dict(getattr(self, "_kwargs", {}) or {}) == next_kwargs
        ):
            return

        try:
            self.unload_all()
        except Exception:
            pass

        self.model_name = next_model
        self.use_mock = use_mock
        self._kwargs = next_kwargs
        self._llm = self._create_llm()

    def _create_llm(self) -> Union[EnhancedTransformersLLM, MockLLMAdapter]:
        """Instantiate underlying LLM.

        Priority order:
        1. Mock (opt-in via `use_mock`, `mock=True`, or `LLM_MOCK=1`)
        2. llama.cpp server (env `LLAMA_SERVER_URL` or `LLM_BACKEND=llamacpp`)
        3. Local HuggingFace Transformers (Qwen / Llama / generic)

        Lack of CUDA should not silently force story generation into mock mode
        when the operator requested real local inference.
        """
        use_mock = (
            bool(self.use_mock)
            if self.use_mock is not None
            else bool(self._kwargs.get("mock", False) or _env_flag("LLM_MOCK", False))
        )

        if use_mock or self.model_name == "mock":
            return MockLLMAdapter("mock")

        # --- llama.cpp server (OpenAI-compatible) ---
        backend = os.getenv("LLM_BACKEND", "").strip().lower()
        server_url = os.getenv("LLAMA_SERVER_URL", "http://localhost:8080").strip()
        if backend == "llamacpp" or server_url != "http://localhost:8080" or _env_flag("LLAMA_CPP_SERVER", False):
            try:
                from .llamacpp_server_adapter import LlamaCppServerLLM

                return LlamaCppServerLLM(
                    model_name=self.model_name,
                    server_url=server_url,
                    timeout=int(os.getenv("LLAMA_SERVER_TIMEOUT", "300")),
                    **self._kwargs,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to create LlamaCppServerLLM (%s); falling back to Transformers", exc
                )

        # --- Local HuggingFace Transformers (original path) ---
        from .model_loader import ModelLoadConfig

        load_config = ModelLoadConfig(model_name=self.model_name, **self._kwargs)  # type: ignore
        lowered = str(self.model_name or "").lower()
        if "qwen" in lowered:
            return QwenLLM(self.model_name, load_config)  # type: ignore[arg-type]
        if "llama" in lowered:
            return LlamaLLM(self.model_name, load_config)  # type: ignore[arg-type]
        return EnhancedTransformersLLM(self.model_name, load_config)

    def chat_completion(
        self,
        messages: List[Union[ChatMessage, Dict[str, str]]],
        max_length: int = 512,
        temperature: float = 0.7,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Return dict-style response for legacy callers."""
        response = self.chat(
            messages=messages,
            max_length=max_length,
            temperature=temperature,
            model_name=model_name,
            **kwargs,
        )
        return {
            "message": response.content,
            "model_used": response.model_name,
            "usage": response.usage,
        }

    def chat(
        self,
        messages: List[Union[ChatMessage, Dict[str, str]]],
        max_length: int = 512,
        temperature: float = 0.7,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Proxy to underlying LLM chat interface."""
        if model_name and model_name != self.model_name:
            self.model_name = model_name
            self._llm = self._create_llm()

        if hasattr(self._llm, "load_model"):
            try:
                self._llm.load_model()  # type: ignore
            except Exception as exc:
                if self._mock_fallback_allowed():
                    logger.warning("LLM load failed; using explicit mock fallback: %s", exc)
                    self._llm = MockLLMAdapter("mock")
                else:
                    raise

        try:
            return self._llm.chat(messages=messages, max_length=max_length, temperature=temperature, **kwargs)  # type: ignore
        except Exception as exc:
            logger.warning("LLM chat failed: %s", exc)
            if self._mock_fallback_allowed():
                logger.warning("Switching to mock fallback after chat failure")
                self._llm = MockLLMAdapter("mock")
                return self._llm.chat(messages=messages, max_length=max_length, temperature=temperature, **kwargs)  # type: ignore
            raise

    def generate_text(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7, **kwargs) -> str:
        """Simple text generation helper."""
        if hasattr(self._llm, "load_model"):
            try:
                self._llm.load_model()  # type: ignore
            except Exception as exc:
                if self._mock_fallback_allowed():
                    logger.warning("LLM load failed; using explicit mock fallback: %s", exc)
                    self._llm = MockLLMAdapter("mock")
                else:
                    raise
        try:
            return self._llm.generate(prompt, max_length=max_tokens, temperature=temperature, **kwargs)  # type: ignore
        except Exception as exc:
            logger.warning("LLM generate failed: %s", exc)
            if self._mock_fallback_allowed():
                logger.warning("Switching to mock fallback after generate failure")
                self._llm = MockLLMAdapter("mock")
                return self._llm.generate(prompt, max_length=max_tokens, temperature=temperature, **kwargs)  # type: ignore
            raise

    def _mock_fallback_allowed(self) -> bool:
        return bool(
            self.use_mock
            or self._kwargs.get("mock", False)
            or _env_flag("LLM_MOCK", False)
            or self.model_name == "mock"
        )

    def is_available(self) -> bool:
        """Check if underlying LLM is available"""
        if hasattr(self._llm, "is_available"):
            return self._llm.is_available()  # type: ignore
        return False

    def list_loaded_models(self) -> List[str]:
        if hasattr(self._llm, "is_available") and self._llm.is_available():  # type: ignore
            return [getattr(self._llm, "model_name", self.model_name)]
        return []

    def unload_all(self) -> bool:
        if hasattr(self._llm, "unload_model"):
            try:
                self._llm.unload_model()  # type: ignore
            except Exception:
                return False
        return True

    def health_check(self) -> Dict[str, Any]:
        """Return LLM health status for /healthz monitoring."""
        info: Dict[str, Any] = {
            "model": getattr(self._llm, "model_name", self.model_name),
            "available": self.is_available(),
            "type": type(self._llm).__name__,
        }
        # For llama.cpp server, do a lightweight connectivity probe
        try:
            from .llamacpp_server_adapter import LlamaCppServerLLM as _LlamaCppServerLLM
        except ImportError:
            _LlamaCppServerLLM = type(None)  # type: ignore[misc,assignment]

        if isinstance(self._llm, _LlamaCppServerLLM):
            import httpx as _httpx

            # Sync probe — health_check is called from sync context
            try:
                with _httpx.Client(
                    base_url=getattr(self._llm, "server_url", ""),
                    timeout=_httpx.Timeout(5, connect=3),
                ) as c:
                    r = c.get("/health")
                    info["server_reachable"] = r.status_code == 200
                    if r.status_code == 200:
                        # Server reachable → mark as available and load it
                        self._llm.load_model()
                        info["available"] = True
            except Exception:
                info["server_reachable"] = False
                info["error"] = "Cannot reach llama.cpp server"
            info["server_url"] = getattr(self._llm, "server_url", "")
        return info


def get_llm_adapter(
    model_name: Optional[str] = None, use_mock: bool = None, **kwargs  # type: ignore
) -> LLMAdapter:
    """Get or create LLM adapter instance"""
    global _enhanced_llm_adapter

    explicit_request = (
        model_name is not None
        or use_mock is not None
        or bool(kwargs)
    )

    # When no explicit model/config is requested, prefer reusing the already-created
    # adapter to keep Story/agents consistent (runtime_preset may have configured it).
    if not explicit_request and _enhanced_llm_adapter is not None:
        return _enhanced_llm_adapter

    resolved_name: Optional[str] = None
    if model_name is not None and str(model_name).strip():
        resolved_name = str(model_name).strip()
    else:
        try:
            cfg = get_config()
            resolved_name = str(getattr(cfg.model, "chat_model", "") or "").strip() or None
        except Exception:
            resolved_name = None

        if resolved_name is None:
            resolved_name = os.getenv("MODEL_CHAT_MODEL") or "Qwen/Qwen-7B-Chat"

    if _enhanced_llm_adapter is None:
        _enhanced_llm_adapter = LLMAdapter(model_name=str(resolved_name), use_mock=use_mock, **kwargs)
        logger.info(f"Created LLM adapter for: {resolved_name} (mock={use_mock})")
        return _enhanced_llm_adapter

    # Explicit request: update adapter in-place if needed.
    if explicit_request:
        try:
            _enhanced_llm_adapter.reconfigure(str(resolved_name), use_mock=use_mock, **kwargs)
        except Exception:
            # Fall back to replacing the singleton as a last resort.
            _enhanced_llm_adapter = LLMAdapter(model_name=str(resolved_name), use_mock=use_mock, **kwargs)

    return _enhanced_llm_adapter


def reset_llm_adapter():
    """Reset the global LLM adapter (for testing)"""
    global _enhanced_llm_adapter

    if _enhanced_llm_adapter and hasattr(_enhanced_llm_adapter, "unload_model"):
        try:
            _enhanced_llm_adapter.unload_model(  # type: ignore[arg-type]
                getattr(_enhanced_llm_adapter, "model_name", None) or "unknown"
            )
        except TypeError:
            try:
                _enhanced_llm_adapter.unload_model()  # type: ignore
            except Exception:
                pass

    _enhanced_llm_adapter = None
    logger.info("LLM adapter reset")


# Backward compatibility - maintain old function signature
def get_enhanced_llm_adapter() -> EnhancedLLMAdapter:
    """Alias for get_llm_adapter for explicit enhanced features"""
    return get_llm_adapter()  # type: ignore
