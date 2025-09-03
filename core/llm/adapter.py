"""
LLM Implementation Adapters
Concrete implementations for different model types
"""

import torch
import logging
from typing import List, Dict, Any, Optional, Union
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)

from .base import BaseLLM, ChatMessage, LLMResponse
from ..exceptions import (
    ModelLoadError,
    ModelNotFoundError,
    CUDAOutOfMemoryError,
    handle_cuda_oom,
    handle_model_error,
)
from ..config import get_config
from ..shared_cache import get_shared_cache

logger = logging.getLogger(__name__)


class TransformersLLM(BaseLLM):
    """Transformers-based LLM implementation (Qwen/Llama/etc.)"""

    def __init__(
        self, model_name: str, device_map: str = "auto", torch_dtype: str = "float16"
    ):
        super().__init__(model_name)
        self.device_map = device_map
        self.torch_dtype = getattr(torch, torch_dtype)
        self.config = get_config()
        self.cache = get_shared_cache()

    @handle_cuda_oom
    @handle_model_error
    def load_model(self) -> None:
        """Load Transformers model with optimizations"""
        if self._loaded:
            return

        try:
            logger.info(f"Loading LLM model: {self.model_name}")

            # Setup quantization for low VRAM
            quantization_config = None
            if self.config.get("performance.low_vram_mode", True):
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                cache_dir=self.cache.cache_root / "hf",  # type: ignore
            )

            # Ensure pad token exists
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # Load model
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device_map,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
                quantization_config=quantization_config,
                cache_dir=self.cache.cache_root / "hf",  # type: ignore
                low_cpu_mem_usage=True,
            )

            # Enable optimizations
            if hasattr(self._model, "gradient_checkpointing_enable"):
                self._model.gradient_checkpointing_enable()

            self._loaded = True
            logger.info(f"LLM model loaded successfully: {self.model_name}")

            # Cache model info
            model_info = {
                "model_name": self.model_name,
                "device_map": str(self.device_map),
                "torch_dtype": str(self.torch_dtype),
                "quantized": quantization_config is not None,
                "parameters": sum(p.numel() for p in self._model.parameters())
                // 1_000_000,
            }
            self.cache.cache_model_info(  # type: ignore
                f"llm_{self.model_name.replace('/', '_')}", model_info
            )

        except Exception as e:
            logger.error(f"Failed to load LLM {self.model_name}: {e}")
            raise ModelLoadError(self.model_name, str(e))

    def unload_model(self) -> None:
        """Unload model to free GPU memory"""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        self._loaded = False

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"LLM model unloaded: {self.model_name}")

    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs,
    ) -> str:
        """Generate text from prompt"""
        if not self.is_available():
            raise ModelNotFoundError(self.model_name)

        self.validate_input(prompt)

        try:
            # Tokenize input
            inputs = self._tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048  # type: ignore
            )

            # Move to model device
            device = next(self._model.parameters()).device  # type: ignore
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self._model.generate(  # type: ignore
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self._tokenizer.pad_token_id,  # type: ignore
                    eos_token_id=self._tokenizer.eos_token_id,  # type: ignore
                    **kwargs,
                )

            # Decode response
            generated_text = self._tokenizer.decode(  # type: ignore
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

            return generated_text.strip()

        except Exception as e:
            logger.error(f"Generation failed for {self.model_name}: {e}")
            raise ModelError(f"Generation failed: {str(e)}", self.model_name)  # type: ignore

    def chat(
        self,
        messages: List[Union[ChatMessage, Dict[str, str]]],
        max_length: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        """Chat completion with message history"""
        if not self.is_available():
            raise ModelNotFoundError(self.model_name)

        # Format messages
        formatted_messages = self.format_messages(messages)

        # Build conversation prompt
        prompt = self._build_chat_prompt(formatted_messages)

        # Generate response
        response_text = self.generate(
            prompt, max_length=max_length, temperature=temperature, **kwargs
        )

        return LLMResponse(
            content=response_text,
            model_name=self.model_name,
            usage={
                "prompt_tokens": len(self._tokenizer.encode(prompt)),  # type: ignore
                "completion_tokens": len(self._tokenizer.encode(response_text)),  # type: ignore
                "total_tokens": len(self._tokenizer.encode(prompt + response_text)),  # type: ignore
            },
        )

    def _build_chat_prompt(self, messages: List[ChatMessage]) -> str:
        """Build chat prompt from messages - override per model"""
        prompt_parts = []

        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")

        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)

    def is_available(self) -> bool:
        """Check if model is loaded and available"""
        return self._loaded and self._model is not None


class QwenLLM(TransformersLLM):
    """Qwen model implementation"""

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


class LlamaLLM(TransformersLLM):
    """Llama model implementation"""

    def _build_chat_prompt(self, messages: List[ChatMessage]) -> str:
        """Llama-specific chat format"""
        conversation = []

        system_message = None
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            elif msg.role == "user" or msg.role == "assistant":
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


class LLMAdapter:
    """LLM Adapter Factory - manages different LLM implementations"""

    def __init__(self):
        self.config = get_config()
        self.cache = get_shared_cache()
        self._models: Dict[str, BaseLLM] = {}

    def get_llm(
        self, model_name: Optional[str] = None, model_type: str = "auto"
    ) -> BaseLLM:
        """Get or create LLM instance"""
        if model_name is None:
            model_name = self.config.model.chat_model  # type: ignore

        if model_name in self._models:
            return self._models[model_name]

        # Determine model type
        if model_type == "auto":
            if "qwen" in model_name.lower():  # type: ignore
                model_type = "qwen"
            elif "llama" in model_name.lower():  # type: ignore
                model_type = "llama"
            else:
                model_type = "generic"

        # Create appropriate LLM instance
        if model_type == "qwen":
            llm = QwenLLM(
                model_name, self.config.model.device, self.config.model.torch_dtype  # type: ignore
            )
        elif model_type == "llama":
            llm = LlamaLLM(
                model_name, self.config.model.device, self.config.model.torch_dtype  # type: ignore
            )
        else:
            llm = TransformersLLM(
                model_name, self.config.model.device, self.config.model.torch_dtype  # type: ignore
            )

        self._models[model_name] = llm  # type: ignore
        return llm

    def chat(
        self,
        messages: List[Union[ChatMessage, Dict[str, str]]],
        model_name: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Convenient chat interface"""
        llm = self.get_llm(model_name)
        if not llm.is_available():
            llm.load_model()
        return llm.chat(messages, **kwargs)

    def generate(self, prompt: str, model_name: Optional[str] = None, **kwargs) -> str:
        """Convenient generation interface"""
        llm = self.get_llm(model_name)
        if not llm.is_available():
            llm.load_model()
        return llm.generate(prompt, **kwargs)

    def unload_all(self) -> None:
        """Unload all models to free memory"""
        for model in self._models.values():
            model.unload_model()
        self._models.clear()

    def list_loaded_models(self) -> List[str]:
        """List currently loaded models"""
        return [name for name, model in self._models.items() if model.is_available()]


# Global adapter instance
_llm_adapter: Optional[LLMAdapter] = None


def get_llm_adapter() -> LLMAdapter:
    """Get global LLM adapter instance"""
    global _llm_adapter
    if _llm_adapter is None:
        _llm_adapter = LLMAdapter()
    return _llm_adapter
