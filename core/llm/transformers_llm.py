# core/llm/transformers_llm.py
import logging
import time, json
import random
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from .base import MinimalLLM
from ..shared_cache import get_shared_cache
from ..config import get_config
from .adapter import LLMAdapter


class TransformersLLM(LLMAdapter):
    """HuggingFace Transformers LLM implementation with low-VRAM support"""

    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", **kwargs):
        super().__init__(model_name, **kwargs)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache = get_shared_cache()
        self.use_4bit = kwargs.get("use_4bit", True)
        self.use_8bit = kwargs.get("use_8bit", False)
        self.trust_remote_code = kwargs.get("trust_remote_code", False)

        self.tokenizer = Optional[AutoTokenizer] = None  # type: ignore
        self.model = Optional[AutoModelForCausalLM] = None  # type: ignore
        self._load_model()

    def _load_model(self):
        """Load model with memory optimization"""
        try:
            # Setup quantization config for low VRAM
            quantization_config = None
            if self.use_4bit and torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            elif self.use_8bit and torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                padding_side="left",
            )

            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with optimizations
            model_kwargs = {
                "trust_remote_code": self.trust_remote_code,
                "torch_dtype": (
                    torch.bfloat16 if torch.cuda.is_available() else torch.float32
                ),
                "device_map": "auto" if torch.cuda.is_available() else None,
            }

            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            else:
                model_kwargs["low_cpu_mem_usage"] = True

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, **model_kwargs
            )

            print(f"Loaded {self.model_name} on {self.device}")

        except Exception as e:
            print(f"Failed to load {self.model_name}: {e}")
            self.model = None
            self.tokenizer = None

    def is_available(self) -> bool:
        """Check if model is loaded and available"""
        return self.model is not None and self.tokenizer is not None

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Generate response from chat messages"""
        if not self.is_available():
            return "模型未載入，請檢查配置"

        try:
            # Convert messages to prompt text
            prompt = self._messages_to_prompt(messages)

            # Tokenize
            inputs = self.tokenizer(  # type: ignore
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True,
            )

            if torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(  # type: ignore
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,  # type: ignore
                    eos_token_id=self.tokenizer.eos_token_id,  # type: ignore
                    **kwargs,
                )

            # Decode only the new tokens
            input_length = inputs["input_ids"].shape[1]
            new_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)  # type: ignore

            return response.strip()

        except Exception as e:
            print(f"Generation error: {e}")
            return f"生成錯誤: {str(e)}"

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to prompt format"""
        prompt_parts = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                prompt_parts.append(f"系統: {content}")
            elif role == "user":
                prompt_parts.append(f"用戶: {content}")
            elif role == "assistant":
                prompt_parts.append(f"助手: {content}")

        prompt_parts.append("助手:")
        return "\n".join(prompt_parts)
