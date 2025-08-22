# core/performance/memory_manager.py
import gc
import torch
import psutil
from typing import Dict, Optional, Any
from contextlib import contextmanager
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    enable_8bit: bool = True
    enable_4bit: bool = False
    cpu_offload: bool = True
    sequential_cpu_offload: bool = False
    low_cpu_mem_usage: bool = True
    attention_slicing: bool = True
    vae_slicing: bool = True
    enable_xformers: bool = True
    torch_compile: bool = False  # experimental


class MemoryManager:
    """Unified memory management for all models"""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.loaded_models = {}
        self.memory_usage = {}

    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory usage"""
        info = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_gb": psutil.virtual_memory().available / 1024**3,
        }

        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_stats()
            info.update(
                {
                    "gpu_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                    "gpu_cached_gb": torch.cuda.memory_reserved() / 1024**3,
                    "gpu_max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
                    "gpu_total_gb": torch.cuda.get_device_properties(0).total_memory
                    / 1024**3,
                }
            )

        return info

    def setup_model_for_inference(self, model, model_type: str = "general"):
        """Apply memory optimizations to model"""
        if not hasattr(model, "device"):
            return model

        # Enable attention slicing for diffusion models
        if hasattr(model, "enable_attention_slicing") and self.config.attention_slicing:
            model.enable_attention_slicing(1)
            logger.info(f"Enabled attention slicing for {model_type}")

        # Enable VAE slicing for diffusion models
        if hasattr(model, "enable_vae_slicing") and self.config.vae_slicing:
            model.enable_vae_slicing()
            logger.info(f"Enabled VAE slicing for {model_type}")

        # Enable xformers if available
        if (
            hasattr(model, "enable_xformers_memory_efficient_attention")
            and self.config.enable_xformers
        ):
            try:
                model.enable_xformers_memory_efficient_attention()
                logger.info(f"Enabled xformers for {model_type}")
            except Exception as e:
                logger.warning(f"Failed to enable xformers: {e}")

        # CPU offload for low VRAM
        if hasattr(model, "enable_model_cpu_offload") and self.config.cpu_offload:
            model.enable_model_cpu_offload()
            logger.info(f"Enabled CPU offload for {model_type}")
        elif (
            hasattr(model, "enable_sequential_cpu_offload")
            and self.config.sequential_cpu_offload
        ):
            model.enable_sequential_cpu_offload()
            logger.info(f"Enabled sequential CPU offload for {model_type}")

        return model

    @contextmanager
    def managed_inference(self, model_key: str):
        """Context manager for inference with memory cleanup"""
        initial_memory = self.get_memory_info()

        try:
            yield
        finally:
            # Force cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            final_memory = self.get_memory_info()
            self.memory_usage[model_key] = {
                "initial": initial_memory,
                "final": final_memory,
                "gpu_delta": final_memory.get("gpu_allocated_gb", 0)
                - initial_memory.get("gpu_allocated_gb", 0),
            }

    def unload_model(self, model_key: str):
        """Safely unload model from memory"""
        if model_key in self.loaded_models:
            model = self.loaded_models[model_key]

            # Move to CPU
            if hasattr(model, "to"):
                model.to("cpu")

            # Delete reference
            del self.loaded_models[model_key]
            del model

            # Force cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            logger.info(f"Unloaded model: {model_key}")

    def get_quantization_config(self):
        """Get BitsAndBytes config for quantization"""
        if not (self.config.enable_4bit or self.config.enable_8bit):
            return None

        try:
            from transformers import BitsAndBytesConfig

            if self.config.enable_4bit:
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            elif self.config.enable_8bit:
                return BitsAndBytesConfig(
                    load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True
                )
        except ImportError:
            logger.warning("BitsAndBytes not available, skipping quantization")
            return None

    def cleanup_all(self):
        """Emergency cleanup - unload everything"""
        for key in list(self.loaded_models.keys()):
            self.unload_model(key)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        logger.info("Performed emergency cleanup")
