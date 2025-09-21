# core/llm/model_loader.py
"""
LLM Model Loader
Handles loading, caching, and memory management for LLM models
"""

import json
import torch
import logging
import hashlib
import gc
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from ..config import get_config
from ..shared_cache import get_shared_cache
from ..exceptions import ModelLoadError, CUDAOutOfMemoryError, handle_cuda_oom
from .base import BaseLLM

logger = logging.getLogger(__name__)


class ModelLoadConfig:
    """Model loading configuration"""

    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        torch_dtype: str = "float16",
        use_quantization: bool = True,
        quantization_bits: int = 4,
        use_flash_attention: bool = True,
        trust_remote_code: bool = True,
        low_cpu_mem_usage: bool = True,
        **kwargs,
    ):
        self.model_name = model_name
        self.device_map = device_map
        # Handle torch_dtype conversion
        if isinstance(torch_dtype, str):
            if torch_dtype == "float16":
                self.torch_dtype = torch.float16
            elif torch_dtype == "float32":
                self.torch_dtype = torch.float32
            elif torch_dtype == "bfloat16":
                self.torch_dtype = torch.bfloat16
            else:
                self.torch_dtype = getattr(torch, torch_dtype, torch.float16)
        else:
            self.torch_dtype = torch_dtype

        # Auto-detect quantization based on GPU availability
        if use_quantization is None:
            self.use_quantization = torch.cuda.is_available()
        else:
            self.use_quantization = use_quantization and torch.cuda.is_available()

        self.quantization_bits = quantization_bits
        self.use_flash_attention = use_flash_attention
        self.trust_remote_code = trust_remote_code
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.kwargs = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "model_name": self.model_name,
            "device_map": self.device_map,
            "torch_dtype": str(self.torch_dtype),
            "use_quantization": self.use_quantization,
            "quantization_bits": self.quantization_bits,
            "use_flash_attention": self.use_flash_attention,
            "trust_remote_code": self.trust_remote_code,
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
            **self.kwargs,
        }

    def get_cache_key(self) -> str:
        """Generate cache key for this configuration"""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def __repr__(self) -> str:
        return (
            f"ModelLoadConfig({self.model_name}, {self.device_map}, {self.torch_dtype})"
        )


class ModelLoader:
    """Advanced model loader with caching and memory management"""

    def __init__(self):
        self.config = get_config()
        self.cache = get_shared_cache()
        self._loaded_models: Dict[str, Dict[str, Any]] = {}
        self._model_configs: Dict[str, ModelLoadConfig] = {}
        # In-memory model cache
        self._model_cache: Dict[
            str, Dict[str, Union[PreTrainedModel, PreTrainedTokenizer]]
        ] = {}
        self._config_cache: Dict[str, ModelLoadConfig] = {}
        logger.info("ModelLoader initialized")

    @handle_cuda_oom
    def load_model(
        self, model_name: str, load_config: Optional[ModelLoadConfig] = None
    ) -> Dict[str, Union[PreTrainedModel, PreTrainedTokenizer]]:
        """
        Load model with advanced configuration and caching

        Args:
            model_name: HuggingFace model name
            load_config: Model loading configuration

        Returns:
            Dict containing 'model' and 'tokenizer'
        """
        # Use default config if none provided
        if load_config is None:
            load_config = self._get_default_config(model_name)

        cache_key = f"{model_name}_{load_config.get_cache_key()}"

        # Check if already loaded
        if cache_key in self._loaded_models:
            logger.info(f"Model {model_name} already loaded, returning cached instance")
            return self._loaded_models[cache_key]

        # Check if model is already loaded
        if cache_key in self._model_cache:
            logger.info(f"Using cached model: {model_name}")
            return self._model_cache[cache_key]

        logger.info(f"Loading model: {model_name}")

        try:
            # Clear GPU cache before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            # Load tokenizer first
            tokenizer = self._load_tokenizer(model_name, load_config)

            # Load model with optimizations
            model = self._load_model_with_config(model_name, load_config)

            # Cache the loaded model
            model_dict = {"model": model, "tokenizer": tokenizer}

            self._model_cache[cache_key] = model_dict
            self._config_cache[cache_key] = load_config
            self._loaded_models[cache_key] = model_dict
            self._model_configs[cache_key] = load_config

            # Log model info
            self._log_model_info(model_name, model, load_config)

            # Save loading metadata to cache
            self._save_model_metadata(cache_key, model_name, load_config)

            return model_dict

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")

            # Clean up on failure
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            raise ModelLoadError(model_name, str(e))

    def _load_tokenizer(
        self, model_name: str, load_config: ModelLoadConfig
    ) -> PreTrainedTokenizer:
        """Load tokenizer with proper configuration"""
        logger.info(f"Loading tokenizer for: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=load_config.trust_remote_code,
            cache_dir=Path(self.cache.cache_root) / "hf",
            **{
                k: v
                for k, v in load_config.kwargs.items()
                if k.startswith("tokenizer_")
            },
        )

        # Setup special tokens
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        return tokenizer

    def _load_model_with_config(
        self, model_name: str, load_config: ModelLoadConfig
    ) -> PreTrainedModel:
        """Load model with advanced configuration"""
        logger.info(f"Loading model: {model_name}")

        # Setup quantization
        quantization_config = None
        if load_config.use_quantization:
            quantization_config = self._create_quantization_config(load_config)

        # Model loading arguments
        model_kwargs = {
            "device_map": load_config.device_map,
            "torch_dtype": load_config.torch_dtype,
            "trust_remote_code": load_config.trust_remote_code,
            "low_cpu_mem_usage": load_config.low_cpu_mem_usage,
            "cache_dir": Path(self.cache.cache_root) / "hf",
        }
        # Add device map if not CPU-only
        if load_config.device_map != "cpu":
            model_kwargs["device_map"] = load_config.device_map

        # Add quantization config
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        # Add custom kwargs (filter out tokenizer-specific ones)
        custom_kwargs = {
            k: v
            for k, v in load_config.kwargs.items()
            if not k.startswith("tokenizer_")
        }
        model_kwargs.update(custom_kwargs)

        # Load the model
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        except Exception as e:
            # Fallback: try loading with minimal config
            logger.warning(f"Failed to load with full config, trying minimal: {e}")
            minimal_kwargs = {
                "torch_dtype": torch.float32,
                "trust_remote_code": True,
                "cache_dir": Path(self.cache.cache_root) / "hf",
            }
            model = AutoModelForCausalLM.from_pretrained(model_name, **minimal_kwargs)

        # Apply post-loading optimizations
        self._apply_model_optimizations(model, load_config)

        return model

    def _create_quantization_config(
        self, load_config: ModelLoadConfig
    ) -> BitsAndBytesConfig:
        """Create quantization configuration"""
        if load_config.quantization_bits == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=load_config.torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_config.quantization_bits == 8:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
        else:
            raise ValueError(
                f"Unsupported quantization bits: {load_config.quantization_bits}"
            )

    def _apply_model_optimizations(
        self, model: PreTrainedModel, load_config: ModelLoadConfig
    ):
        """Apply post-loading optimizations"""
        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
            except Exception as e:
                logger.warning(f"Failed to enable gradient checkpointing: {e}")

        # Enable attention slicing for VRAM optimization
        if hasattr(model, "enable_attention_slicing"):
            try:
                model.enable_attention_slicing("auto")  # type: ignore
                logger.info("Attention slicing enabled")
            except Exception as e:
                logger.warning(f"Failed to enable attention slicing: {e}")

        # Set model to eval mode for inference
        model.eval()

        logger.info("Applied model optimizations")

    def _get_default_config(self, model_name: str) -> ModelLoadConfig:
        """Get default loading configuration for model"""
        # Get base config from app config
        device_map = self.config.model.device_map
        torch_dtype = self.config.model.torch_dtype
        use_quantization = self.config.model.use_4bit_loading

        return ModelLoadConfig(
            model_name=model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            use_quantization=use_quantization,
        )

    def _log_model_info(
        self, model_name: str, model: PreTrainedModel, load_config: ModelLoadConfig
    ):
        """Log model loading information"""
        try:
            param_count = sum(p.numel() for p in model.parameters()) // 1_000_000
            device = (
                next(model.parameters()).device if model.parameters() else "unknown"
            )

            logger.info(f"Model {model_name} loaded successfully:")
            logger.info(f"  - Parameters: {param_count}M")
            logger.info(f"  - Device: {device}")
            logger.info(f"  - Dtype: {load_config.torch_dtype}")
            logger.info(f"  - Quantized: {load_config.use_quantization}")

            # GPU memory info
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(
                    f"  - GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved"
                )

        except Exception as e:
            logger.warning(f"Failed to log model info: {e}")

    def _save_model_metadata(
        self, cache_key: str, model_name: str, load_config: ModelLoadConfig
    ):
        """Save model metadata to cache"""
        try:
            metadata = {
                "model_name": model_name,
                "cache_key": cache_key,
                "config": load_config.to_dict(),
                "loaded_at": (
                    torch.cuda.Event(enable_timing=True)
                    if torch.cuda.is_available()
                    else None
                ),
            }

            metadata_path = (
                Path(self.cache.cache_root)
                / "models"
                / "metadata"
                / f"{cache_key}.json"
            )
            metadata_path.parent.mkdir(parents=True, exist_ok=True)

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

        except Exception as e:
            logger.warning(f"Failed to save model metadata: {e}")

    def unload_model(self, model_name: str, config_hash: Optional[str] = None) -> bool:
        """
        Unload specific model from memory

        Args:
            model_name: Model name to unload
            config_hash: Specific config hash (optional)

        Returns:
            True if model was unloaded, False if not found
        """
        unloaded = False

        # If config_hash provided, unload specific instance
        if config_hash:
            cache_key = f"{model_name}_{config_hash}"
            if cache_key in self._loaded_models:
                del self._loaded_models[cache_key]
                del self._model_configs[cache_key]
                unloaded = True
                logger.info(f"Unloaded model: {cache_key}")
        else:
            # Unload all instances of this model
            keys_to_remove = [
                k for k in self._loaded_models.keys() if k.startswith(f"{model_name}_")
            ]
            for key in keys_to_remove:
                del self._loaded_models[key]
                if key in self._model_configs:
                    del self._model_configs[key]
                unloaded = True
                logger.info(f"Unloaded model: {key}")

        # Clear GPU cache if any model was unloaded
        if unloaded:
            # Force garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            logger.info(f"Unloaded model: {model_name}")

        return unloaded

    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models"""
        return list(set(key.split(":")[0] for key in self._model_cache.keys()))

    def unload_all(self) -> int:
        """Unload all models from memory"""
        count = len(self._loaded_models)
        self._loaded_models.clear()
        self._model_configs.clear()

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Unloaded {count} models, GPU cache cleared")
        return count

    def list_loaded_models(self) -> List[Dict[str, Any]]:
        """List all loaded models with their configurations"""
        models = []
        for cache_key, model_dict in self._loaded_models.items():
            config = self._model_configs.get(cache_key)
            if config:
                model_info = {
                    "cache_key": cache_key,
                    "model_name": config.model_name,
                    "device_map": config.device_map,
                    "torch_dtype": str(config.torch_dtype),
                    "quantized": config.use_quantization,
                }

                # Add parameter count if available
                try:
                    model = model_dict["model"]
                    model_info["parameters_m"] = (
                        sum(p.numel() for p in model.parameters()) // 1_000_000
                    )
                    model_info["device"] = str(next(model.parameters()).device)
                except:
                    pass

                models.append(model_info)

        return models

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        stats = {
            "loaded_models_count": len(self._loaded_models),
            "cpu_memory_mb": 0,  # Would need psutil for accurate measurement
            "gpu_memory": {},
        }

        if torch.cuda.is_available():
            stats["gpu_memory"] = {
                "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "max_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
            }

        return stats

    def cleanup(self):
        """Clean up all loaded models"""
        model_count = len(self._model_cache)

        self._model_cache.clear()
        self._config_cache.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        logger.info(f"Cleaned up {model_count} models from cache")


# Global model loader instance
_model_loader: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """Get global model loader instance"""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader
