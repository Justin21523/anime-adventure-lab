# core/performance/quantization.py
"""
Model Quantization Support

提供模型量化功能以減少記憶體使用和提升推理速度：
- 8-bit quantization (LLM.int8())
- 4-bit quantization (QLoRA)
- Dynamic quantization for inference
"""

import logging
from typing import Optional, Dict, Any, Literal
from dataclasses import dataclass
import torch
from transformers import BitsAndBytesConfig

logger = logging.getLogger(__name__)


QuantizationMode = Literal["none", "int8", "int4", "dynamic"]


@dataclass
class QuantizationConfig:
    """Quantization configuration"""

    mode: QuantizationMode = "none"

    # 8-bit quantization settings
    load_in_8bit: bool = False
    llm_int8_threshold: float = 6.0
    llm_int8_has_fp16_weight: bool = False

    # 4-bit quantization settings
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"  # or "bfloat16"
    bnb_4bit_quant_type: str = "nf4"  # or "fp4"
    bnb_4bit_use_double_quant: bool = True

    # Dynamic quantization settings
    dynamic_quant_dtype: str = "qint8"  # or "quint8", "qint32"

    def __post_init__(self):
        """Validate and auto-configure settings"""
        if self.mode == "int8":
            self.load_in_8bit = True
            self.load_in_4bit = False
        elif self.mode == "int4":
            self.load_in_4bit = True
            self.load_in_8bit = False
        elif self.mode == "none":
            self.load_in_8bit = False
            self.load_in_4bit = False


class QuantizationManager:
    """Manage model quantization for different model types"""

    def __init__(self, config: Optional[QuantizationConfig] = None):
        self.config = config or QuantizationConfig()

    def get_bnb_config(self) -> Optional[BitsAndBytesConfig]:
        """
        Get BitsAndBytes configuration for model loading

        Returns:
            BitsAndBytesConfig if quantization is enabled, None otherwise
        """
        if self.config.mode == "none":
            return None

        try:
            if self.config.load_in_8bit:
                logger.info("Using 8-bit quantization (LLM.int8())")
                return BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=self.config.llm_int8_threshold,
                    llm_int8_has_fp16_weight=self.config.llm_int8_has_fp16_weight,
                )

            elif self.config.load_in_4bit:
                logger.info("Using 4-bit quantization (QLoRA)")

                # Determine compute dtype
                compute_dtype = torch.float16
                if self.config.bnb_4bit_compute_dtype == "bfloat16":
                    compute_dtype = torch.bfloat16

                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
                )

        except Exception as e:
            logger.error(f"Failed to create BitsAndBytes config: {e}")
            return None

        return None

    def quantize_model_dynamic(
        self, model: torch.nn.Module, dtype: Optional[torch.dtype] = None
    ) -> torch.nn.Module:
        """
        Apply dynamic quantization to a model

        Dynamic quantization is best for models where inference speed
        is more important than memory usage (e.g., CPU inference)

        Args:
            model: PyTorch model to quantize
            dtype: Quantization data type (default: torch.qint8)

        Returns:
            Quantized model
        """
        if self.config.mode != "dynamic":
            logger.warning("Dynamic quantization not enabled in config")
            return model

        try:
            dtype = dtype or torch.qint8

            # Quantize linear and LSTM layers
            quantized_model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear, torch.nn.LSTM}, dtype=dtype
            )

            logger.info(f"Applied dynamic quantization with dtype={dtype}")
            return quantized_model

        except Exception as e:
            logger.error(f"Dynamic quantization failed: {e}")
            return model

    def get_model_load_kwargs(self) -> Dict[str, Any]:
        """
        Get model loading kwargs including quantization config

        Returns:
            Dictionary of kwargs for model loading
        """
        kwargs: Dict[str, Any] = {}

        # Add quantization config
        bnb_config = self.get_bnb_config()
        if bnb_config is not None:
            kwargs["quantization_config"] = bnb_config

            # Adjust device_map for quantized models
            kwargs["device_map"] = "auto"

            # Disable cache for quantized models (saves memory)
            kwargs["use_cache"] = False

        return kwargs

    def estimate_memory_savings(
        self, original_params: int, mode: Optional[QuantizationMode] = None
    ) -> Dict[str, Any]:
        """
        Estimate memory savings from quantization

        Args:
            original_params: Number of parameters in original model
            mode: Quantization mode (uses self.config.mode if not specified)

        Returns:
            Dictionary with memory estimates
        """
        mode = mode or self.config.mode

        # Base memory (float32)
        base_memory_gb = original_params * 4 / (1024**3)

        if mode == "none":
            quantized_memory_gb = base_memory_gb
            bits_per_param = 32
            savings_pct = 0.0

        elif mode == "int8":
            # 8-bit quantization
            bits_per_param = 8
            quantized_memory_gb = base_memory_gb / 4
            savings_pct = 75.0

        elif mode == "int4":
            # 4-bit quantization
            # NF4 with double quant is slightly more than 4 bits
            bits_per_param = 4.5
            quantized_memory_gb = base_memory_gb * 4.5 / 32
            savings_pct = 85.9

        elif mode == "dynamic":
            # Dynamic quantization (approximate)
            bits_per_param = 8
            quantized_memory_gb = base_memory_gb * 0.3  # Only some layers quantized
            savings_pct = 70.0

        else:
            bits_per_param = 32
            quantized_memory_gb = base_memory_gb
            savings_pct = 0.0

        return {
            "mode": mode,
            "original_memory_gb": round(base_memory_gb, 2),
            "quantized_memory_gb": round(quantized_memory_gb, 2),
            "memory_saved_gb": round(base_memory_gb - quantized_memory_gb, 2),
            "savings_percentage": round(savings_pct, 1),
            "bits_per_parameter": bits_per_param,
        }

    def get_recommended_mode(
        self, available_vram_gb: float, model_params: int
    ) -> QuantizationMode:
        """
        Recommend quantization mode based on available VRAM

        Args:
            available_vram_gb: Available VRAM in GB
            model_params: Number of model parameters

        Returns:
            Recommended quantization mode
        """
        # Estimate memory requirements
        fp32_memory = model_params * 4 / (1024**3)
        fp16_memory = model_params * 2 / (1024**3)
        int8_memory = model_params * 1 / (1024**3)
        int4_memory = model_params * 0.5 / (1024**3)

        # Add 20% overhead for activations and buffers
        overhead_factor = 1.2

        if available_vram_gb >= fp16_memory * overhead_factor:
            return "none"  # Can run in fp16 without quantization
        elif available_vram_gb >= int8_memory * overhead_factor:
            return "int8"
        elif available_vram_gb >= int4_memory * overhead_factor:
            return "int4"
        else:
            logger.warning(
                f"Insufficient VRAM ({available_vram_gb}GB) for model "
                f"({model_params} params). Consider using CPU or smaller model."
            )
            return "int4"  # Use most aggressive quantization

    def log_quantization_info(self):
        """Log current quantization configuration"""
        logger.info("=" * 60)
        logger.info("Quantization Configuration")
        logger.info("=" * 60)
        logger.info(f"Mode: {self.config.mode}")

        if self.config.mode == "int8":
            logger.info(f"  8-bit threshold: {self.config.llm_int8_threshold}")
            logger.info(f"  FP16 weights: {self.config.llm_int8_has_fp16_weight}")

        elif self.config.mode == "int4":
            logger.info(f"  Compute dtype: {self.config.bnb_4bit_compute_dtype}")
            logger.info(f"  Quant type: {self.config.bnb_4bit_quant_type}")
            logger.info(f"  Double quant: {self.config.bnb_4bit_use_double_quant}")

        elif self.config.mode == "dynamic":
            logger.info(f"  Target dtype: {self.config.dynamic_quant_dtype}")

        logger.info("=" * 60)


# Convenience functions
def create_quantization_config(
    mode: QuantizationMode = "none", **kwargs
) -> QuantizationConfig:
    """
    Create quantization configuration

    Args:
        mode: Quantization mode ("none", "int8", "int4", "dynamic")
        **kwargs: Additional configuration options

    Returns:
        QuantizationConfig object
    """
    return QuantizationConfig(mode=mode, **kwargs)


def get_quantization_manager(
    config: Optional[QuantizationConfig] = None,
) -> QuantizationManager:
    """
    Get quantization manager

    Args:
        config: Quantization configuration (creates default if None)

    Returns:
        QuantizationManager instance
    """
    return QuantizationManager(config)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test 8-bit quantization
    config_8bit = QuantizationConfig(mode="int8")
    manager_8bit = QuantizationManager(config_8bit)
    manager_8bit.log_quantization_info()

    # Test memory estimation
    model_params = 7_000_000_000  # 7B model
    estimates = manager_8bit.estimate_memory_savings(model_params)
    print("\nMemory Estimates (7B model):")
    for key, value in estimates.items():
        print(f"  {key}: {value}")

    # Test recommended mode
    available_vram = 8.0  # 8GB VRAM
    recommended = manager_8bit.get_recommended_mode(available_vram, model_params)
    print(f"\nRecommended mode for 8GB VRAM: {recommended}")
