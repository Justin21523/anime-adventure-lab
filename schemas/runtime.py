"""
Runtime preset schemas.

提供前端選擇硬體/品質預設（例如 RTX 5080 16GB）用。
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class RuntimePresetLLM(BaseModel):
    model_name: str = Field(..., min_length=1, description="LLM model id/name")
    torch_dtype: str = Field("float16", description="torch dtype (float16|bfloat16|float32)")
    device_map: str = Field("auto", description="Transformers device_map")
    use_quantization: bool = Field(True, description="Whether to use quantization (bitsandbytes)")
    quantization_bits: int = Field(4, ge=2, le=8, description="Quantization bits (4 or 8 recommended)")


class RuntimePresetT2I(BaseModel):
    model_id: Optional[str] = Field(None, description="Diffusion model id/path hint (optional)")
    torch_dtype: str = Field("float16", description="torch dtype for diffusion pipeline")
    enable_attention_slicing: bool = Field(True)
    enable_vae_slicing: bool = Field(True)
    enable_vae_tiling: bool = Field(False)
    enable_cpu_offload: bool = Field(False)
    enable_sequential_cpu_offload: bool = Field(False)

    default_width: int = Field(1024, ge=256, le=2048)
    default_height: int = Field(1024, ge=256, le=2048)
    default_steps: int = Field(30, ge=1, le=200)
    default_guidance_scale: float = Field(6.0, ge=0.0, le=20.0)

    max_width: int = Field(1024, ge=256, le=2048)
    max_height: int = Field(1024, ge=256, le=2048)
    max_steps: int = Field(50, ge=1, le=300)


class RuntimePreset(BaseModel):
    preset_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    description: str = Field("", description="Short description for UI")
    llm: RuntimePresetLLM = Field(..., description="LLM preset settings")
    t2i: RuntimePresetT2I = Field(default_factory=RuntimePresetT2I, description="T2I preset settings")


class RuntimePresetCatalogResponse(BaseModel):
    success: bool = True
    default_preset_id: str = Field(..., min_length=1)
    presets: List[RuntimePreset] = Field(default_factory=list)
