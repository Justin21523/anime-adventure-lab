# core/t2i/memory_utils.py
"""Memory optimization utilities for efficient VRAM usage"""

import torch
import psutil
import logging
from typing import List, Dict, Tuple, Any, Optional
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """Memory optimization utilities for efficient VRAM usage"""

    def __init__(self):
        self.optimizations_applied = []

    def optimize_pipeline(self, pipeline: DiffusionPipeline) -> DiffusionPipeline:
        """Apply comprehensive memory optimizations to pipeline"""
        self.optimizations_applied = []

        try:
            # Enable attention slicing
            if hasattr(pipeline, "enable_attention_slicing"):
                pipeline.enable_attention_slicing()
                self.optimizations_applied.append("attention_slicing")

            # Enable VAE slicing for large images
            if hasattr(pipeline, "enable_vae_slicing"):
                pipeline.enable_vae_slicing()
                self.optimizations_applied.append("vae_slicing")

            # Enable CPU offload if VRAM is limited
            if self._should_use_cpu_offload():
                if hasattr(pipeline, "enable_model_cpu_offload"):
                    pipeline.enable_model_cpu_offload()
                    self.optimizations_applied.append("cpu_offload")

            # Enable xFormers if available
            try:
                if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
                    pipeline.enable_xformers_memory_efficient_attention()
                    self.optimizations_applied.append("xformers")
            except Exception:
                logger.debug("xFormers not available, skipping")

            # Set memory format for efficiency
            if torch.cuda.is_available():
                try:
                    pipeline = pipeline.to(memory_format=torch.channels_last)
                    self.optimizations_applied.append("channels_last")
                except Exception:
                    logger.debug("Channels last memory format not supported")

            logger.info(f"Memory optimizations applied: {self.optimizations_applied}")
            return pipeline

        except Exception as e:
            logger.warning(f"Some memory optimizations failed: {e}")
            return pipeline

    def _should_use_cpu_offload(self) -> bool:
        """Determine if CPU offload should be used based on available VRAM"""
        if not torch.cuda.is_available():
            return False

        try:
            # Get GPU memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory
            total_memory_gb = total_memory / (1024**3)

            # Use CPU offload for GPUs with less than 8GB VRAM
            return total_memory_gb < 8.0

        except Exception:
            return False

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        stats = {
            "optimizations_applied": self.optimizations_applied,
            "system_ram_gb": round(psutil.virtual_memory().total / (1024**3), 1),
            "system_ram_available_gb": round(
                psutil.virtual_memory().available / (1024**3), 1
            ),
        }

        if torch.cuda.is_available():
            stats.update(
                {
                    "gpu_memory_allocated_mb": round(
                        torch.cuda.memory_allocated() / (1024**2), 1
                    ),
                    "gpu_memory_reserved_mb": round(
                        torch.cuda.memory_reserved() / (1024**2), 1
                    ),
                    "gpu_memory_total_gb": round(
                        torch.cuda.get_device_properties(0).total_memory / (1024**3), 1
                    ),
                }
            )

        return stats

    def clear_memory_cache(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU memory cache cleared")

    def estimate_memory_requirements(
        self, width: int, height: int, batch_size: int = 1
    ) -> Dict[str, float]:
        """Estimate memory requirements for generation"""
        # Base memory requirements (approximate, in MB)
        base_memory = {
            "model_weights": 3500,  # ~3.5GB for SD 1.5
            "intermediate_tensors": (width * height * batch_size)
            / 10000,  # Rough estimate
            "vae_decode": (width * height * 3 * 4) / (1024 * 1024),  # RGB float32
        }

        total_estimated = sum(base_memory.values())

        return {
            **base_memory,
            "total_estimated_mb": round(total_estimated, 1),
            "total_estimated_gb": round(total_estimated / 1024, 2),
            "recommended_optimizations": self._get_optimization_recommendations(
                total_estimated  # type: ignore
            ),
        }

    def _get_optimization_recommendations(
        self, estimated_memory_mb: float
    ) -> List[str]:
        """Get optimization recommendations based on estimated memory usage"""
        recommendations = []

        if estimated_memory_mb > 4000:  # 4GB+
            recommendations.extend(["attention_slicing", "vae_slicing"])

        if estimated_memory_mb > 6000:  # 6GB+
            recommendations.append("cpu_offload")

        if estimated_memory_mb > 8000:  # 8GB+
            recommendations.extend(["gradient_checkpointing", "fp16"])

        return recommendations
