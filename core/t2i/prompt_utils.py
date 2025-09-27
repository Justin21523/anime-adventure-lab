# core/t2i/prompt_utils.py
"""Prompt processing and enhancement utilities"""

import time
import re
import torch
import json
from PIL.PngImagePlugin import PngInfo
import io
import base64
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
from datetime import datetime

from .lora_manager import LoRAManager
from .pipeline import PipelineManager
from .memory_utils import MemoryOptimizer
from .controlnet import ControlNetManager
from ..shared_cache import get_shared_cache
from ..config import get_config

logger = logging.getLogger(__name__)


class PromptProcessor:
    """Prompt processing and enhancement utilities"""

    def __init__(self):
        self.cache = get_shared_cache()
        self.config = get_config()
        self.device = self.config.model.device
        # Initialize all required attributes
        self.quality_boosters = [
            "masterpiece",
            "best quality",
            "high quality",
            "detailed",
        ]

        self.negative_defaults = [
            "worst quality",
            "low quality",
            "blurry",
            "bad anatomy",
            "bad hands",
            "text",
            "watermark",
            "signature",
        ]
        self.max_prompt_length = 500
        self.auto_enhance = True
        # Common prompt patterns
        self.weight_pattern = re.compile(r"\(([^)]+)\)")
        self.attention_pattern = re.compile(r"\[([^\]]+)\]")

        # Component references - initialized as None
        self.lora_manager = None
        self.safety_engine = None
        self.compliance_logger = None
        self.controlnet_manager = None
        self.attribution_manager = None
        self.generation_stats = None

        # Processing stats
        self.processing_stats = {
            "total_processed": 0,
            "enhanced_prompts": 0,
            "filtered_prompts": 0,
        }
        try:
            self.pipeline_manager = PipelineManager(
                str(self.cache.cache_root), self.device
            )
            logger.debug("PipelineManager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize PipelineManager: {e}")
            self.pipeline_manager = None

        try:
            self.lora_manager = LoRAManager(str(self.cache.cache_root))
            logger.debug("LoRAManager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize LoRAManager: {e}")
            self.lora_manager = None

        try:
            self.controlnet_manager = ControlNetManager(str(self.cache.cache_root))
            logger.debug("ControlNetManager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize ControlNetManager: {e}")
            self.controlnet_manager = None

        try:
            self.memory_optimizer = MemoryOptimizer()
            logger.debug("MemoryOptimizer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize MemoryOptimizer: {e}")
            self.memory_optimizer = None

        try:
            self.prompt_processor = PromptProcessor()
            logger.debug("PromptProcessor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize PromptProcessor: {e}")
            self.prompt_processor = None

        try:
            from .model_config import ModelConfigManager

            self.model_config_manager = ModelConfigManager(str(self.cache.cache_root))
            logger.debug("ModelConfigManager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize ModelConfigManager: {e}")
            self.model_config_manager = None

        # Initialize safety components
        try:
            from ..safety.detector import SafetyEngine

            self.safety_engine = SafetyEngine(self.config.get("safety", {}))
            logger.debug("SafetyEngine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize SafetyEngine: {e}")
            self.safety_engine = None

        try:
            from ..safety.watermark import AttributionManager, ComplianceLogger

            self.attribution_manager = AttributionManager(str(self.cache.cache_root))
            self.compliance_logger = ComplianceLogger(str(self.cache.cache_root))
            logger.debug("Safety components initialized")
        except Exception as e:
            logger.error(f"Failed to initialize safety components: {e}")
            self.attribution_manager = None
            self.compliance_logger = None

        logger.info(f"T2I Engine initialized with device: {self.device}")
        logger.info(
            f"Component status: PM={self.pipeline_manager is not None}, "
            f"LM={self.lora_manager is not None}, "
            f"CM={self.controlnet_manager is not None}, "
            f"SE={self.safety_engine is not None}"
        )

        logger.info("PromptProcessor initialized")

    def initialize(self, components: Dict = None):  # type: ignore
        """Initialize component references"""
        if components:
            self.lora_manager = components.get("lora_manager")
            self.safety_engine = components.get("safety_engine")
            self.compliance_logger = components.get("compliance_logger")
            self.controlnet_manager = components.get("controlnet_manager")
            self.attribution_manager = components.get("attribution_manager")
            self.generation_stats = components.get("generation_stats")
        logger.debug("PromptProcessor components initialized")

    def process(
        self, prompt: str, is_negative: bool = False, enhance: bool = True
    ) -> str:
        """Process prompt with safety and enhancement"""
        if not prompt or not prompt.strip():
            return ""

        self.processing_stats["total_processed"] += 1

        # Clean prompt
        processed = self._clean_prompt(prompt)

        # Safety check if available
        if self.safety_engine:
            safety_result = self.safety_engine.check_prompt(processed)
            if not safety_result["is_safe"]:
                self.processing_stats["filtered_prompts"] += 1
                return "safe content" if not is_negative else "inappropriate content"

        # Enhancement
        if enhance and not is_negative:
            processed = self._add_quality_boosters(processed)
            self.processing_stats["enhanced_prompts"] += 1
        elif enhance and is_negative:
            processed = self._add_negative_defaults(processed)

        # Length validation
        processed = self._validate_length(processed)
        return processed

    def _has_unsafe_content(self, prompt: str) -> bool:
        """Simple unsafe content detection - NO EXTERNAL DEPENDENCIES"""
        unsafe_terms = ["nsfw", "explicit", "nude", "sexual", "porn", "violence"]
        prompt_lower = prompt.lower()
        return any(term in prompt_lower for term in unsafe_terms)

    def _clean_prompt(self, prompt: str) -> str:
        """Clean and normalize prompt text"""
        # Remove extra whitespace
        cleaned = re.sub(r"\s+", " ", prompt.strip())

        # Remove redundant commas
        cleaned = re.sub(r",+", ",", cleaned)
        cleaned = re.sub(r"^,|,$", "", cleaned)

        # Remove empty parentheses
        cleaned = re.sub(r"\(\s*\)", "", cleaned)
        cleaned = re.sub(r"\[\s*\]", "", cleaned)

        return cleaned

    def _add_quality_boosters(self, prompt: str) -> str:
        """Add quality improvement terms to positive prompts"""
        if not prompt:
            return prompt

        tokens = [token.strip() for token in prompt.split(",") if token.strip()]

        # Add quality boosters if not already present
        for booster in self.quality_boosters:
            if not any(booster.lower() in token.lower() for token in tokens):
                # Insert after main subject (first token)
                if len(tokens) > 0:
                    tokens.insert(1, booster)
                else:
                    tokens.append(booster)
                break  # Only add one booster to avoid over-enhancement

        return ", ".join(tokens)

    def _add_negative_defaults(self, prompt: str) -> str:
        """Add common negative terms if not present"""
        if not prompt:
            return ", ".join(self.negative_defaults)

        tokens = [token.strip() for token in prompt.split(",") if token.strip()]
        existing_tokens_lower = [token.lower() for token in tokens]

        # Add missing negative defaults
        for negative in self.negative_defaults:
            if negative.lower() not in existing_tokens_lower:
                tokens.append(negative)

        return ", ".join(tokens)

    def _validate_length(self, prompt: str, max_length: int = 500) -> str:
        """Validate and truncate prompt if too long"""
        if len(prompt) <= max_length:
            return prompt

        logger.warning(
            f"Prompt truncated from {len(prompt)} to {max_length} characters"
        )

        # Truncate at word boundary
        truncated = prompt[:max_length]
        last_comma = truncated.rfind(",")

        if last_comma > max_length * 0.8:  # If comma is reasonably close to end
            truncated = truncated[:last_comma]

        return truncated

    def parse_weighted_prompt(self, prompt: str) -> List[Tuple[str, float]]:
        """Parse prompt with attention weights (prompt:weight) format"""
        # Simple implementation - can be enhanced later
        return [(prompt, 1.0)]

    def suggest_improvements(self, prompt: str) -> List[str]:
        """Suggest prompt improvements"""
        suggestions = []

        if len(prompt) < 10:
            suggestions.append("Consider adding more descriptive details")

        if not any(quality in prompt.lower() for quality in self.quality_boosters):
            suggestions.append("Add quality boosters like 'high quality', 'detailed'")

        if prompt.count(",") < 2:
            suggestions.append("Add more descriptive terms separated by commas")

        # Check for common issues
        if "hand" in prompt.lower() and "detailed hands" not in prompt.lower():
            suggestions.append(
                "Consider adding 'detailed hands' for better hand rendering"
            )

        return suggestions

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        total = max(1, self.processing_stats["total_processed"])

        return {
            **self.processing_stats,
            "enhancement_rate": self.processing_stats["enhanced_prompts"] / total,
            "filter_rate": self.processing_stats["filtered_prompts"] / total,
        }

    def reset_stats(self):
        """Reset processing statistics"""
        self.processing_stats = {
            "total_processed": 0,
            "enhanced_prompts": 0,
            "filtered_prompts": 0,
        }
        logger.info("Processing statistics reset")

    # Method 1: Fix _unload_current_model to not use None references
    async def _unload_current_model(self):
        """Safely unload current model and free memory"""
        if self.current_pipeline is not None:
            # Only call if lora_manager exists and is not None
            if hasattr(self, "lora_manager") and self.lora_manager is not None:
                try:
                    self.lora_manager.unload_all_loras(self.current_pipeline)
                except Exception as e:
                    logger.warning(f"Failed to unload LoRAs: {e}")

            # Update model config if available
            if (
                hasattr(self, "model_config_manager")
                and self.model_config_manager is not None
                and self.current_model_id
            ):
                try:
                    self.model_config_manager.set_model_loaded(
                        self.current_model_id, False
                    )
                except Exception as e:
                    logger.warning(f"Failed to update model config: {e}")

            # Clear pipeline references
            self.current_pipeline = None
            self.pipeline = None
            self.loaded_model = None

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Model unloaded and memory cleared")

    # Method 2: Fix _setup_controlnet to not use None references
    async def _setup_controlnet(self, controlnet_config: Dict) -> Dict:
        """Setup ControlNet - SAFE VERSION"""
        controlnet_type = controlnet_config["type"]
        control_image = controlnet_config["image"]
        conditioning_scale = controlnet_config.get("conditioning_scale", 1.0)

        # Check if controlnet_manager is available
        if not hasattr(self, "controlnet_manager") or self.controlnet_manager is None:
            logger.warning(
                "ControlNet manager not available, skipping ControlNet setup"
            )
            return {
                "model": None,
                "params": {},
                "info": {
                    "type": controlnet_type,
                    "conditioning_scale": conditioning_scale,
                    "status": "skipped",
                },
            }

        try:
            # Load ControlNet model
            controlnet_model = self.controlnet_manager.load_controlnet(controlnet_type)

            # Process control image
            if isinstance(control_image, str):
                if control_image.startswith("data:"):
                    control_image = self._decode_base64_image(control_image)
                else:
                    from PIL import Image

                    control_image = Image.open(control_image)

            # Preprocess control image
            processed_control = self.controlnet_manager.preprocess_control_image(
                control_image, controlnet_type
            )

            return {
                "model": controlnet_model,
                "params": {
                    "image": processed_control,
                    "controlnet_conditioning_scale": conditioning_scale,
                },
                "info": {
                    "type": controlnet_type,
                    "conditioning_scale": conditioning_scale,
                    "status": "loaded",
                },
            }

        except Exception as e:
            logger.error(f"ControlNet setup failed: {e}")
            return {
                "model": None,
                "params": {},
                "info": {
                    "type": controlnet_type,
                    "conditioning_scale": conditioning_scale,
                    "status": "failed",
                    "error": str(e),
                },
            }

    # Method 3: Fix _post_process_image to not use None references
    async def _post_process_image(
        self, image: Image.Image, request: Dict, safety_result: Dict
    ) -> Dict:
        """Post-process generated image - SAFE VERSION"""
        try:
            # Safety check on generated image (if available)
            if hasattr(self, "safety_engine") and self.safety_engine is not None:
                try:
                    image_safety = self.safety_engine.check_image(image)

                    if not image_safety["is_safe"]:
                        image = self.safety_engine.apply_safety_filter(image)
                        safety_result.update(image_safety)
                except Exception as e:
                    logger.warning(f"Image safety check failed: {e}")

            # Add watermark (if available)
            if (
                self.config.get("watermark_enabled", True)
                and hasattr(self, "attribution_manager")
                and self.attribution_manager is not None
            ):
                try:
                    attribution_text = self._generate_attribution_text(request)
                    image = self.attribution_manager.add_watermark(
                        image, attribution_text
                    )
                except Exception as e:
                    logger.warning(f"Watermark addition failed: {e}")

            # Save image with metadata
            output_path = await self._save_image_with_metadata(
                image, request, safety_result
            )

            return {
                "image": image,
                "output_path": output_path,
                "safety_result": safety_result,
            }

        except Exception as e:
            logger.error(f"Image post-processing failed: {e}")
            # Return basic result if post-processing fails
            return {
                "image": image,
                "output_path": "generation_failed.png",
                "safety_result": safety_result,
            }

    async def txt2img(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image from text prompt"""
        start_time = time.time()

        try:
            # Ensure model is loaded
            if self.current_pipeline is None:
                await self.initialize()  # type: ignore

            # Process and validate prompt
            processed_prompt = await self._process_prompt(request.get("prompt", ""))
            processed_negative = await self._process_prompt(
                request.get("negative_prompt", ""), is_negative=True
            )

            # Safety check on prompt
            safety_result = self.safety_engine.check_prompt(processed_prompt)  # type: ignore
            if not safety_result["is_safe"]:
                raise ValueError(
                    f"Prompt safety violation: {safety_result['violations']}"
                )

            # Prepare generation parameters
            generation_params = self._prepare_generation_params(
                request, processed_prompt, processed_negative
            )

            # Load LoRAs if specified
            lora_info = []
            if request.get("lora_configs"):
                lora_info = await self._apply_loras(request["lora_configs"])

            # Setup ControlNet if specified
            controlnet_info = None
            if request.get("controlnet_config"):
                controlnet_info = await self._setup_controlnet(
                    request["controlnet_config"]
                )
                generation_params.update(controlnet_info["params"])

            # Generate images
            with torch.autocast(
                self.device.split(":")[0] if ":" in self.device else self.device
            ):
                images = self.current_pipeline(**generation_params).images  # type: ignore

            # Post-process images (safety check, watermarking)
            processed_images = []
            for image in images:
                processed_image = await self._post_process_image(
                    image, request, safety_result
                )
                processed_images.append(processed_image)

            # Prepare response
            generation_time = time.time() - start_time
            response = await self._prepare_response(
                processed_images,
                generation_params,
                generation_time,
                lora_info,
                controlnet_info,  # type: ignore
            )

            # Update stats
            self._update_generation_stats(generation_time)

            # Log generation event
            self.compliance_logger.log_generation(  # type: ignore
                response["metadata"]["output_paths"][0], request, safety_result
            )

            return response

        except Exception as e:
            logger.error(f"txt2img generation failed: {e}")
            raise RuntimeError(f"Generation failed: {e}")

        finally:
            # Cleanup LoRAs after generation
            if request.get("lora_configs"):
                self.lora_manager.unload_all_loras(self.current_pipeline)  # type: ignore

    async def img2img(self, request: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """Generate image from initial image + prompt"""
        # Similar implementation to txt2img but with init_image parameter
        # This would be implemented following the same pattern
        pass

    async def _process_prompt(self, prompt: str, is_negative: bool = False) -> str:
        """Process and enhance prompt"""
        return self.prompt_processor.process(prompt, is_negative=is_negative)  # type: ignore

    def _prepare_generation_params(
        self, request: Dict, prompt: str, negative_prompt: str
    ) -> Dict:
        """Prepare parameters for pipeline generation"""
        params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": request.get("num_inference_steps", 20),
            "guidance_scale": request.get("guidance_scale", 7.5),
            "width": request.get("width", 768),
            "height": request.get("height", 768),
            "num_images_per_prompt": request.get("batch_size", 1),
        }

        # Add seed if specified
        if request.get("seed") is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(request["seed"])
            params["generator"] = generator

        return params

    async def _apply_loras(self, lora_configs: List[Dict]) -> List[Dict]:
        """Apply LoRA configurations to pipeline"""
        lora_info = []

        for config in lora_configs:
            success = self.lora_manager.load_lora(  # type: ignore
                self.current_pipeline, config["lora_id"], config.get("weight", 1.0)
            )

            if success:
                lora_info.append(
                    {
                        "lora_id": config["lora_id"],
                        "weight": config.get("weight", 1.0),
                        "loaded": True,
                    }
                )
            else:
                logger.warning(f"Failed to load LoRA: {config['lora_id']}")

        return lora_info

    def _decode_base64_image(self, base64_string: str) -> Image.Image:
        """Decode base64 image string"""
        header, data = base64_string.split(",", 1)
        image_data = base64.b64decode(data)
        return Image.open(io.BytesIO(image_data))

    def _generate_attribution_text(self, request: Dict) -> str:
        """Generate attribution text for watermark"""
        base_text = "Generated by CharaForge T2I"
        if self.current_model_id:
            model_name = self.current_model_id.split("/")[-1]
            base_text += f" • {model_name}"
        return base_text

    async def _save_image_with_metadata(
        self, image: Image.Image, request: Dict, safety_result: Dict
    ) -> str:
        """Save image with comprehensive metadata"""

        # Generate output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.cache.get_path("OUTPUT_DIR")) / datetime.now().strftime(
            "%Y-%m-%d"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = (
            output_dir / f"{timestamp}_{hash(str(request)) & 0x7FFFFFFF:08x}.png"
        )

        # Create comprehensive metadata
        metadata = self.attribution_manager.create_attribution_metadata(  # type: ignore
            generation_params=request,
            model_info={
                "model_id": self.current_model_id,
                "device": self.device,
                "pipeline_type": type(self.current_pipeline).__name__,
            },
        )

        # Save image with metadata
        image.save(output_path, pnginfo=self._create_png_info(metadata))

        # Save separate metadata JSON
        metadata_path = output_path.with_suffix(".json")

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Image saved: {output_path}")
        return str(output_path)

    def _create_png_info(self, metadata: Dict) -> Any:
        """Create PNG info for metadata embedding"""

        pnginfo = PngInfo()
        for key, value in metadata.items():
            if isinstance(value, (dict, list)):
                import json

                pnginfo.add_text(f"CharaForge_{key}", json.dumps(value))
            else:
                pnginfo.add_text(f"CharaForge_{key}", str(value))

        return pnginfo

    async def _prepare_response(
        self,
        images: List[Dict],
        params: Dict,
        generation_time: float,
        lora_info: List[Dict],
        controlnet_info: Dict,
    ) -> Dict:
        """Prepare API response"""
        # Convert images to base64 for API response
        image_data = []
        output_paths = []

        for img_info in images:
            # Convert to base64
            buffer = io.BytesIO()
            img_info["image"].save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            image_data.append(img_base64)
            output_paths.append(img_info["output_path"])

        return {
            "images": image_data,
            "metadata": {
                "generation_time": round(generation_time, 3),
                "model_used": self.current_model_id,
                "parameters": params,
                "loras_applied": lora_info,
                "controlnet_used": controlnet_info["info"] if controlnet_info else None,
                "output_paths": output_paths,
                "device": self.device,
                "safety_checks": "passed",
            },
        }

    def _update_generation_stats(self, generation_time: float):
        """Update generation statistics"""
        self.generation_stats["total_generations"] += 1  # type: ignore
        self.generation_stats["total_time"] += generation_time  # type: ignore
        self.generation_stats["avg_time_per_image"] = (  # type: ignore
            self.generation_stats["total_time"]  # type: ignore
            / self.generation_stats["total_generations"]  # type: ignore
        )

    # ===== MODEL MANAGEMENT METHODS =====

    async def list_available_models(self) -> List[Dict]:
        """List available diffusion models"""
        return self.model_config_manager.list_available_models()  # type: ignore

    async def get_model_info(self, model_id: str) -> Dict:
        """Get detailed model information"""
        return self.model_config_manager.get_model_info(model_id)  # type: ignore

    async def unload_model(self):
        """Unload current model to free memory"""
        await self._unload_current_model()
        self.current_model_id = None
        logger.info("Model unloaded successfully")

    # ===== LORA MANAGEMENT METHODS =====

    async def list_available_loras(self) -> List[Dict]:
        """List available LoRA adapters - SAFE VERSION"""
        if hasattr(self, "lora_manager") and self.lora_manager is not None:
            try:
                return self.lora_manager.list_available_loras()
            except Exception as e:
                logger.error(f"Failed to list LoRAs: {e}")

        # Return empty list if manager is not available
        return []

    async def get_lora_info(self, lora_id: str) -> Dict:
        """Get detailed LoRA information"""
        return self.lora_manager.get_lora_info(lora_id)  # type: ignore

    # ===== CONTROLNET MANAGEMENT METHODS =====

    async def list_available_controlnets(self) -> List[str]:
        """List available ControlNet types"""
        return list(self.controlnet_manager.available_controlnets.keys())  # type: ignore

    # ===== HEALTH AND STATUS METHODS =====

    def get_status(self) -> Dict:
        """Get engine status and statistics"""
        memory_info = {}
        if torch.cuda.is_available():
            memory_info = {
                "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**2,  # MB
                "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**2,  # MB
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory
                / 1024**2,  # MB
            }

        return {
            "status": "ready" if self.current_pipeline else "initializing",
            "current_model": self.current_model_id,
            "device": self.device,
            "memory_info": memory_info,
            "generation_stats": self.generation_stats.copy(),  # type: ignore
            "loaded_loras": len(self.lora_manager.get_loaded()),  # type: ignore
        }

    async def health_check(self) -> Dict:
        """Comprehensive health check"""
        try:
            # Basic functionality test
            if self.current_pipeline is None:
                await self.initialize()  # type: ignore

            # Quick generation test (if requested)
            # This could be made optional to avoid unnecessary GPU usage

            return {
                "status": "healthy",
                "model_loaded": self.current_model_id is not None,
                "device_available": self.device != "cpu"
                or not torch.cuda.is_available(),
                "last_check": time.time(),
            }

        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "last_check": time.time()}


# ===== ENHANCED LORA MANAGER =====
