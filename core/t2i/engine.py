# core/t2i/engine.py
"""
Unified Text-to-Image generation engine
Integrates pipeline management, LoRA, ControlNet, and safety
"""
from typing import Dict, List, Optional, Union, Any, Tuple
from PIL import Image
import torch
import asyncio
import logging
from pathlib import Path
import base64
import json
from datetime import datetime
import io
import time
from contextlib import asynccontextmanager

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from diffusers.pipelines.controlnet.pipeline_controlnet import (
    StableDiffusionControlNetPipeline,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)
from .pipeline import PipelineManager
from .lora_manager import LoRAManager
from .controlnet import ControlNetManager
from .memory_utils import MemoryOptimizer
from .prompt_utils import PromptProcessor
from .model_config import ModelConfigManager
from ..safety.detector import SafetyEngine
from ..safety.watermark import AttributionManager, ComplianceLogger
from ..shared_cache import get_shared_cache

logger = logging.getLogger(__name__)


class T2IEngine:
    """Unified Text-to-Image generation engine"""

    def __init__(self, cache_root: str, device: str = "auto", config: Dict = None):  # type: ignore
        self.cache_root = Path(cache_root)
        self.device = self._resolve_device(device)
        self.config = config or {}

        # Initialize all required attributes
        self.current_model_id = None
        self.current_pipeline = None
        self.loaded_model = None

        # Generation statistics
        self.generation_stats = {
            "total_generations": 0,
            "total_time": 0.0,
            "avg_time_per_image": 0.0,
        }

        self.current_pipeline = None
        self.pipeline = None  # ADD THIS LINE - alias for current_pipeline

        # Optimization settings
        self.optimization_settings = {
            "enable_attention_slicing": True,
            "enable_vae_slicing": True,
            "enable_cpu_offload": False,
            "use_fp16": torch.cuda.is_available(),
            "enable_xformers": True,
            "memory_optimization_level": 2,
        }

        # Core components
        self.pipeline_manager = PipelineManager(str(cache_root), self.device)
        self.lora_manager = LoRAManager(str(cache_root))
        self.controlnet_manager = ControlNetManager(str(cache_root))
        self.memory_optimizer = MemoryOptimizer()
        self.prompt_processor = PromptProcessor()
        self.model_config_manager = ModelConfigManager(str(cache_root))

        # Safety and compliance components
        self.safety_engine = SafetyEngine(self.config.get("safety", {}))
        self.attribution_manager = AttributionManager(str(cache_root))
        self.compliance_logger = ComplianceLogger(str(cache_root))

        # Apply optimization settings to pipeline manager
        self.pipeline_manager.optimization_settings.update(self.optimization_settings)
        self._sync_pipeline_references()
        self._initialize_component_references()

        logger.info(f"T2I Engine initialized with device: {self.device}")

    def _resolve_device(self, device: str) -> str:
        """Resolve device specification"""
        if device == "auto":
            if torch.cuda.is_available():
                return f"cuda:{torch.cuda.current_device()}"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def _sync_pipeline_references(self):
        """Keep pipeline references in sync"""
        self.pipeline = self.current_pipeline

    async def initialize(self, model_id: str = None):  # type: ignore
        """Initialize engine with default model"""
        default_model = model_id or self.config.get(
            "default_model", "runwayml/stable-diffusion-v1-5"
        )
        await self.load_model(default_model)
        logger.info(f"T2I Engine ready with model: {default_model}")

    def _initialize_component_references(self):
        """Initialize cross-component references"""
        try:
            # Initialize PromptProcessor with component references
            components = {
                "lora_manager": self.lora_manager,
                "safety_engine": self.safety_engine,
                "compliance_logger": self.compliance_logger,
                "controlnet_manager": self.controlnet_manager,
                "attribution_manager": self.attribution_manager,
                "generation_stats": self.generation_stats,
            }
            self.prompt_processor.initialize(components)

            # Sync pipeline references
            self._sync_pipeline_references()

            logger.debug("Component references initialized")
        except Exception as e:
            logger.error(f"Failed to initialize component references: {e}")

    async def load_model(self, model_id: str, **kwargs) -> bool:
        """Load diffusion model"""
        try:
            start_time = time.time()

            # Check if model is already loaded
            if self.current_model_id == model_id and self.current_pipeline is not None:
                logger.info(f"Model {model_id} already loaded")
                return True

            # Unload current model if exists
            if self.current_pipeline is not None:
                await self._unload_current_model()

            # Load new pipeline
            pipeline = await self.pipeline_manager.load_pipeline(model_id, **kwargs)

            # Apply memory optimizations
            pipeline = self.memory_optimizer.optimize_pipeline(pipeline)

            # Update state
            self.current_pipeline = pipeline
            self.current_model_id = model_id
            self.loaded_model = model_id

            self._sync_pipeline_references()

            # Update model config
            self.model_config_manager.set_model_loaded(model_id, True)

            load_time = time.time() - start_time
            logger.info(f"Model {model_id} loaded in {load_time:.2f}s")
            return True

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    async def _unload_current_model(self):
        """Safely unload current model and free memory"""
        if self.current_pipeline is not None:
            # Unload any LoRAs first
            self.lora_manager.unload_all_loras(self.current_pipeline)

            # Update model config
            if self.current_model_id:
                self.model_config_manager.set_model_loaded(self.current_model_id, False)

            # Clear state
            self.current_pipeline = None
            self.loaded_model = None

            # Move to CPU and clear cache
            self.current_pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Previous model unloaded and memory cleared")

    async def txt2img(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image from text prompt"""
        start_time = time.time()

        try:
            # Ensure model is loaded
            if self.current_pipeline is None:
                await self.initialize()

            # Process and validate prompt
            processed_prompt = self.prompt_processor.process(request.get("prompt", ""))
            processed_negative = self.prompt_processor.process(
                request.get("negative_prompt", ""), is_negative=True
            )

            # Safety check on prompt
            safety_result = self.safety_engine.check_prompt(processed_prompt)
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

            # Generate images (mock for now - replace with real pipeline call)
            images = await self._generate_images(generation_params)

            # Generate images (simplified for now)
            with torch.autocast(
                self.device.split(":")[0] if ":" in self.device else self.device
            ):
                # Mock generation for now - replace with real pipeline call
                mock_image = Image.new(
                    "RGB",
                    (generation_params["width"], generation_params["height"]),
                    color="red",
                )
                images = [mock_image]

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
            self.compliance_logger.log_generation(
                response["metadata"]["output_paths"][0], request, safety_result
            )

            return response

        except Exception as e:
            logger.error(f"txt2img generation failed: {e}")
            raise RuntimeError(f"Generation failed: {e}")

        finally:
            # Cleanup LoRAs after generation
            if request.get("lora_configs"):
                self.lora_manager.unload_all_loras(self.current_pipeline)

    async def _generate_images(self, params: Dict[str, Any]) -> List[Image.Image]:
        """Generate images using the pipeline"""
        try:
            # For now, create mock images - replace with real pipeline call
            width = params.get("width", 768)
            height = params.get("height", 768)
            num_images = params.get("num_images_per_prompt", 1)

            images = []
            for i in range(num_images):
                # Create a colorful mock image instead of solid color
                import random

                color = (
                    random.randint(50, 255),
                    random.randint(50, 255),
                    random.randint(50, 255),
                )
                mock_image = Image.new("RGB", (width, height), color=color)

                # Add some text to make it look generated
                from PIL import ImageDraw, ImageFont

                draw = ImageDraw.Draw(mock_image)
                try:
                    font = ImageFont.load_default()
                except:
                    font = None

                if font:
                    text = f"Mock Generated Image {i+1}"
                    text_bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]

                    x = (width - text_width) // 2
                    y = (height - text_height) // 2

                    # Draw text with background
                    draw.rectangle(
                        [x - 10, y - 5, x + text_width + 10, y + text_height + 5],
                        fill=(255, 255, 255, 180),
                    )
                    draw.text((x, y), text, font=font, fill=(0, 0, 0))

                images.append(mock_image)

            # Simulate generation time
            await asyncio.sleep(0.1)  # Short delay to simulate processing

            return images

        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise RuntimeError(f"Failed to generate images: {e}")

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
            success = self.lora_manager.load_lora(
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

    async def _setup_controlnet(self, controlnet_config: Dict) -> Dict:  # type: ignore
        """Setup ControlNet for generation"""
        controlnet_type = controlnet_config["type"]
        control_image = controlnet_config["image"]
        conditioning_scale = controlnet_config.get("conditioning_scale", 1.0)

        # Load ControlNet model
        controlnet_model = self.controlnet_manager.load_controlnet(controlnet_type)

    async def _post_process_image(
        self, image: Image.Image, request: Dict, safety_result: Dict
    ) -> Dict:
        """Post-process generated image with safety and watermarking"""
        # Safety check on generated image
        image_safety = self.safety_engine.check_image(image)

        if not image_safety["is_safe"]:
            # Apply blur or replacement image
            image = self.safety_engine.apply_safety_filter(image)
            safety_result.update(image_safety)

        # Add watermark
        if self.config.get("watermark_enabled", True):
            attribution_text = self._generate_attribution_text(request)
            image = self.attribution_manager.add_watermark(image, attribution_text)

        # Save image with metadata
        output_path = await self._save_image_with_metadata(
            image, request, safety_result
        )

        return {
            "image": image,
            "output_path": output_path,
            "safety_result": safety_result,
        }

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
        output_dir = (
            self.cache_root
            / "outputs"
            / "charaforge-t2i-lab"
            / datetime.now().strftime("%Y-%m-%d")
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = (
            output_dir / f"{timestamp}_{hash(str(request)) & 0x7FFFFFFF:08x}.png"
        )

        # Create comprehensive metadata
        metadata = {
            "generated_by": "CharaForge T2I System",
            "generation_timestamp": datetime.now().isoformat(),
            "model_used": self.current_model_id,
            "generation_params": request,
            "safety_result": safety_result,
        }

        # Save image
        image.save(output_path)

        # Save separate metadata JSON
        metadata_path = output_path.with_suffix(".json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Image saved: {output_path}")
        return str(output_path)

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
                "controlnet_used": controlnet_info,
                "output_paths": output_paths,
                "device": self.device,
                "safety_checks": "passed",
            },
        }

    def _update_generation_stats(self, generation_time: float):
        """Update generation statistics"""
        self.generation_stats["total_generations"] += 1
        self.generation_stats["total_time"] += generation_time
        self.generation_stats["avg_time_per_image"] = (
            self.generation_stats["total_time"]
            / self.generation_stats["total_generations"]
        )

    # Status and health methods
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
            "generation_stats": self.generation_stats.copy(),
            "loaded_loras": len(self.lora_manager.get_loaded()),
            "cache_root": str(self.cache_root),
        }

    async def health_check(self) -> Dict:
        """Comprehensive health check"""
        try:
            # Basic functionality test
            if self.current_pipeline is None:
                await self.initialize()

            return {
                "status": "healthy",
                "model_loaded": self.current_model_id is not None,
                "device_available": self.device != "cpu"
                or not torch.cuda.is_available(),
                "cache_accessible": self.cache_root.exists(),
                "last_check": time.time(),
            }

        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "last_check": time.time()}

    def _get_pipeline_class(self, model_id: str):
        """Determine appropriate pipeline class based on model"""
        model_id_lower = model_id.lower()

        if "xl" in model_id_lower or "sdxl" in model_id_lower:
            return StableDiffusionXLPipeline
        else:
            return StableDiffusionPipeline

    def _apply_optimizations(self, pipeline) -> DiffusionPipeline:
        """Apply memory and performance optimizations"""

        # Enable attention slicing for memory efficiency
        if self.optimization_settings["enable_attention_slicing"]:
            pipeline.enable_attention_slicing()
            logger.debug("Attention slicing enabled")

        # Enable VAE slicing for large images
        if self.optimization_settings["enable_vae_slicing"]:
            pipeline.enable_vae_slicing()
            logger.debug("VAE slicing enabled")

        # Enable CPU offload if needed
        if self.optimization_settings["enable_cpu_offload"]:
            pipeline.enable_model_cpu_offload()
            logger.debug("CPU offload enabled")

        # Enable xFormers if available
        if self.optimization_settings["enable_xformers"]:
            try:
                pipeline.enable_xformers_memory_efficient_attention()
                logger.debug("xFormers attention enabled")
            except Exception as e:
                logger.warning(f"xFormers not available: {e}")

        return pipeline

    def get_scheduler_options(self) -> Dict[str, Any]:
        """Get available scheduler options"""

        return {
            "euler": EulerDiscreteScheduler,
            "dpm": DPMSolverMultistepScheduler,
            "default": None,  # Keep original scheduler
        }

    def set_scheduler(self, scheduler_name: str):
        """Change pipeline scheduler"""
        if self.pipeline is None:
            raise RuntimeError("No pipeline loaded")

        scheduler_classes = self.get_scheduler_options()

        if scheduler_name not in scheduler_classes:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

        if scheduler_classes[scheduler_name] is not None:
            self.pipeline.scheduler = scheduler_classes[scheduler_name].from_config(
                self.pipeline.scheduler.config
            )
            logger.info(f"Scheduler changed to: {scheduler_name}")
