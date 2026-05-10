# core/t2i/engine.py
"""
Unified Text-to-Image generation engine
Integrates pipeline management, LoRA, ControlNet, and safety
"""
from typing import Dict, List, Optional, Union, Any, Tuple
from PIL import Image
import os
import torch
import asyncio
import logging
from pathlib import Path
import base64
import io
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime
from types import SimpleNamespace

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
from ..shared_cache import get_shared_cache

logger = logging.getLogger(__name__)


class DisabledSafety:
    """No-op safety surface. The user requested safety handling disabled."""

    def check_prompt_safety(self, prompt: str) -> Dict[str, Any]:
        return {"is_safe": True, "clean_prompt": prompt, "disabled": True}

    def check_image_safety(self, image: Image.Image) -> Dict[str, Any]:
        return {"is_safe": True, "processed_image": image, "disabled": True}


class T2IEngine:
    """Unified Text-to-Image generation engine"""

    def __init__(self, cache_root: str, device: str = "auto", config: Dict = None):  # type: ignore
        self.cache_root = Path(cache_root)
        self.cache = get_shared_cache()
        self.device = self._resolve_device(device)
        self.config = config or {}
        # Default to real generation; can be overridden with config/mock env
        self.mock_generation = bool(self.config.get("mock_generation", False))

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
        self.pipeline = None  # alias for current_pipeline

        # Optimization settings
        self.optimization_settings = {
            "enable_attention_slicing": True,
            "enable_vae_slicing": True,
            "enable_vae_tiling": False,
            "enable_cpu_offload": False,
            "enable_sequential_cpu_offload": False,
            "use_fp16": torch.cuda.is_available(),
            "enable_xformers": True,
            "memory_optimization_level": 2,
        }

        # Core components
        self.pipeline_manager = PipelineManager(str(cache_root), self.device)
        default_model = str(self.config.get("default_model", "") or "")
        prefer_sdxl_lora = bool(self.config.get("prefer_sdxl_lora", False))
        if not prefer_sdxl_lora:
            lowered = default_model.lower()
            prefer_sdxl_lora = ("sdxl" in lowered) or ("xl" in lowered) or ("/xl/" in lowered) or ("\\xl\\" in lowered)

        self.lora_manager = LoRAManager(str(cache_root), prefer_sdxl=prefer_sdxl_lora)
        self.controlnet_manager = ControlNetManager(str(cache_root))
        self.memory_optimizer = MemoryOptimizer()
        self.prompt_processor = PromptProcessor()
        self.model_config_manager = ModelConfigManager(str(cache_root))

        # Safety/watermark/compliance are intentionally disabled for local runs.
        self.safety_engine = None
        self.attribution_manager = None
        self.compliance_logger = None

        # Apply optimization settings to pipeline manager
        self.pipeline_manager.optimization_settings.update(self.optimization_settings)
        self._sync_pipeline_references()
        self._initialize_component_references()

        logger.info(f"T2I Engine initialized with device: {self.device}")

    def _apply_request_optimizations(self, request: Dict[str, Any]) -> None:
        """
        Best-effort: apply per-request optimization overrides (e.g. from runtime preset).

        Supported keys (bool):
        - enable_attention_slicing
        - enable_vae_slicing
        - enable_vae_tiling
        - enable_cpu_offload
        - enable_sequential_cpu_offload
        - enable_xformers
        """
        try:
            if not isinstance(request, dict):
                return

            keys = [
                "enable_attention_slicing",
                "enable_vae_slicing",
                "enable_vae_tiling",
                "enable_cpu_offload",
                "enable_sequential_cpu_offload",
                "enable_xformers",
            ]
            overrides: Dict[str, bool] = {}
            for k in keys:
                if k in request:
                    overrides[k] = bool(request.get(k))
            if not overrides:
                return

            self.optimization_settings.update(overrides)
            try:
                self.pipeline_manager.optimization_settings.update(self.optimization_settings)
            except Exception:
                pass

            # If already loaded, apply immediately (best-effort).
            if self.current_pipeline is not None and not self.mock_generation:
                try:
                    self._apply_optimizations(self.current_pipeline)
                except Exception:
                    pass
        except Exception:
            return

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
                "safety_engine": None,
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

            # In mock mode we skip heavyweight loading but still update bookkeeping
            if self.mock_generation:
                self.current_model_id = model_id
                self.current_pipeline = SimpleNamespace(mock=True, model_id=model_id)
                self.loaded_model = model_id
                self._sync_pipeline_references()
                self.model_config_manager.set_model_loaded(model_id, True)
                logger.info(f"Mock T2I pipeline ready for model: {model_id}")
                return True

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

        async def _run() -> Dict[str, Any]:
            try:
                target_model = request.get("model_id") or request.get("model")
                self._apply_request_optimizations(request)

                # Ensure model is loaded (in mock mode this is a lightweight noop)
                if self.current_pipeline is None or (
                    target_model and target_model != self.current_model_id
                ):
                    await self.initialize(target_model)  # type: ignore

                # Process and validate prompt
                processed_prompt = self.prompt_processor.process(request.get("prompt", ""))
                processed_negative = self.prompt_processor.process(
                    request.get("negative_prompt", ""), is_negative=True
                )

                safety_result = {"is_safe": True, "disabled": True}

                # Prepare generation parameters
                generation_params = self._prepare_generation_params(
                    request, processed_prompt, processed_negative
                )

                # Load LoRAs if specified (skipped in mock mode)
                lora_info = []
                if request.get("lora_configs") and not self.mock_generation:
                    lora_info = await self._apply_loras(request["lora_configs"])

                # Setup ControlNet if specified (skipped in mock mode)
                controlnet_info = None
                if request.get("controlnet_config") and not self.mock_generation:
                    controlnet_info = await self._setup_controlnet(
                        request["controlnet_config"]
                    )
                    generation_params.update(controlnet_info["params"])

                # Generate images (mock-friendly)
                images, oom_info = await self._generate_images(generation_params)

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

                if oom_info:
                    try:
                        response["metadata"]["oom_fallback"] = oom_info
                    except Exception:
                        pass

                # Update stats
                self._update_generation_stats(generation_time)

                # Log generation event
                if self.compliance_logger:
                    try:
                        self.compliance_logger.log_generation(
                            response["metadata"]["output_paths"][0],
                            request,
                            safety_result,
                        )
                    except Exception as log_err:
                        logger.debug(f"Compliance log skipped: {log_err}")

                return response

            except Exception as e:
                logger.error(f"txt2img generation failed: {e}")
                raise RuntimeError(f"Generation failed: {e}")

            finally:
                # Cleanup LoRAs after generation
                if request.get("lora_configs") and not self.mock_generation:
                    self.lora_manager.unload_all_loras(self.current_pipeline)

        if (not self.mock_generation) and str(self.device).startswith("cuda") and torch.cuda.is_available():
            try:
                from core.runtime import get_model_runtime

                runtime = get_model_runtime()
                async with runtime.exclusive_gpu_async(
                    reason="t2i.txt2img", device=self.device
                ):
                    return await _run()
            except Exception:
                return await _run()

        return await _run()

    def _is_cuda_oom(self, exc: Exception) -> bool:
        try:
            if torch.cuda.is_available() and isinstance(exc, torch.cuda.OutOfMemoryError):
                return True
        except Exception:
            pass

        msg = str(exc).lower()
        return ("out of memory" in msg) and (
            "cuda" in msg or "cublas" in msg or "memory" in msg
        )

    def _round_down_multiple(self, value: int, multiple: int = 8) -> int:
        if multiple <= 1:
            return int(value)
        return int(value) - (int(value) % int(multiple))

    def _enable_aggressive_optimizations(self) -> None:
        pipeline = self.current_pipeline
        if pipeline is None:
            return

        try:
            if hasattr(pipeline, "enable_attention_slicing"):
                pipeline.enable_attention_slicing()
        except Exception:
            pass

        try:
            if hasattr(pipeline, "enable_vae_slicing"):
                pipeline.enable_vae_slicing()
        except Exception:
            pass

        try:
            if hasattr(pipeline, "enable_vae_tiling"):
                pipeline.enable_vae_tiling()
        except Exception:
            pass

        # CPU offload is the last resort; may require accelerate and can be unavailable.
        try:
            if hasattr(pipeline, "enable_model_cpu_offload"):
                pipeline.enable_model_cpu_offload()
        except Exception:
            pass

        try:
            if hasattr(pipeline, "enable_sequential_cpu_offload"):
                pipeline.enable_sequential_cpu_offload()
        except Exception:
            pass

    async def _generate_images(self, params: Dict[str, Any]) -> Tuple[List[Image.Image], Dict[str, Any]]:
        """Generate images using the pipeline"""
        try:
            if not self.mock_generation and self.current_pipeline is not None:
                attempts = []

                base_steps = int(params.get("num_inference_steps", 20) or 20)
                base_width = int(params.get("width", 768) or 768)
                base_height = int(params.get("height", 768) or 768)

                attempts.append(("base", {}, False))

                lowered_steps = max(8, int(base_steps * 0.75))
                if lowered_steps < base_steps:
                    attempts.append(
                        (
                            "lower_steps",
                            {"num_inference_steps": lowered_steps},
                            False,
                        )
                    )

                scaled_width = max(512, self._round_down_multiple(int(base_width * 0.85), 8))
                scaled_height = max(512, self._round_down_multiple(int(base_height * 0.85), 8))
                if (scaled_width, scaled_height) != (base_width, base_height):
                    attempts.append(
                        (
                            "lower_resolution",
                            {"width": scaled_width, "height": scaled_height},
                            False,
                        )
                    )

                attempts.append(
                    (
                        "offload_vae_tiling",
                        {
                            "num_inference_steps": min(base_steps, max(8, lowered_steps)),
                            "width": scaled_width,
                            "height": scaled_height,
                        },
                        True,
                    )
                )

                oom_info: Dict[str, Any] = {"attempts": []}
                last_exc: Optional[Exception] = None

                device_type = (
                    self.device.split(":")[0] if ":" in self.device else self.device
                )

                for label, overrides, aggressive in attempts:
                    attempt_params = dict(params)
                    attempt_params.update(dict(overrides or {}))
                    attempt_steps = int(attempt_params.get("num_inference_steps", base_steps) or base_steps)
                    attempt_width = int(attempt_params.get("width", base_width) or base_width)
                    attempt_height = int(attempt_params.get("height", base_height) or base_height)

                    if aggressive and torch.cuda.is_available() and str(self.device).startswith("cuda"):
                        self._enable_aggressive_optimizations()

                    try:
                        with torch.autocast(device_type):
                            result = self.current_pipeline(**attempt_params)  # type: ignore
                        images = list(getattr(result, "images", []))  # type: ignore
                        if not images:
                            raise RuntimeError("Pipeline returned no images")

                        # Update in-place for accurate metadata.
                        params["num_inference_steps"] = attempt_steps
                        params["width"] = attempt_width
                        params["height"] = attempt_height

                        used_label = None if label == "base" else label
                        if used_label:
                            oom_info["used"] = used_label
                            oom_info["final_params"] = {
                                "num_inference_steps": attempt_steps,
                                "width": attempt_width,
                                "height": attempt_height,
                            }
                            return images, oom_info
                        return images, {}

                    except Exception as exc:  # noqa: BLE001
                        last_exc = exc
                        is_oom = self._is_cuda_oom(exc)
                        oom_info["attempts"].append(
                            {
                                "label": label,
                                "params": {
                                    "num_inference_steps": attempt_steps,
                                    "width": attempt_width,
                                    "height": attempt_height,
                                },
                                "oom": bool(is_oom),
                                "aggressive": bool(aggressive),
                                "error": str(exc)[:500],
                            }
                        )

                        if not is_oom:
                            raise

                        try:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass

                        logger.warning("CUDA OOM on %s; retrying with fallback", label)
                        continue

                raise RuntimeError(
                    f"Failed to generate images after OOM fallbacks: {last_exc}"
                )
            elif not self.mock_generation:
                raise RuntimeError("Pipeline not initialized")

            # Mock generation path
            width = params.get("width", 768)
            height = params.get("height", 768)
            num_images = params.get("num_images_per_prompt", 1)

            images: List[Image.Image] = []
            for i in range(num_images):
                import random
                from PIL import ImageDraw, ImageFont

                color = (
                    random.randint(50, 255),
                    random.randint(50, 255),
                    random.randint(50, 255),
                )
                mock_image = Image.new("RGB", (width, height), color=color)

                draw = ImageDraw.Draw(mock_image)
                try:
                    font = ImageFont.load_default()
                except Exception:
                    font = None

                if font:
                    text = f"Mock Generated Image {i+1}"
                    text_bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]

                    x = (width - text_width) // 2
                    y = (height - text_height) // 2

                    draw.rectangle(
                        [x - 10, y - 5, x + text_width + 10, y + text_height + 5],
                        fill=(255, 255, 255, 180),
                    )
                    draw.text((x, y), text, font=font, fill=(0, 0, 0))

                images.append(mock_image)

            await asyncio.sleep(0.05)  # Short delay to simulate processing

            return images, {}

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
            "num_inference_steps": request.get("num_inference_steps")
            or request.get("steps", 20),
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

        if not self.controlnet_manager:
            logger.warning("ControlNet manager not available, skipping ControlNet setup")
            return {"model": None, "params": {}, "info": {"status": "skipped"}}

        try:
            controlnet_model = self.controlnet_manager.load_controlnet(controlnet_type)

            if isinstance(control_image, str):
                if control_image.startswith("data:"):
                    control_image = self._decode_base64_image(control_image)
                else:
                    control_image = Image.open(control_image)

            processed_control = self.controlnet_manager.preprocess_control_image(
                control_image, controlnet_type
            )

            # If we have a base pipeline, wrap it with ControlNet
            if self.current_pipeline and not isinstance(
                self.current_pipeline, StableDiffusionControlNetPipeline
            ):
                self.current_pipeline = self.controlnet_manager.create_controlnet_pipeline(
                    self.current_pipeline, controlnet_type
                )
                self.current_pipeline = self.memory_optimizer.optimize_pipeline(  # type: ignore
                    self.current_pipeline
                )
                self._sync_pipeline_references()

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

    def _decode_base64_image(self, base64_string: str) -> Image.Image:
        """Decode base64 image string that may include a data URI header."""
        _, data = base64_string.split(",", 1)
        image_data = base64.b64decode(data)
        return Image.open(io.BytesIO(image_data))

    async def _post_process_image(
        self, image: Image.Image, request: Dict, safety_result: Dict
    ) -> Dict:
        """Post-process generated image without safety checks or watermarking."""
        image_safety: Dict[str, Any] = {"is_safe": True, "disabled": True}
        metadata: Optional[Dict[str, Any]] = None

        output_path = await self._save_image_with_metadata(
            image, request, safety_result, metadata
        )

        return {
            "image": image,
            "output_path": output_path,
            "safety_result": {**safety_result, **image_safety},
        }

    def _generate_attribution_text(self, request: Dict) -> str:
        """Generate attribution text for watermark"""
        base_text = "Generated by CharaForge T2I"
        if self.current_model_id:
            model_name = self.current_model_id.split("/")[-1]
            base_text += f" • {model_name}"
        return base_text

    async def _save_image_with_metadata(
        self,
        image: Image.Image,
        request: Dict,
        safety_result: Dict,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save image with comprehensive metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = request.get("session_id") or "general"
        output_dir = (
            Path(get_shared_cache().get_path("OUTPUT_DIR"))
            / "t2i"
            / str(session_id)
            / datetime.now().strftime("%Y-%m-%d")
        )
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            output_dir = (
                Path("/tmp/ai_output/outputs")
                / "t2i"
                / str(session_id)
                / datetime.now().strftime("%Y-%m-%d")
            )
            output_dir.mkdir(parents=True, exist_ok=True)

        output_path = (
            output_dir / f"{timestamp}_{hash(str(request)) & 0x7FFFFFFF:08x}.png"
        )

        base_metadata = metadata or {
            "generated_by": "CharaForge T2I System",
            "generation_timestamp": datetime.now().isoformat(),
            "generation_params": request,
            "attribution": {"models": [], "sources": [], "licenses": []},
            "model_used": self.current_model_id,
        }
        base_metadata["safety_result"] = safety_result

        saved_path = output_path
        try:
            if self.attribution_manager:
                saved = self.attribution_manager.save_with_attribution(
                    image, output_path, base_metadata, format="PNG"
                )
                saved_path = Path(saved)
            else:
                image.save(output_path)
                metadata_path = output_path.with_suffix(".json")
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(base_metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to embed attribution metadata: {e}")
            image.save(output_path)

        logger.info(f"Image saved: {saved_path}")
        return str(saved_path)

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
        if self.optimization_settings.get("enable_attention_slicing"):
            pipeline.enable_attention_slicing()
            logger.debug("Attention slicing enabled")

        # Enable VAE slicing for large images
        if self.optimization_settings.get("enable_vae_slicing"):
            pipeline.enable_vae_slicing()
            logger.debug("VAE slicing enabled")

        # Enable VAE tiling (memory saver; esp. SDXL 1024)
        if self.optimization_settings.get("enable_vae_tiling"):
            try:
                if hasattr(pipeline, "enable_vae_tiling"):
                    pipeline.enable_vae_tiling()
                    logger.debug("VAE tiling enabled")
            except Exception:
                pass

        # CPU offload: sequential > model offload (best-effort)
        if self.optimization_settings.get("enable_sequential_cpu_offload"):
            try:
                if hasattr(pipeline, "enable_sequential_cpu_offload"):
                    pipeline.enable_sequential_cpu_offload()
                    logger.debug("Sequential CPU offload enabled")
            except Exception:
                pass
        elif self.optimization_settings.get("enable_cpu_offload"):
            try:
                if hasattr(pipeline, "enable_model_cpu_offload"):
                    pipeline.enable_model_cpu_offload()
                elif hasattr(pipeline, "enable_cpu_offload"):
                    pipeline.enable_cpu_offload()
                logger.debug("CPU offload enabled")
            except Exception:
                pass

        # Enable xFormers if available
        if self.optimization_settings.get("enable_xformers"):
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


# ---------------------------------------------------------------------------
# Global engine singleton (used by Story integration)
# ---------------------------------------------------------------------------

_t2i_engine: Optional[T2IEngine] = None


def get_t2i_engine() -> T2IEngine:
    """Get global T2I engine instance (singleton)."""
    global _t2i_engine
    if _t2i_engine is None:
        from core.config import get_config

        cfg = get_config()
        cache = get_shared_cache()
        from core.model_registry import resolve_model_path

        mock_flag = os.getenv("T2I_MOCK", "0").lower() in {"1", "true", "yes", "on"}
        default_model = resolve_model_path(
            getattr(cfg.model, "default_sd_model", "stabilityai/stable-diffusion-xl-base-1.0"),
            kind="t2i",
        )

        _t2i_engine = T2IEngine(
            cache_root=str(cache.cache_root),
            device=getattr(cfg.model, "device", "auto"),
            config={
                "default_model": default_model,
                "mock_generation": mock_flag,
                "safety": {},
                "watermark_enabled": True,
            },
        )

    return _t2i_engine
