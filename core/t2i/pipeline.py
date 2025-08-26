# core/t2i/pipeline.py
import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import torch

from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)
from diffusers.schedulers.scheduling_euler_ancestral_discrete import (
    EulerAncestralDiscreteScheduler,
)

from PIL import Image
import yaml

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from core.shared_cache import get_shared_cache

# Global pipeline cache
_pipelines = {}


def get_t2i_pipeline(model_id: str = "runwayml/stable-diffusion-v1-5"):
    """Get T2I pipeline with caching"""
    if model_id not in _pipelines:
        print(f"Loading T2I pipeline: {model_id}")

        if "xl" in model_id.lower():
            pipeline_cls = StableDiffusionXLPipeline
        else:
            pipeline_cls = StableDiffusionPipeline

        pipeline = pipeline_cls.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            cache_dir=f"{get_shared_cache().cache_root}/models",
        )

        if torch.cuda.is_available():
            pipeline.enable_attention_slicing("auto")
            pipeline.enable_vae_slicing()

        _pipelines[model_id] = pipeline

    return _pipelines[model_id]


def save_image_to_cache(
    image: Image.Image, metadata: dict, prefix: str = "t2i"
) -> tuple:
    """Save image and metadata to cache with timestamped naming"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create dated folder
    date_folder = Path(
        get_shared_cache().get_path("OUTPUT_DIR")
    ) / datetime.now().strftime("%Y-%m-%d")
    date_folder.mkdir(parents=True, exist_ok=True)

    # Generate unique filename
    counter = 1
    while True:
        base_name = f"{prefix}_{timestamp}_{counter:04d}"
        image_path = date_folder / f"{base_name}.png"
        metadata_path = date_folder / f"{base_name}.json"

        if not image_path.exists():
            break
        counter += 1

    # Save image
    image.save(image_path, "PNG")

    # Save metadata
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return str(image_path), str(metadata_path)


class T2IPipeline:
    """Text-to-Image pipeline manager with VRAM optimization"""

    def __init__(self, low_vram: bool = True):
        self.cache = get_shared_cache()
        self.low_vram = low_vram
        self.current_model = None
        self.pipeline = None
        self.style_presets = self._load_style_presets()

    def _load_style_presets(self) -> Dict[str, Any]:
        """Load style presets from configs"""
        presets = {}
        preset_dir = Path("configs/presets")
        if preset_dir.exists():
            for preset_file in preset_dir.glob("*.yaml"):
                with open(preset_file, "r", encoding="utf-8") as f:
                    preset_data = yaml.safe_load(f)
                    presets[preset_file.stem] = preset_data
        return presets

    def load_model(
        self, model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    ) -> bool:
        """Load T2I model with VRAM optimization"""
        try:
            if self.current_model == model_id and self.pipeline is not None:
                print(f"[T2I] Model {model_id} already loaded")
                return True

            print(f"[T2I] Loading model: {model_id}")

            # Determine if SDXL or SD1.5
            is_sdxl = "xl" in model_id.lower()

            # Common kwargs for low VRAM
            common_kwargs = {
                "torch_dtype": torch.float16,
                "use_safetensors": True,
            }

            if self.low_vram:
                common_kwargs.update(
                    {
                        "variant": "fp16",
                        "device_map": "auto",
                    }
                )

            # Load pipeline
            if is_sdxl:
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_id, **common_kwargs
                )
            else:
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    model_id, **common_kwargs
                )

            # Scheduler optimization
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )

            # VRAM optimizations
            if self.low_vram:
                self.pipeline.enable_attention_slicing()
                self.pipeline.enable_vae_slicing()
                if hasattr(self.pipeline, "enable_cpu_offload"):
                    self.pipeline.enable_cpu_offload()
                else:
                    self.pipeline = self.pipeline.to("cuda")
            else:
                self.pipeline = self.pipeline.to("cuda")

            # Enable xformers if available
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                print("[T2I] xformers enabled")
            except Exception as e:
                print(f"[T2I] xformers not available: {e}")

            self.current_model = model_id
            print(f"[T2I] Model {model_id} loaded successfully")
            return True

        except Exception as e:
            print(f"[T2I] Error loading model {model_id}: {e}")
            return False

    def apply_style_preset(
        self, style_id: str, prompt: str, negative: str = ""
    ) -> Tuple[str, str]:
        """Apply style preset to prompts"""
        if style_id not in self.style_presets:
            print(f"[T2I] Style preset {style_id} not found")
            return prompt, negative

        preset = self.style_presets[style_id]

        # Apply prompt template
        if "prompt_template" in preset:
            prompt = preset["prompt_template"].format(prompt=prompt)

        # Apply style suffix
        if "style_suffix" in preset:
            prompt = f"{prompt}, {preset['style_suffix']}"

        # Apply negative prompt
        if "negative_prompt" in preset:
            if negative:
                negative = f"{negative}, {preset['negative_prompt']}"
            else:
                negative = preset["negative_prompt"]

        return prompt, negative

    def generate_filename(self, scene_id: str, image_type: str, seed: int) -> str:
        """Generate standardized filename"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{scene_id}_{image_type}_{seed}_{timestamp}.png"

    def save_metadata(self, image_path: Path, metadata: Dict[str, Any]) -> Path:
        """Save generation metadata as sidecar JSON"""
        metadata_path = image_path.with_suffix(".json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return metadata_path

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 768,
        height: int = 768,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: int = None,  # type: ignore
        style_id: str = None,  # type: ignore
        scene_id: str = "scene_001",
        image_type: str = "bg",
    ) -> Dict[str, Any]:
        """Generate image with metadata"""

        if self.pipeline is None:
            success = self.load_model()
            if not success:
                raise RuntimeError("Failed to load T2I model")

        # Set seed
        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()

        generator = torch.Generator(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        generator.manual_seed(seed)

        # Apply style preset
        if style_id:
            prompt, negative_prompt = self.apply_style_preset(
                style_id, prompt, negative_prompt
            )

        # Generation parameters
        gen_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }

        # Generate image
        print(f"[T2I] Generating: {prompt[:50]}...")
        start_time = datetime.now()

        try:
            result = self.pipeline(**gen_kwargs)  # type: ignore
            image = result.images[0]  # type: ignore

            # Save image
            output_dir = Path(
                self.cache.get_path("OUTPUT_DIR")
            ) / datetime.now().strftime("%Y-%m-%d")
            output_dir.mkdir(parents=True, exist_ok=True)

            filename = self.generate_filename(scene_id, image_type, seed)
            image_path = output_dir / filename
            image.save(image_path, quality=95)

            # Create metadata
            elapsed = (datetime.now() - start_time).total_seconds()
            metadata = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "model": self.current_model,
                "style_id": style_id,
                "scene_id": scene_id,
                "image_type": image_type,
                "width": width,
                "height": height,
                "steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "timestamp": datetime.now().isoformat(),
                "elapsed_seconds": elapsed,
                "filename": filename,
            }

            # Save metadata
            metadata_path = self.save_metadata(image_path, metadata)

            return {
                "success": True,
                "image_path": str(image_path),
                "metadata_path": str(metadata_path),
                "metadata": metadata,
            }

        except Exception as e:
            print(f"[T2I] Generation error: {e}")
            return {"success": False, "error": str(e)}

    def get_available_styles(self) -> Dict[str, str]:
        """Get available style presets"""
        return {k: v.get("description", k) for k, v in self.style_presets.items()}

    def unload_model(self):
        """Unload current model to free VRAM"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            self.current_model = None
            torch.cuda.empty_cache()
            print("[T2I] Model unloaded")


# Global instance
t2i_pipeline = T2IPipeline(low_vram=True)
