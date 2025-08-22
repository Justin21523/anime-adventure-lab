"""
Batch generation script for LoRA evaluation and testing
Generates comparison grids (before/after LoRA) with fixed seeds
"""

import os
import argparse
import json
from pathlib import Path
from typing import List, Dict
import yaml

import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from peft import PeftModel

# Shared cache bootstrap
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.shared_cache import bootstrap_cache

# setup cache
cache = bootstrap_cache()


class LoRABatchGenerator:
    """Batch generator for LoRA evaluation"""

    def __init__(self, base_model: str, device: str = "auto"):
        self.base_model = base_model
        self.device = (
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load base pipeline
        print(f"[Generator] Loading base model: {base_model}")
        if "xl" in base_model.lower():
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                base_model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True,
            )
        else:
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                base_model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True,
            )

        self.pipeline.to(self.device)
        self.pipeline.set_progress_bar_config(disable=True)

        # Enable memory efficient attention
        if hasattr(self.pipeline, "enable_attention_slicing"):
            self.pipeline.enable_attention_slicing()
        if hasattr(self.pipeline, "enable_xformers_memory_efficient_attention"):
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
            except:
                pass

        self.current_lora = None
        print(f"[Generator] Ready on {self.device}")

    def load_lora(self, lora_path: str, scale: float = 0.75):
        """Load LoRA adapter"""
        print(f"[Generator] Loading LoRA: {lora_path} (scale={scale})")

        # Unload current LoRA if any
        if self.current_lora:
            self.unload_lora()

        # Load new LoRA
        self.pipeline.unet = PeftModel.from_pretrained(
            self.pipeline.unet, lora_path, adapter_name="lora"
        )

        # Set scale
        self.pipeline.unet.set_adapter_scale("lora", scale)
        self.current_lora = lora_path

    def unload_lora(self):
        """Unload current LoRA"""
        if self.current_lora:
            print("[Generator] Unloading LoRA")
            self.pipeline.unet.disable_adapters()
            self.current_lora = None

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 768,
        height: int = 768,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: int = None,  # type: ignore
    ) -> Image.Image:
        """Generate single image"""
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        return result.images[0]

    def generate_comparison_grid(
        self,
        prompts: List[str],
        lora_path: str,
        output_dir: Path,
        lora_scale: float = 0.75,
        negative_prompt: str = "lowres, blurry, extra fingers, bad anatomy",
        seeds: List[int] = None,  # type: ignore
        width: int = 768,
        height: int = 768,
    ):
        """Generate before/after comparison grid"""
        if seeds is None:
            seeds = [42, 123, 456]  # Default seeds

        output_dir.mkdir(parents=True, exist_ok=True)

        # Grid dimensions
        cols = len(seeds)
        rows = len(prompts) * 2  # Before and after for each prompt

        grid_width = cols * width
        grid_height = rows * height
        grid_image = Image.new("RGB", (grid_width, grid_height), "white")

        results = []

        for row_idx, prompt in enumerate(prompts):
            print(
                f"[Generator] Processing prompt {row_idx + 1}/{len(prompts)}: {prompt[:50]}..."
            )

            for col_idx, seed in enumerate(seeds):
                # Generate without LoRA (before)
                self.unload_lora()
                before_image = self.generate_image(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    seed=seed,
                )

                # Generate with LoRA (after)
                self.load_lora(lora_path, lora_scale)
                after_image = self.generate_image(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    seed=seed,
                )

                # Place in grid
                before_y = row_idx * 2 * height
                after_y = (row_idx * 2 + 1) * height
                x = col_idx * width

                grid_image.paste(before_image, (x, before_y))
                grid_image.paste(after_image, (x, after_y))

                # Save individual images
                before_image.save(
                    output_dir / f"before_prompt{row_idx:02d}_seed{seed}.png"
                )
                after_image.save(
                    output_dir / f"after_prompt{row_idx:02d}_seed{seed}.png"
                )

                results.append(
                    {
                        "prompt": prompt,
                        "seed": seed,
                        "before_path": f"before_prompt{row_idx:02d}_seed{seed}.png",
                        "after_path": f"after_prompt{row_idx:02d}_seed{seed}.png",
                    }
                )

        # Add labels to grid
        draw = ImageDraw.Draw(grid_image)
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()

        # Add "Before" and "After" labels
        for row_idx in range(len(prompts)):
            before_y = row_idx * 2 * height + 10
            after_y = (row_idx * 2 + 1) * height + 10

            draw.text((10, before_y), "BEFORE", fill="red", font=font)
            draw.text((10, after_y), "AFTER", fill="green", font=font)

        # Add seed labels
        for col_idx, seed in enumerate(seeds):
            x = col_idx * width + width // 2 - 30
            draw.text((x, 10), f"Seed {seed}", fill="blue", font=font)

        # Save grid
        grid_path = output_dir / "comparison_grid.png"
        grid_image.save(grid_path)

        # Save metadata
        metadata = {
            "lora_path": str(lora_path),
            "lora_scale": lora_scale,
            "prompts": prompts,
            "seeds": seeds,
            "negative_prompt": negative_prompt,
            "results": results,
            "grid_path": str(grid_path),
            "generated_at": __import__("datetime").datetime.now().isoformat(),
        }

        with open(output_dir / "comparison_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[Generator] Comparison grid saved: {grid_path}")
        return grid_path, metadata


def load_preset_config(preset_id: str) -> Dict:
    """Load LoRA preset configuration"""
    preset_file = Path(f"configs/presets/{preset_id}.yaml")
    if not preset_file.exists():
        raise FileNotFoundError(f"Preset not found: {preset_file}")

    with open(preset_file, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Batch LoRA evaluation generator")
    parser.add_argument("--preset", required=True, help="LoRA preset ID")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--prompts", help="Path to prompts file (one per line)")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 123, 456], help="Seeds to use"
    )
    parser.add_argument("--scale", type=float, default=None, help="LoRA scale override")
    parser.add_argument("--width", type=int, default=768, help="Image width")
    parser.add_argument("--height", type=int, default=768, help="Image height")
    parser.add_argument("--steps", type=int, default=25, help="Inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument(
        "--negative",
        default="lowres, blurry, extra fingers, bad anatomy",
        help="Negative prompt",
    )

    args = parser.parse_args()

    # Load preset config
    preset_config = load_preset_config(args.preset)
    base_model = preset_config["base_model"]
    lora_path = preset_config["lora_path"]
    lora_scale = (
        args.scale if args.scale is not None else preset_config.get("lora_scale", 0.75)
    )
    character_name = preset_config.get("character_name", "character")

    # Load prompts
    if args.prompts:
        with open(args.prompts, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        # Default evaluation prompts
        token = preset_config.get("instance_token", "<token>")
        prompts = [
            f"portrait of {token} in school uniform, looking at viewer, anime style",
            f"{token} sitting in cafe, gentle smile, warm lighting",
            f"close-up of {token}, outdoor scene, cherry blossoms",
            f"{token} in casual clothes, walking in city street",
            f"{token} reading book in library, soft lighting",
        ]

    print(f"[Main] Evaluating preset: {args.preset}")
    print(f"[Main] Character: {character_name}")
    print(f"[Main] Base model: {base_model}")
    print(f"[Main] LoRA path: {lora_path}")
    print(f"[Main] LoRA scale: {lora_scale}")
    print(f"[Main] Prompts: {len(prompts)}")
    print(f"[Main] Seeds: {args.seeds}")

    # Create generator
    generator = LoRABatchGenerator(base_model)

    # Generate comparison grid
    output_dir = Path(args.output)
    grid_path, metadata = generator.generate_comparison_grid(
        prompts=prompts,
        lora_path=lora_path,
        output_dir=output_dir,
        lora_scale=lora_scale,
        negative_prompt=args.negative,
        seeds=args.seeds,
        width=args.width,
        height=args.height,
    )

    print(f"[Main] Evaluation completed!")
    print(f"[Main] Grid saved: {grid_path}")
    print(f"[Main] Individual images: {len(metadata['results'])}")

    # Calculate some basic metrics (placeholder)
    print("\n[Metrics] Basic evaluation:")
    print(f"  - Prompts tested: {len(prompts)}")
    print(f"  - Seeds tested: {len(args.seeds)}")
    print(f"  - Total images: {len(metadata['results']) * 2}")  # Before + After
    print(f"  - LoRA scale: {lora_scale}")

    return grid_path


if __name__ == "__main__":
    main()
