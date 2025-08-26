# core/train/lora_trainer.py
"""
LoRA Trainer for Stable Diffusion (SD1.5/SDXL)
Supports low-VRAM training with bitsandbytes + gradient checkpointing
"""

import os
import json
import time
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import yaml

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
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator
from accelerate.utils import set_seed
import bitsandbytes as bnb

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append("../..")


@dataclass
class TrainingConfig:
    """Training configuration dataclass"""

    base_model: str
    model_type: str  # "sd15" or "sdxl"
    resolution: int
    rank: int
    alpha: int
    learning_rate: float
    text_encoder_lr: float
    train_steps: int
    batch_size: int
    gradient_accumulation_steps: int
    seed: int
    mixed_precision: str
    gradient_checkpointing: bool
    use_8bit_adam: bool
    caption_dropout: float
    min_snr_gamma: float
    noise_offset: float
    validation_prompts: List[str]
    validation_steps: int
    num_validation_images: int
    dataset: Dict
    output: Dict
    optimizations: Dict

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        """Load config from YAML file"""
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


class AnimeDataset(Dataset):
    """Dataset for anime character training"""

    def __init__(
        self,
        root_dir: str,
        train_list: str,
        caption_dir: str,
        image_dir: str,
        resolution: int,
        instance_token: str = "<token>",
        dropout_tags: List[str] = None,  # type: ignore
        caption_dropout: float = 0.05,
    ):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / image_dir
        self.caption_dir = self.root_dir / caption_dir
        self.resolution = resolution
        self.instance_token = instance_token
        self.dropout_tags = dropout_tags or []
        self.caption_dropout = caption_dropout

        # Load file list
        train_list_path = self.root_dir / train_list
        with open(train_list_path, "r") as f:
            self.image_files = [line.strip() for line in f if line.strip()]

        print(f"[Dataset] Loaded {len(self.image_files)} images from {train_list_path}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = self.image_dir / image_file

        # Load image
        image = Image.open(image_path).convert("RGB")
        image = image.resize(
            (self.resolution, self.resolution), Image.Resampling.LANCZOS
        )

        # Convert to tensor and normalize to [-1, 1]
        image = torch.tensor(np.array(image)).float() / 127.5 - 1.0
        image = image.permute(2, 0, 1)  # HWC -> CHW

        # Load caption
        caption_file = self.caption_dir / f"{Path(image_file).stem}.txt"
        if caption_file.exists():
            with open(caption_file, "r", encoding="utf-8") as f:
                caption = f.read().strip()
        else:
            caption = f"{self.instance_token}, anime character"

        # Filter dropout tags
        for tag in self.dropout_tags:
            caption = caption.replace(tag, "").replace("  ", " ").strip()

        # Caption dropout for better generalization
        if random.random() < self.caption_dropout:
            caption = ""

        return {
            "pixel_values": image,
            "input_ids": caption,  # Will be tokenized in collate_fn
        }


class LoRATrainer:
    """Main LoRA trainer class"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )

        # Set seed
        set_seed(config.seed)

        # Setup output directory
        self.output_dir = self._setup_output_dir()

        # Progress callback
        self.progress_callback: Optional[Callable] = None

    def _setup_output_dir(self) -> Path:
        """Setup output directory with timestamp"""
        base_dir = Path(self.config.output["dir"])

        if "${timestamp}" in self.config.output["run_id"]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = self.config.output["run_id"].replace("${timestamp}", timestamp)
        else:
            run_id = self.config.output["run_id"]

        output_dir = base_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(output_dir / "config.yaml", "w") as f:
            yaml.dump(self.config.__dict__, f)

        return output_dir

    def load_models(self):
        """Load and setup models"""
        print(f"[Trainer] Loading models from {self.config.base_model}")

        # Load tokenizer and text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config.base_model, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config.base_model, subfolder="text_encoder"
        )

        # Load UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config.base_model, subfolder="unet"
        )

        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            self.config.base_model, subfolder="vae"
        )

        # Load noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.config.base_model, subfolder="scheduler"
        )

        # Freeze VAE and text encoder
        self.vae.requires_grad_(False)
        if self.config.text_encoder_lr == 0:
            self.text_encoder.requires_grad_(False)

        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if self.config.text_encoder_lr > 0:
                self.text_encoder.gradient_checkpointing_enable()

        # Setup LoRA for UNet
        lora_config = LoraConfig(
            r=self.config.rank,
            lora_alpha=self.config.alpha,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=0.1,
            task_type=TaskType.DIFFUSION,  # type: ignore
        )
        self.unet = get_peft_model(self.unet, lora_config)  # type: ignore

        # Print trainable parameters
        trainable_params = sum(
            p.numel() for p in self.unet.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.unet.parameters())
        print(
            f"[LoRA] Trainable: {trainable_params:,} / Total: {total_params:,} ({100*trainable_params/total_params:.1f}%)"
        )

    def setup_dataset_and_dataloader(self):
        """Setup dataset and dataloader"""
        self.dataset = AnimeDataset(
            root_dir=self.config.dataset["root"],
            train_list=self.config.dataset["train_list"],
            caption_dir=self.config.dataset["caption_dir"],
            image_dir=self.config.dataset["image_dir"],
            resolution=self.config.resolution,
            instance_token=self.config.dataset["instance_token"],
            dropout_tags=self.config.dataset.get("dropout_tags", []),
            caption_dropout=self.config.caption_dropout,
        )

        def collate_fn(examples):
            pixel_values = torch.stack(
                [example["pixel_values"] for example in examples]
            )
            captions = [example["input_ids"] for example in examples]

            # Tokenize captions
            inputs = self.tokenizer(
                captions,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            return {
                "pixel_values": pixel_values,
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
            }

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
        )

    def setup_optimizer(self):
        """Setup optimizer"""
        # Prepare optimizer parameters
        params_to_optimize = []

        # UNet LoRA parameters
        unet_lora_params = [p for p in self.unet.parameters() if p.requires_grad]
        params_to_optimize.append(
            {
                "params": unet_lora_params,
                "lr": self.config.learning_rate,
            }
        )

        # Text encoder parameters (if training)
        if self.config.text_encoder_lr > 0:
            text_encoder_params = [
                p for p in self.text_encoder.parameters() if p.requires_grad
            ]
            params_to_optimize.append(
                {
                    "params": text_encoder_params,
                    "lr": self.config.text_encoder_lr,
                }
            )

        # Use 8-bit Adam if specified
        if self.config.use_8bit_adam:
            try:
                optimizer_cls = bnb.optim.AdamW8bit
            except ImportError:
                print("[Warning] bitsandbytes not available, using regular AdamW")
                optimizer_cls = torch.optim.AdamW
        else:
            optimizer_cls = torch.optim.AdamW

        self.optimizer = optimizer_cls(
            params_to_optimize,
            weight_decay=1e-2,
            eps=1e-8,
        )

    def train_step(self, batch, step):
        """Single training step"""
        with self.accelerator.accumulate(self.unet):
            # Convert images to latent space
            latents = self.vae.encode(
                batch["pixel_values"].to(self.vae.dtype)
            ).latent_dist.sample()  # type: ignore
            latents = latents * self.vae.config.scaling_factor  # type: ignore

            # Sample noise
            noise = torch.randn_like(latents)
            if self.config.noise_offset > 0:
                noise += self.config.noise_offset * torch.randn(
                    (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                )

            # Sample timesteps
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,  # type: ignore
                (latents.shape[0],),
                device=latents.device,
            ).long()

            # Add noise to latents
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)  # type: ignore

            # Get text embeddings
            encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]

            # Predict noise
            model_pred = self.unet(
                noisy_latents, timesteps, encoder_hidden_states
            ).sample

            # Get target
            if self.noise_scheduler.config.prediction_type == "epsilon":  # type: ignore
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":  # type: ignore
                target = self.noise_scheduler.get_velocity(latents, noise, timesteps)  # type: ignore
            else:
                raise ValueError(
                    f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"  # type: ignore
                )

            # Calculate loss with min SNR gamma
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean([1, 2, 3])

            if self.config.min_snr_gamma > 0:
                # Apply min SNR gamma weighting
                snr = self.noise_scheduler.alphas_cumprod[timesteps] / (
                    1 - self.noise_scheduler.alphas_cumprod[timesteps]
                )
                min_snr_gamma = self.config.min_snr_gamma
                snr_weight = (
                    torch.where(
                        snr < min_snr_gamma, snr, torch.ones_like(snr) * min_snr_gamma
                    )
                    / snr
                )
                loss = loss * snr_weight

            loss = loss.mean()

            # Backward pass
            self.accelerator.backward(loss)

            # Gradient clipping
            if self.accelerator.sync_gradients:
                params_to_clip = []
                params_to_clip.extend(self.unet.parameters())
                if self.config.text_encoder_lr > 0:
                    params_to_clip.extend(self.text_encoder.parameters())
                self.accelerator.clip_grad_norm_(params_to_clip, 1.0)

            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.detach().item()

    def generate_validation_images(self, step):
        """Generate validation images"""
        print(f"[Validation] Generating images at step {step}")

        # Create pipeline for validation
        if self.config.model_type == "sdxl":
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.config.base_model,
                unet=self.accelerator.unwrap_model(self.unet),
                text_encoder=self.accelerator.unwrap_model(self.text_encoder),
                torch_dtype=torch.float16,
            )
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.config.base_model,
                unet=self.accelerator.unwrap_model(self.unet),
                text_encoder=self.accelerator.unwrap_model(self.text_encoder),
                torch_dtype=torch.float16,
            )

        pipeline.to(self.accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        # Generate images
        validation_dir = self.output_dir / "validation" / f"step_{step:06d}"
        validation_dir.mkdir(parents=True, exist_ok=True)

        for i, prompt in enumerate(self.config.validation_prompts):
            for j in range(self.config.num_validation_images):
                image = pipeline(
                    prompt,
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    height=self.config.resolution,
                    width=self.config.resolution,
                    generator=torch.Generator(
                        device=self.accelerator.device
                    ).manual_seed(step + j),
                ).images[  # type: ignore
                    0
                ]
                image.save(validation_dir / f"prompt_{i:02d}_seed_{j:02d}.png")

        # Clean up
        del pipeline
        torch.cuda.empty_cache()

    def save_checkpoint(self, step):
        """Save model checkpoint"""
        print(f"[Checkpoint] Saving at step {step}")

        checkpoint_dir = self.output_dir / "checkpoints" / f"step_{step:06d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA weights
        self.accelerator.unwrap_model(self.unet).save_pretrained(
            checkpoint_dir / "unet_lora"
        )

        if self.config.text_encoder_lr > 0:
            self.accelerator.unwrap_model(self.text_encoder).save_pretrained(
                checkpoint_dir / "text_encoder_lora"
            )

    def train(self):
        """Main training loop"""
        print(f"[Training] Starting training for {self.config.train_steps} steps")

        # Setup models and data
        self.load_models()
        self.setup_dataset_and_dataloader()
        self.setup_optimizer()

        # Prepare for training with accelerator
        self.unet, self.optimizer, self.dataloader = self.accelerator.prepare(
            self.unet, self.optimizer, self.dataloader
        )

        if self.config.text_encoder_lr > 0:
            self.text_encoder = self.accelerator.prepare(self.text_encoder)

        # Training metrics
        total_loss = 0.0
        step = 0

        # Training loop
        while step < self.config.train_steps:
            for batch in self.dataloader:
                if step >= self.config.train_steps:
                    break

                # Training step
                loss = self.train_step(batch, step)
                total_loss += loss

                step += 1

                # Logging
                if step % 50 == 0:
                    avg_loss = total_loss / 50
                    print(f"[Step {step:06d}] Loss: {avg_loss:.4f}")

                    # Call progress callback if provided
                    if self.progress_callback:
                        self.progress_callback(step, self.config.train_steps, avg_loss)

                    total_loss = 0.0

                # Validation
                if step % self.config.validation_steps == 0:
                    self.generate_validation_images(step)

                # Save checkpoint
                if step % self.config.output["save_every"] == 0:
                    self.save_checkpoint(step)

        # Final save
        print("[Training] Completed! Saving final model...")
        final_dir = self.output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)

        self.accelerator.unwrap_model(self.unet).save_pretrained(
            final_dir / "unet_lora"
        )
        if self.config.text_encoder_lr > 0:
            self.accelerator.unwrap_model(self.text_encoder).save_pretrained(
                final_dir / "text_encoder_lora"
            )

        # Save training metrics
        metrics = {
            "train_steps": step,
            "final_loss": loss,
            "config": self.config.__dict__,
            "completed_at": datetime.now().isoformat(),
        }

        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        return self.output_dir


def train_lora_from_config(
    config_path: str, progress_callback: Optional[Callable] = None
) -> Path:
    """Train LoRA from config file"""
    config = TrainingConfig.from_yaml(config_path)
    trainer = LoRATrainer(config)

    if progress_callback:
        trainer.progress_callback = progress_callback

    return trainer.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    args = parser.parse_args()

    output_dir = train_lora_from_config(args.config)
    print(f"Training completed! Output: {output_dir}")
