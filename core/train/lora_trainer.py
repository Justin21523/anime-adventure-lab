# core/train/lora_trainer.py
"""
LoRA Trainer for Stable Diffusion (SD1.5/SDXL)
Supports low-VRAM training with bitsandbytes + gradient checkpointing
"""

import os
import json
import time
import logging
import numpy as np
import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Callable, Union, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings

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

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append("../..")

from ..exceptions import TrainingError, LoRAError, ModelLoadError, ConfigurationError
from .config import TrainingConfig, DatasetConfig, AnimeDatasetConfig
from .dataset import TrainingDataset, DatasetFactory
from .evaluators import ModelEvaluator
from .registry import get_model_registry


logger = logging.getLogger(__name__)


# Fix: 創建動態配置物件來處理缺失屬性
class DynamicConfig:
    """動態配置物件，處理缺失屬性"""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattr__(self, name):
        # 為常見的配置屬性提供預設值
        defaults = {
            "scaling_factor": 0.18215,
            "num_train_timesteps": 1000,
            "prediction_type": "epsilon",
        }
        return defaults.get(name, None)


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

        # Load file list with error handling
        train_list_path = self.root_dir / train_list
        self.image_files = []

        if train_list_path.exists():
            try:
                with open(train_list_path, "r", encoding="utf-8") as f:
                    self.image_files = [line.strip() for line in f if line.strip()]
            except Exception as e:
                logger.warning(f"Failed to load train list {train_list_path}: {e}")

        # Fallback: scan image directory if train list is empty
        if not self.image_files and self.image_dir.exists():
            for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
                self.image_files.extend(
                    [f.name for f in self.image_dir.glob(f"*{ext}")]
                )
                self.image_files.extend(
                    [f.name for f in self.image_dir.glob(f"*{ext.upper()}")]
                )

        logger.info(f"AnimeDataset loaded {len(self.image_files)} images")

    def __len__(self):
        return max(len(self.image_files), 1)  # Ensure at least 1 for empty datasets

    def __getitem__(self, idx):
        if not self.image_files:
            # Return dummy data for empty datasets
            image = torch.zeros(
                3, self.resolution, self.resolution, dtype=torch.float32
            )
            caption = f"{self.instance_token}, placeholder"
            return {
                "pixel_values": image,
                "input_ids": caption,
            }

        # Handle index out of range
        idx = idx % len(self.image_files)
        image_file = self.image_files[idx]
        image_path = self.image_dir / image_file

        # Load image with error handling
        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize(
                (self.resolution, self.resolution), Image.Resampling.LANCZOS
            )
            # Convert to tensor and normalize to [-1, 1]
            image = torch.tensor(np.array(image)).float() / 127.5 - 1.0
            image = image.permute(2, 0, 1)  # HWC -> CHW
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            # Return black image as fallback
            image = torch.zeros(
                3, self.resolution, self.resolution, dtype=torch.float32
            )

        # Load caption with error handling
        caption_file = self.caption_dir / f"{Path(image_file).stem}.txt"
        if caption_file.exists():
            try:
                with open(caption_file, "r", encoding="utf-8") as f:
                    caption = f.read().strip()
            except Exception:
                caption = f"{self.instance_token}, anime character"
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
            "input_ids": caption,
        }


@dataclass
class TrainingState:
    """Training state container"""

    global_step: int = 0
    epoch: int = 0
    best_loss: float = float("inf")
    last_checkpoint_step: int = 0
    training_start_time: float = 0.0


class LoRATrainer:
    """Main LoRA trainer class"""

    def __init__(
        self,
        config: Union[TrainingConfig, str, Dict[str, Any]],
    ):
        # Handle different config input types
        if isinstance(config, str):
            self.config = TrainingConfig.load(config)
        elif isinstance(config, dict):
            self.config = TrainingConfig.from_dict(config)
        else:
            self.config = config

        # Check dependencies
        missing_deps = []
        missing_deps.append("diffusers")
        missing_deps.append("peft")
        missing_deps.append("transformers")

        if missing_deps:
            raise LoRAError(f"Missing required dependencies: {', '.join(missing_deps)}")

        # Initialize accelerator
        self.accelerator = self._create_accelerator()

        # Training components
        self.pipeline = None
        self.unet = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.noise_scheduler = None
        self.optimizer = None
        self.lr_scheduler = None
        self.dataset = None

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")

        # Training state
        self.state = TrainingState()

        # Setup directories
        self._setup_directories()
        self._setup_output_dir()

    def _create_accelerator(self) -> Optional[Any]:
        """Create accelerator if available"""

        try:
            from accelerate.utils import ProjectConfiguration

            project_config = ProjectConfiguration(
                project_dir=self.config.output_dir,
                automatic_checkpoint_naming=True,
            )

            return Accelerator(
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                mixed_precision=self.config.mixed_precision,
                project_config=project_config,
            )
        except Exception as e:
            logger.warning(f"Failed to create accelerator: {e}")
            return None

    def _setup_directories(self):
        """Setup output directories"""
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logging_dir = Path(self.config.logging_dir or self.output_dir / "logs")
        self.logging_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

    def _setup_output_dir(self) -> Path:
        """Setup output directory with timestamp"""
        if not self.config.output_dir:
            self.config.output_dir = f"./output/lora_training_{int(time.time())}"

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = output_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)

        self.output_dir = output_dir
        return output_dir

    def load_models(self):
        """Load and setup models"""
        logger.info(f"Loading models from {self.config.base_model}")

        try:
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

            # Fix: 確保所有物件都有 config 屬性
            if not hasattr(self.vae, "config"):
                self.vae.config = DynamicConfig(scaling_factor=0.18215)  # type: ignore

            if not hasattr(self.noise_scheduler, "config"):
                self.noise_scheduler.config = DynamicConfig(
                    num_train_timesteps=1000, prediction_type="epsilon"  # type: ignore
                )

            # Freeze VAE and text encoder
            self.vae.requires_grad_(False)
            text_encoder_lr = getattr(self.config, "text_encoder_lr", 0.0)
            if text_encoder_lr == 0:
                self.text_encoder.requires_grad_(False)

            # Enable gradient checkpointing
            gradient_checkpointing = getattr(
                self.config, "gradient_checkpointing", True
            )
            if gradient_checkpointing:
                self.unet.enable_gradient_checkpointing()
                if text_encoder_lr > 0:
                    self.text_encoder.gradient_checkpointing_enable()

            logger.info("Models loaded successfully")

        except Exception as e:
            raise ModelLoadError(self.config.base_model, str(e))

    def setup_dataset_and_dataloader(self):
        """Setup dataset and dataloader with fixed attribute access"""
        try:
            dataset_config = getattr(self.config, "dataset", None)
            if dataset_config is None:
                raise TrainingError("No dataset configuration found", "dataset")

            # Get dataset parameters with defaults
            root_dir = getattr(
                dataset_config, "root", getattr(dataset_config, "path", "./data")
            )
            train_list = getattr(dataset_config, "train_list", "train_list.txt")
            caption_dir = getattr(dataset_config, "caption_dir", "captions")
            image_dir = getattr(dataset_config, "image_dir", "images")
            instance_token = getattr(dataset_config, "instance_token", "<token>")
            dropout_tags = getattr(dataset_config, "dropout_tags", [])

            resolution = getattr(self.config, "resolution", 512)
            caption_dropout = getattr(self.config, "caption_dropout", 0.05)

            # Create AnimeDataset
            self.dataset = AnimeDataset(
                root_dir=root_dir,
                train_list=train_list,
                caption_dir=caption_dir,
                image_dir=image_dir,
                resolution=resolution,
                instance_token=instance_token,
                dropout_tags=dropout_tags,
                caption_dropout=caption_dropout,
            )

            def collate_fn(examples):
                pixel_values = torch.stack(
                    [example["pixel_values"] for example in examples]
                )
                captions = [example["input_ids"] for example in examples]

                # Tokenize captions
                inputs = self.tokenizer(  # type: ignore
                    captions,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,  # type: ignore
                    truncation=True,
                    return_tensors="pt",
                )

                return {
                    "pixel_values": pixel_values,
                    "input_ids": inputs.input_ids,
                }

            self.dataloader = DataLoader(
                self.dataset,
                batch_size=self.config.train_batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=2,
                pin_memory=True,
            )

            logger.info(f"Dataset setup complete: {len(self.dataset)} samples")

        except Exception as e:
            raise TrainingError(f"Dataset setup failed: {e}", "dataset")

    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        try:
            # Get trainable parameters
            params_to_optimize = [p for p in self.unet.parameters() if p.requires_grad]  # type: ignore

            # Get optimization config
            opt_config = getattr(self.config, "optimization", None)
            learning_rate = (
                getattr(opt_config, "learning_rate", 1e-4) if opt_config else 1e-4
            )
            use_8bit_adam = getattr(self.config, "use_8bit_adam", True)

            # Create optimizer
            if use_8bit_adam:
                optimizer_class = bnb.optim.AdamW8bit
                logger.info("Using 8-bit AdamW optimizer")
            else:
                optimizer_class = torch.optim.AdamW
                logger.info("Using regular AdamW optimizer")

            self.optimizer = optimizer_class(
                params_to_optimize,
                lr=learning_rate,
                betas=(0.9, 0.999),
                weight_decay=0.01,
                eps=1e-8,
            )

            # Create learning rate scheduler
            max_train_steps = getattr(self.config, "max_train_steps", 1000)
            num_warmup_steps = (
                getattr(opt_config, "lr_warmup_steps", 100) if opt_config else 100
            )

            try:
                from transformers import get_scheduler

                self.lr_scheduler = get_scheduler(
                    "cosine",
                    optimizer=self.optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=max_train_steps,
                )
            except ImportError:
                # Fallback to torch scheduler
                from torch.optim.lr_scheduler import CosineAnnealingLR

                self.lr_scheduler = CosineAnnealingLR(
                    self.optimizer, T_max=max_train_steps
                )

            logger.info("Optimizer and scheduler setup complete")

        except Exception as e:
            raise TrainingError(f"Optimizer setup failed: {e}", "optimizer")

    def setup_lora(self):
        """Setup LoRA for UNet"""
        try:
            lora_config_data = getattr(self.config, "lora", None)
            if lora_config_data:
                rank = getattr(lora_config_data, "rank", 16)
                alpha = getattr(lora_config_data, "alpha", 32)
                target_modules = getattr(
                    lora_config_data,
                    "target_modules",
                    ["to_k", "to_q", "to_v", "to_out.0"],
                )
            else:
                rank = 16
                alpha = 32
                target_modules = ["to_k", "to_q", "to_v", "to_out.0"]

            lora_config = LoraConfig(
                r=self.config.lora.rank,
                lora_alpha=self.config.lora.alpha,
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                lora_dropout=0.1,
                task_type=TaskType.DIFFUSION,
            )
            self.unet = get_peft_model(self.unet, lora_config)  # type: ignore

            # Print trainable parameters
            trainable_params = sum(
                p.numel() for p in self.unet.parameters() if p.requires_grad
            )
            total_params = sum(p.numel() for p in self.unet.parameters())
            logger.info(
                f"LoRA Trainable: {trainable_params:,} / Total: {total_params:,} "
                f"({100*trainable_params/total_params:.1f}%)"
            )

        except Exception as e:
            raise LoRAError(f"LoRA setup failed: {e}")

    def prepare_dataset(self) -> DataLoader:
        """Prepare training dataset"""
        try:
            # Create dataset
            dataset = DatasetFactory.create_dataset(
                config=self.config.dataset,
                tokenizer=self.tokenizer,
                resolution=self.config.resolution,
                augment=True,
            )

            # Create dataloader
            dataloader = DatasetFactory.create_dataloader(
                dataset=dataset,
                batch_size=self.config.train_batch_size,
                shuffle=True,
                num_workers=2,
            )

            logger.info(f"Dataset prepared: {len(dataset)} samples")
            return dataloader

        except Exception as e:
            raise TrainingError(f"Dataset preparation failed: {e}", "dataset")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step with proper error handling"""
        try:
            device = next(self.unet.parameters()).device  # type: ignore

            # Move batch to device
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)

            # Encode images to latent space
            with torch.no_grad():
                latents = self.vae.encode(pixel_values).latent_dist.sample()  # type: ignore
                latents = latents * self.vae.config.scaling_factor  # type: ignore

            # Sample noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample random timesteps
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,  # type: ignore
                (bsz,),
                device=device,
            ).long()

            # Add noise to latents
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)  # type: ignore

            # Get text embeddings
            with torch.no_grad():
                encoder_hidden_states = self.text_encoder(input_ids)[0]  # type: ignore

            # Predict noise
            model_pred = self.unet(
                noisy_latents, timesteps, encoder_hidden_states  # type: ignore
            ).sample

            # Calculate loss
            prediction_type = getattr(
                self.noise_scheduler.config, "prediction_type", "epsilon"  # type: ignore
            )
            if prediction_type == "epsilon":
                target = noise
            elif prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(latents, noise, timesteps)  # type: ignore
            else:
                target = noise  # fallback

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            return loss  # type: ignore

        except Exception as e:
            logger.error(f"Training step failed: {e}")
            # Return a dummy loss to prevent training from crashing
            return torch.tensor(1.0, requires_grad=True)  # type: ignore

    def validation_step(self):
        """Run validation"""
        if (
            not hasattr(self.config, "validation_prompt")
            or not self.config.validation_prompt
        ):
            return

        try:
            self.unet.eval()  # type: ignore

            # Create pipeline for validation
            pipeline = StableDiffusionPipeline(
                vae=self.vae,  # type: ignore
                text_encoder=self.text_encoder,  # type: ignore
                tokenizer=self.tokenizer,  # type: ignore
                unet=self.unet,  # type: ignore
                scheduler=DPMSolverMultistepScheduler.from_pretrained(
                    self.config.base_model, subfolder="scheduler"  # type: ignore
                ),
                safety_checker=None,  # type: ignore
                feature_extractor=None,  # type: ignore
                requires_safety_checker=False,
            )

            with torch.no_grad():
                images = pipeline(
                    self.config.validation_prompt,
                    num_images_per_prompt=getattr(
                        self.config, "num_validation_images", 2
                    ),
                    guidance_scale=7.5,
                    num_inference_steps=20,
                ).images  # type: ignore

            # Save validation images
            val_dir = self.output_dir / "validation" / f"step_{self.global_step}"
            val_dir.mkdir(parents=True, exist_ok=True)

            for i, image in enumerate(images):
                image.save(val_dir / f"image_{i}.png")

            logger.info(f"Saved {len(images)} validation images to {val_dir}")

        except Exception as e:
            logger.error(f"Validation failed: {e}")
        finally:
            self.unet.train()  # type: ignore

    def generate_validation_images(self, step: int, num_images: int = 4):
        """Generate validation images for monitoring training progress"""
        if not hasattr(self, "validation_prompts"):
            self.validation_prompts = [
                "a beautiful landscape",
                "a cute cat",
                "abstract art",
                "portrait of a person",
            ]

        print(f"[Validation] Generating images at step {step}")

        validation_dir = self.output_dir / "validation" / f"step_{step:06d}"
        validation_dir.mkdir(parents=True, exist_ok=True)

        # Temporarily switch to eval mode
        self.unet.eval()  # type: ignore
        if self.config.text_encoder_lr > 0:
            self.text_encoder.eval()  # type: ignore

        with torch.no_grad():
            for i, prompt in enumerate(self.validation_prompts[:num_images]):
                try:
                    # Tokenize prompt
                    text_input = self.tokenizer(  # type: ignore
                        prompt,
                        padding="max_length",
                        max_length=self.tokenizer.model_max_length,  # type: ignore
                        truncation=True,
                        return_tensors="pt",
                    )

                    # Get text embeddings
                    text_embeddings = self.text_encoder(  # type: ignore
                        text_input.input_ids.to(self.accelerator.device)  # type: ignore
                    )[0]

                    # Generate latents
                    latents = torch.randn(  # type: ignore
                        (1, self.unet.config.in_channels, 64, 64),  # type: ignore
                        device=self.accelerator.device,  # type: ignore
                    )
                    latents = latents * self.noise_scheduler.init_noise_sigma  # type: ignore

                    # Denoising loop (simplified)
                    for t in self.noise_scheduler.timesteps[  # type: ignore
                        -10:
                    ]:  # Quick validation, use fewer steps
                        latent_model_input = self.noise_scheduler.scale_model_input(  # type: ignore
                            latents, t  # type: ignore
                        )
                        noise_pred = self.unet(
                            latent_model_input, t, text_embeddings  # type: ignore
                        ).sample
                        latents = self.noise_scheduler.step(  # type: ignore
                            noise_pred, t, latents  # type: ignore
                        ).prev_sample

                    # Decode latents to image
                    latents = 1 / self.vae.config.scaling_factor * latents  # type: ignore
                    images = self.vae.decode(latents).sample  # type: ignore
                    images = (images / 2 + 0.5).clamp(0, 1)
                    images = images.cpu().permute(0, 2, 3, 1).numpy()

                    # Save image
                    import PIL.Image

                    image = PIL.Image.fromarray((images[0] * 255).astype("uint8"))
                    image.save(
                        validation_dir
                        / f"image_{i:02d}_{prompt.replace(' ', '_')[:20]}.png"
                    )

                except Exception as e:
                    print(f"[Validation] Error generating image {i}: {e}")
                    continue

        # Switch back to train mode
        self.unet.train()  # type: ignore
        if self.config.text_encoder_lr > 0:
            self.text_encoder.train()  # type: ignore

    def save_checkpoint(self, step: int):
        """Save training checkpoint"""
        try:
            checkpoint_dir = (
                self.output_dir / "checkpoints" / f"step_{self.global_step}"
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Save LoRA weights
            self.unet.save_pretrained(checkpoint_dir)  # type: ignore

            # Save training state
            state = {
                "step": step,
                "global_step": self.global_step,
                "epoch": self.epoch,
                "best_loss": self.best_loss,
                "config": self.config.__dict__,
            }

            if hasattr(self, "optimizer"):
                state["optimizer_state_dict"] = (
                    self.optimizer.state_dict()  # type: ignore
                )
            if hasattr(self, "lr_scheduler"):
                state["scheduler_state_dict"] = (
                    self.lr_scheduler.state_dict()  # type: ignore
                )

            torch.save(state, checkpoint_dir / "training_state.pt")
            logger.info(f"Checkpoint saved to {checkpoint_dir}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, checkpoint_path: Path):
        """Load training checkpoint"""
        try:
            checkpoint_path = Path(checkpoint_path)
            state_file = checkpoint_path / "training_state.pt"

            if state_file.exists():
                state_dict = torch.load(state_file, map_location="cpu")

                self.state.global_step = state_dict.get("global_step", 0)
                self.state.epoch = state_dict.get("epoch", 0)
                self.state.best_loss = state_dict.get("best_loss", float("inf"))

                if self.optimizer and "optimizer" in state_dict:
                    self.optimizer.load_state_dict(state_dict["optimizer"])
                if self.lr_scheduler and "lr_scheduler" in state_dict:
                    self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])

                logger.info(f"Checkpoint loaded from {checkpoint_path}")

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")

    def train(self):
        """Main training loop"""
        try:
            logger.info("Starting LoRA training")

            # Setup
            self.load_models()
            self.setup_lora()
            self.setup_dataset_and_dataloader()
            self.setup_optimizer()

            # Prepare with accelerator if available
            if self.accelerator:
                try:
                    (
                        self.unet,
                        self.optimizer,
                        self.dataloader,
                        self.lr_scheduler,
                    ) = self.accelerator.prepare(
                        self.unet, self.optimizer, self.dataloader, self.lr_scheduler
                    )
                except Exception as e:
                    logger.warning(f"Accelerator preparation failed: {e}")

            # Set seed for reproducibility
            seed = getattr(self.config, "seed", 42)
            if seed:
                set_seed(seed)
            #
            # Training loop
            max_train_steps = getattr(self.config, "max_train_steps", 1000)
            save_steps = getattr(self.config, "save_steps", 250)
            eval_steps = getattr(self.config, "eval_steps", 100)

            progress_bar = None
            try:
                from tqdm.auto import tqdm

                progress_bar = tqdm(total=self.config.max_train_steps)
            except ImportError:
                pass

            self.global_step = 0
            running_loss = 0.0

            while self.global_step < max_train_steps:
                for batch in self.dataloader:
                    try:
                        if self.accelerator:
                            with self.accelerator.accumulate(self.unet):
                                loss = self.train_step(batch)
                                self.accelerator.backward(loss)
                                self.optimizer.step()  # type: ignore
                                self.lr_scheduler.step()  # type: ignore
                                self.optimizer.zero_grad()  # type: ignore
                        else:
                            loss = self.train_step(batch)
                            loss.backward()  # type: ignore
                            self.optimizer.step()  # type: ignore
                            self.lr_scheduler.step()  # type: ignore
                            self.optimizer.zero_grad()  # type: ignore

                        # Update tracking
                        running_loss += loss.item()  # type: ignore
                        self.global_step += 1

                        # Logging
                        if self.global_step % self.config.logging_steps == 0:
                            avg_loss = running_loss / self.config.logging_steps
                            print(f"[Step {self.global_step:06d}] Loss: {avg_loss:.4f}")

                            # Store for final metadata
                            self._final_avg_loss = avg_loss
                            running_loss = 0.0

                        if progress_bar:
                            progress_bar.update(1)
                            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})  # type: ignore

                        # Update best loss
                        if loss.item() < self.best_loss:  # type: ignore
                            self.best_loss = loss.item()  # type: ignore

                        # Validation and checkpointing
                        if self.global_step % eval_steps == 0:
                            logger.info(
                                f"Step {self.global_step}: Loss = {loss.item():.4f}"  # type: ignore
                            )

                        # Validation and checkpointing
                        if self.global_step % self.config.validation_steps == 0:
                            self.generate_validation_images(self.global_step)

                        if self.global_step % save_steps == 0:
                            self.save_checkpoint(self.global_step)

                        if self.global_step >= max_train_steps:
                            break

                    except Exception as e:
                        logger.error(f"Training step {self.global_step} failed: {e}")
                        continue  # Continue training despite individual step failures

                self.epoch += 1

            # Save final model
            logger.info("Saving final model...")
            self.save_final_model()

            logger.info(f"Training completed! Output: {self.output_dir}")
            return self.output_dir

        except Exception as e:
            raise TrainingError(f"Training failed: {e}", "training")

    def save_final_model(self) -> Path:
        """Save the final trained model"""
        print(f"[Final] Saving final model...")
        final_dir = self.output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA weights for UNet
        self.accelerator.unwrap_model(self.unet).save_pretrained(final_dir / "unet_lora")  # type: ignore

        # Save LoRA weights for text encoder if trained
        if self.config.text_encoder_lr > 0:
            self.accelerator.unwrap_model(self.text_encoder).save_pretrained(  # type: ignore
                final_dir / "text_encoder_lora"
            )

        # Save tokenizer (for inference compatibility)
        self.tokenizer.save_pretrained(final_dir)  # type: ignore

        # Save training metrics and metadata
        training_metadata = {
            "training_completed": True,
            "final_step": self.global_step,
            "config": self.config.__dict__,
            "completed_at": datetime.now().isoformat(),
            "model_info": {
                "base_model": self.config.base_model,
                "unet_lora": str(final_dir / "unet_lora"),
                "text_encoder_lora": (
                    str(final_dir / "text_encoder_lora")
                    if self.config.text_encoder_lr > 0
                    else None
                ),
                "tokenizer": str(final_dir),
            },
            "performance": {
                "total_steps": self.global_step,
                "avg_loss": getattr(self, "_final_avg_loss", 0.0),
            },
        }

        with open(final_dir / "training_metadata.json", "w", encoding="utf-8") as f:
            json.dump(training_metadata, f, indent=2, ensure_ascii=False)

        # Create model card for easy usage reference
        self._create_model_card(final_dir)

        print(f"[Final] Model saved to: {final_dir}")
        return final_dir

    def _create_model_card(self, model_dir: Path) -> None:
        """Create model card with usage instructions"""
        model_card = f"""# LoRA Diffusion Model

    ## 模型資訊 / Model Information
    - **基礎模型 / Base Model**: {self.config.base_model}
    - **LoRA Rank**: {self.config.rank}
    - **Alpha**: {self.config.alpha}
    - **訓練完成 / Training Completed**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    ## 使用方法 / Usage

    ### Python 載入 / Load in Python
    ```python
    from diffusers import StableDiffusionPipeline
    import torch

    # Load base pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        "{self.config.base_model}",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load LoRA weights
    pipeline.unet.load_attn_procs("{model_dir / 'unet_lora'}")

    # Generate image
    image = pipeline("your prompt here", num_inference_steps=25).images[0]
    ```

    ### API 呼叫 / API Call
    ```bash
    curl -X POST "http://localhost:8000/api/v1/generate" \\
    -H "Content-Type: application/json" \\
    -d '{{
        "prompt": "your prompt",
        "lora_model": "{model_dir.name}",
        "steps": 25
    }}'
    ```

    ## 限制 / Limitations
    - 需要基礎模型配合使用 / Requires base model for inference
    - 針對特定領域微調 / Fine-tuned for specific domain
    - GPU記憶體需求依基礎模型而定 / GPU memory requirements vary by base model

    ## 授權 / License
    與基礎模型相同授權 / Same as base model license
    """

        with open(model_dir / "MODEL_CARD.md", "w", encoding="utf-8") as f:
            f.write(model_card)


class TrainingManager:
    """High-level training management"""

    def __init__(self):
        self.active_trainers = {}
        self.registry = get_model_registry()

    def start_training(
        self,
        config: Union[TrainingConfig, str, Dict[str, Any]],
    ) -> str:
        """Start a new training job"""
        job_id = f"train_{int(time.time())}"

        try:
            trainer = LoRATrainer(config)
            self.active_trainers[job_id] = trainer

            # Start training (in real implementation, this would be async)
            trainer.train()

            return job_id

        except Exception as e:
            if job_id in self.active_trainers:
                del self.active_trainers[job_id]
            raise TrainingError(f"Failed to start training: {e}", "start_training")

    def get_training_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of training job"""
        if job_id not in self.active_trainers:
            return {"status": "not_found"}

        trainer = self.active_trainers[job_id]
        return {
            "status": "running",
            "global_step": trainer.state.global_step,
            "epoch": trainer.state.epoch,
            "max_steps": trainer.config.max_train_steps,
            "best_loss": trainer.state.best_loss,
        }

    def stop_training(self, job_id: str) -> bool:
        """Stop training job"""
        if job_id in self.active_trainers:
            # In a real implementation, you'd need proper cleanup
            del self.active_trainers[job_id]
            return True
        return False


# Factory functions
def get_training_manager() -> TrainingManager:
    """Get training manager instance"""
    return TrainingManager()


def create_lora_trainer(
    config: Union[TrainingConfig, str, Dict[str, Any]],
) -> LoRATrainer:
    """Create LoRA trainer instance"""
    return LoRATrainer(config)


def train_lora_from_config(config_path: str) -> Path:
    """Train LoRA from config file"""
    trainer = LoRATrainer(config_path)
    return trainer.train()
