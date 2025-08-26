# core/train/config.py
"""Training configuration management"""
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any
from ..shared_cache import get_shared_cache


@dataclass
class TrainingConfig:
    """LoRA training configuration"""

    # Model settings
    base_model: str = "runwayml/stable-diffusion-v1-5"
    model_type: str = "sd15"  # sd15, sdxl
    resolution: int = 768

    # LoRA settings
    rank: int = 16
    alpha: int = 16

    # Training hyperparameters
    learning_rate: float = 1e-4
    text_encoder_lr: float = 5e-5
    train_steps: int = 8000
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    seed: int = 42

    # Optimization settings
    mixed_precision: str = "fp16"
    gradient_checkpointing: bool = True
    use_8bit_adam: bool = True
    caption_dropout: float = 0.1
    min_snr_gamma: float = 5.0
    noise_offset: float = 0.1

    # Validation settings
    validation_prompts: List[str] = field(
        default_factory=lambda: [
            "anime girl, masterpiece, best quality",
            "anime boy, detailed face, high resolution",
        ]
    )
    validation_steps: int = 1000
    num_validation_images: int = 4

    # Dataset settings
    dataset: Dict[str, Any] = field(
        default_factory=lambda: {
            "metadata_path": "",
            "image_column": "image_path",
            "caption_column": "tags",
            "train_split": 0.8,
            "shuffle": True,
        }
    )

    # Output settings
    output: Dict[str, Any] = field(
        default_factory=lambda: {
            "output_dir": "",
            "save_steps": 1000,
            "save_total_limit": 5,
            "logging_steps": 100,
            "report_to": None,  # wandb, tensorboard, etc.
        }
    )

    # Performance optimizations
    optimizations: Dict[str, Any] = field(
        default_factory=lambda: {
            "enable_attention_slicing": True,
            "enable_vae_slicing": True,
            "enable_cpu_offload": False,
            "dataloader_num_workers": 4,
            "pin_memory": True,
        }
    )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        """Load configuration from YAML file"""
        try:
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)

            return cls(**data)

        except Exception as e:
            raise RuntimeError(f"Failed to load config from {yaml_path}: {str(e)}")

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file"""
        try:
            # Convert dataclass to dict
            config_dict = self.__dict__.copy()

            with open(yaml_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        except Exception as e:
            raise RuntimeError(f"Failed to save config to {yaml_path}: {str(e)}")

    def validate(self) -> Dict[str, List[str]]:
        """Validate configuration settings"""
        errors = []
        warnings = []

        # Check model type
        if self.model_type not in ["sd15", "sdxl"]:
            errors.append("model_type must be 'sd15' or 'sdxl'")

        # Check resolution compatibility
        if self.model_type == "sd15" and self.resolution > 768:
            warnings.append("High resolution with SD1.5 may cause issues")
        elif self.model_type == "sdxl" and self.resolution < 1024:
            warnings.append("SDXL works best with 1024px resolution")

        # Check LoRA settings
        if self.rank > 64:
            warnings.append("Very high LoRA rank may cause overfitting")
        if self.alpha != self.rank:
            warnings.append("Alpha != rank may affect training stability")

        # Check training settings
        if self.learning_rate > 1e-3:
            warnings.append("High learning rate may cause instability")
        if self.batch_size * self.gradient_accumulation_steps < 4:
            warnings.append("Very small effective batch size may slow convergence")

        return {"errors": errors, "warnings": warnings}
