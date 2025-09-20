# core/train/config.py
"""
Training Configuration Management
Centralized training configuration and parameter management
"""

import yaml
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict, field
import os

from ..config import get_config
from ..exceptions import ValidationError, ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Dataset configuration"""

    name: str = ""
    path: str = ""
    type: str = "imagefolder"  # imagefolder, json, parquet, huggingface
    caption_column: str = "text"
    image_column: str = "image"
    split: str = "train"
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    validation_split: float = 0.1
    cache_dir: Optional[str] = None


@dataclass
class ModelConfig:
    """Model configuration"""

    base_model_id: str = "runwayml/stable-diffusion-v1-5"
    model_type: str = "sd15"  # sd15, sdxl, sd21
    revision: str = "main"
    torch_dtype: str = "float16"
    device_map: str = "auto"
    low_cpu_mem_usage: bool = True


@dataclass
class LoRAConfig:
    """LoRA configuration"""

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=list)
    bias: str = "none"
    task_type: str = "DIFFUSION_IMAGE_GENERATION"


@dataclass
class OptimizationConfig:
    """Optimization configuration"""

    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 500
    weight_decay: float = 1e-2
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    use_8bit_adam: bool = True
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0


@dataclass
class TrainingConfig:
    """Complete training configuration"""

    # Metadata
    name: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # Model settings
    base_model: str = "runwayml/stable-diffusion-v1-5"
    model_type: str = "sd15"  # sd15, sdxl
    resolution: int = 768

    # Components
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    # Training parameters
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_train_steps: int = 8000
    save_steps: int = 1000
    validation_steps: int = 500
    logging_steps: int = 10
    # LoRA settings
    rank: int = 16
    alpha: int = 16

    # Mixed precision and performance
    mixed_precision: str = "fp16"
    dataloader_num_workers: int = 4

    # Output settings
    output_dir: str = ""
    logging_dir: str = ""
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    output: Dict[str, Any] = field(
        default_factory=lambda: {
            "output_dir": "",
            "save_steps": 1000,
            "save_total_limit": 5,
            "logging_steps": 100,
            "report_to": None,  # wandb, tensorboard, etc.
        }
    )

    # Validation
    validation_prompts: List[str] = field(default_factory=list)
    num_validation_images: int = 4
    validation_guidance_scale: float = 7.5
    validation_inference_steps: int = 25

    # Advanced settings
    train_text_encoder: bool = False
    text_encoder_lr: float = 5e-6
    noise_offset: float = 0.0
    snr_gamma: Optional[float] = None
    use_ema: bool = False

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

    def __post_init__(self):
        # Set default target modules based on model type
        if not self.lora.target_modules:
            if self.model.model_type == "sdxl":
                self.lora.target_modules = [
                    "to_k",
                    "to_q",
                    "to_v",
                    "to_out.0",
                    "ff.net.0.proj",
                    "ff.net.2",
                ]
            else:
                self.lora.target_modules = ["to_k", "to_q", "to_v", "to_out.0"]

        # Set default output directory
        if not self.output_dir:
            ai_cache_root = os.getenv("AI_CACHE_ROOT", "/tmp/ai_cache")
            self.output_dir = f"{ai_cache_root}/models/lora/{self.name or 'training'}"

        # Set default logging directory
        if not self.logging_dir:
            self.logging_dir = f"{self.output_dir}/logs"

        # Set default validation prompts
        if not self.validation_prompts:
            self.validation_prompts = [
                "a photo of a cat",
                "a beautiful landscape",
                "anime character portrait",
                "abstract art painting",
            ]

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


class TrainingConfigManager:
    """Centralized training configuration management"""

    def __init__(self):
        self.config = get_config()
        self.config_dir = Path("configs/train")
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Load available configurations
        self.configs: Dict[str, TrainingConfig] = {}
        self.templates: Dict[str, TrainingConfig] = {}

        self._load_configs()
        self._create_default_templates()

    def _load_configs(self):
        """Load all training configurations from files"""
        for config_file in self.config_dir.glob("*.yaml"):
            try:
                config_name = config_file.stem
                config_data = self.load_config_file(config_file)
                self.configs[config_name] = config_data
                logger.info(f"✅ Loaded training config: {config_name}")
            except Exception as e:
                logger.error(f"❌ Failed to load config {config_file}: {e}")

    def _create_default_templates(self):
        """Create default configuration templates"""
        # SD 1.5 template
        sd15_config = TrainingConfig(
            name="sd15_template",
            description="Default SD 1.5 LoRA training configuration",
            model=ModelConfig(
                base_model_id="runwayml/stable-diffusion-v1-5", model_type="sd15"
            ),
            resolution=512,
            train_batch_size=1,
            max_train_steps=8000,
        )

        # SDXL template
        sdxl_config = TrainingConfig(
            name="sdxl_template",
            description="Default SDXL LoRA training configuration",
            model=ModelConfig(
                base_model_id="stabilityai/stable-diffusion-xl-base-1.0",
                model_type="sdxl",
            ),
            resolution=1024,
            train_batch_size=1,
            max_train_steps=6000,
            lora=LoRAConfig(rank=32, alpha=64),
        )

        # Character training template
        character_config = TrainingConfig(
            name="character_template",
            description="Character consistency training configuration",
            model=ModelConfig(
                base_model_id="runwayml/stable-diffusion-v1-5", model_type="sd15"
            ),
            resolution=512,
            train_batch_size=2,
            max_train_steps=12000,
            lora=LoRAConfig(rank=64, alpha=128, dropout=0.05),
            optimization=OptimizationConfig(
                learning_rate=8e-5, lr_scheduler="cosine_with_restarts"
            ),
            train_text_encoder=True,
            validation_prompts=[
                "a portrait of the character",
                "the character smiling",
                "the character in different clothes",
                "the character from side view",
            ],
        )

        self.templates = {
            "sd15": sd15_config,
            "sdxl": sdxl_config,
            "character": character_config,
        }

    def load_config_file(self, config_path: Path) -> TrainingConfig:
        """Load training configuration from YAML file"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)

            # Nested dataclass construction
            dataset_config = DatasetConfig(**config_dict.get("dataset", {}))
            model_config = ModelConfig(**config_dict.get("model", {}))
            lora_config = LoRAConfig(**config_dict.get("lora", {}))
            opt_config = OptimizationConfig(**config_dict.get("optimization", {}))

            # Remove nested configs from main dict
            main_config = {
                k: v
                for k, v in config_dict.items()
                if k not in ["dataset", "model", "lora", "optimization"]
            }

            training_config = TrainingConfig(
                dataset=dataset_config,
                model=model_config,
                lora=lora_config,
                optimization=opt_config,
                **main_config,
            )

            return training_config

        except Exception as e:
            raise ConfigurationError(f"Failed to load training config: {e}")

    def save_config(
        self, config: TrainingConfig, config_name: Optional[str] = None
    ) -> Path:
        """Save training configuration to file"""
        config_name = config_name or config.name or "unnamed_config"
        config_path = self.config_dir / f"{config_name}.yaml"

        try:
            config_dict = asdict(config)

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)

            # Update internal registry
            self.configs[config_name] = config

            logger.info(f"💾 Saved training config: {config_path}")
            return config_path

        except Exception as e:
            raise ConfigurationError(f"Failed to save training config: {e}")

    def get_config(self, config_name: str) -> TrainingConfig:
        """Get training configuration by name"""
        if config_name in self.configs:
            return self.configs[config_name]
        elif config_name in self.templates:
            return self.templates[config_name]
        else:
            raise ValidationError(f"Training config not found: {config_name}")

    def list_configs(self) -> Dict[str, Dict[str, Any]]:
        """List all available configurations"""
        result = {"configs": {}, "templates": {}}

        for name, config in self.configs.items():
            result["configs"][name] = {
                "description": config.description,
                "model_type": config.model.model_type,
                "resolution": config.resolution,
                "max_steps": config.max_train_steps,
            }

        for name, template in self.templates.items():
            result["templates"][name] = {
                "description": template.description,
                "model_type": template.model.model_type,
                "resolution": template.resolution,
                "max_steps": template.max_train_steps,
            }

        return result

    def create_config_from_template(
        self, template_name: str, custom_params: Dict[str, Any]
    ) -> TrainingConfig:
        """Create new configuration from template with custom parameters"""
        if template_name not in self.templates:
            raise ValidationError(f"Template not found: {template_name}")

        # Get template as dict
        template_dict = asdict(self.templates[template_name])

        # Deep merge custom parameters
        def deep_merge(
            base_dict: Dict[str, Any], update_dict: Dict[str, Any]
        ) -> Dict[str, Any]:
            result = base_dict.copy()
            for key, value in update_dict.items():
                if (
                    key in result
                    and isinstance(result[key], dict)
                    and isinstance(value, dict)
                ):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        merged_dict = deep_merge(template_dict, custom_params)

        # Reconstruct TrainingConfig
        try:
            dataset_config = DatasetConfig(**merged_dict.get("dataset", {}))
            model_config = ModelConfig(**merged_dict.get("model", {}))
            lora_config = LoRAConfig(**merged_dict.get("lora", {}))
            opt_config = OptimizationConfig(**merged_dict.get("optimization", {}))

            main_config = {
                k: v
                for k, v in merged_dict.items()
                if k not in ["dataset", "model", "lora", "optimization"]
            }

            new_config = TrainingConfig(
                dataset=dataset_config,
                model=model_config,
                lora=lora_config,
                optimization=opt_config,
                **main_config,
            )

            return new_config

        except Exception as e:
            raise ConfigurationError(f"Failed to create config from template: {e}")

    def validate_config(self, config: TrainingConfig) -> List[str]:
        """Validate training configuration and return any warnings/errors"""
        warnings = []

        # Dataset validation
        if not config.dataset.path:
            warnings.append("Dataset path is not specified")
        elif not Path(config.dataset.path).exists():
            warnings.append(f"Dataset path does not exist: {config.dataset.path}")

        # Model validation
        if not config.model.base_model_id:
            warnings.append("Base model ID is not specified")

        # LoRA validation
        if config.lora.rank <= 0:
            warnings.append("LoRA rank must be positive")

        if config.lora.alpha <= 0:
            warnings.append("LoRA alpha must be positive")

        # Training validation
        if config.max_train_steps <= 0:
            warnings.append("Max training steps must be positive")

        if config.train_batch_size <= 0:
            warnings.append("Batch size must be positive")

        # Resolution validation
        if config.resolution % 8 != 0:
            warnings.append("Resolution should be divisible by 8")

        # Output directory validation
        output_path = Path(config.output_dir)
        if output_path.exists() and list(output_path.iterdir()):
            warnings.append(f"Output directory is not empty: {config.output_dir}")

        return warnings
