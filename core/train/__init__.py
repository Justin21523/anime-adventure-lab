# core/train/__init__.py
"""
Training Module
LoRA training, dataset management, and evaluation
"""
from .config import (
    DatasetConfig,
    ModelConfig,
    LoRAConfig,
    OptimizationConfig,
    TrainingConfig,
    TrainingConfigManager,
    get_training_config_manager,
    # Fix: 添加缺失的導入
    LoRATrainingConfig,
    load_training_config,
    AnimeDatasetConfig,
    create_default_anime_config,
    VAEConfig,
    NoiseSchedulerConfig,
)
from .dataset import (
    TrainingDataset,
    DatasetFactory,
    get_dataset_info,
    ImageCaptionDataset,
)
from .evaluators import (
    EvaluationMetrics,
    ModelEvaluator,
    get_model_evaluator,
    # Fix: 添加整合的舊版評估器
    TrainingEvaluator,
    CLIPSimilarityEvaluator,
    TagConsistencyEvaluator,
    get_training_evaluator,
    get_clip_evaluator,
    get_tag_evaluator,
)

try:
    from .lora_trainer import (
        LoRATrainer,
        TrainingManager,
        create_lora_trainer,
        train_lora_from_config,
        AnimeDataset,
    )
except Exception:  # pragma: no cover - optional heavy dependencies
    LoRATrainer = None  # type: ignore
    TrainingManager = None  # type: ignore
    create_lora_trainer = None  # type: ignore
    train_lora_from_config = None  # type: ignore
    AnimeDataset = None  # type: ignore

from .registry import (
    ModelRegistry,
    get_model_registry,
)
from .job_manager import TrainJobManager
from .executor import TrainingExecutor


__all__ = [
    # Config classes and functions
    "DatasetConfig",
    "ModelConfig",
    "LoRAConfig",
    "OptimizationConfig",
    "TrainingConfig",
    "TrainingConfigManager",
    "get_training_config_manager",
    "LoRATrainingConfig",  # Fix: 添加到導出列表
    "load_training_config",  # Fix: 添加到導出列表
    "AnimeDatasetConfig",
    "create_default_anime_config",
    "VAEConfig",
    "NoiseSchedulerConfig",
    # Dataset classes
    "TrainingDataset",
    "DatasetFactory",
    "get_dataset_info",
    "ImageCaptionDataset",
    # Evaluation classes
    "EvaluationMetrics",
    "ModelEvaluator",
    "get_model_evaluator",
    "TrainingEvaluator",
    "CLIPSimilarityEvaluator",
    "TagConsistencyEvaluator",
    "get_training_evaluator",
    "get_clip_evaluator",
    "get_tag_evaluator",
    # Registry classes
    "ModelRegistry",
    "get_model_registry",
    "TrainJobManager",
    "TrainingExecutor",
    # Training classes (may not be available)
    "LoRATrainer",
    "TrainingManager",
    "create_lora_trainer",
    "train_lora_from_config",
    "AnimeDataset",
]
