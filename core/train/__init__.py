"""Training domain with lazy exports.

Training is an experimental worker capability; importing its lightweight job
metadata must not initialize Transformers, Diffusers, or evaluation models.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_CONFIG_EXPORTS = [
    "DatasetConfig",
    "ModelConfig",
    "LoRAConfig",
    "OptimizationConfig",
    "TrainingConfig",
    "TrainingConfigManager",
    "get_training_config_manager",
    "LoRATrainingConfig",
    "load_training_config",
    "AnimeDatasetConfig",
    "create_default_anime_config",
    "VAEConfig",
    "NoiseSchedulerConfig",
]
_DATASET_EXPORTS = ["TrainingDataset", "DatasetFactory", "get_dataset_info", "ImageCaptionDataset"]
_EVALUATOR_EXPORTS = [
    "EvaluationMetrics",
    "ModelEvaluator",
    "get_model_evaluator",
    "TrainingEvaluator",
    "CLIPSimilarityEvaluator",
    "TagConsistencyEvaluator",
    "get_training_evaluator",
    "get_clip_evaluator",
    "get_tag_evaluator",
]
_TRAINER_EXPORTS = [
    "LoRATrainer",
    "TrainingManager",
    "create_lora_trainer",
    "train_lora_from_config",
    "AnimeDataset",
]

_EXPORTS = {
    **{name: ("core.train.config", name) for name in _CONFIG_EXPORTS},
    **{name: ("core.train.dataset", name) for name in _DATASET_EXPORTS},
    **{name: ("core.train.evaluators", name) for name in _EVALUATOR_EXPORTS},
    **{name: ("core.train.lora_trainer", name) for name in _TRAINER_EXPORTS},
    "ModelRegistry": ("core.train.registry", "ModelRegistry"),
    "get_model_registry": ("core.train.registry", "get_model_registry"),
    "TrainJobManager": ("core.train.job_manager", "TrainJobManager"),
    "TrainingExecutor": ("core.train.executor", "TrainingExecutor"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
