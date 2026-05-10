# core/train/executor.py
"""Training executor that bridges API jobs to real or simulated LoRA training."""

from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from core.train.job_manager import TrainJobManager
from core.train.registry import ModelRegistry
from core.shared_cache import get_shared_cache

logger = logging.getLogger(__name__)


class TrainingExecutor:
    """
    執行訓練工作的協調器。
    - 預設採用模擬模式（快速、避免下載大模型）
    - 若 simulate=False，則嘗試調用 LoRATrainer 進行真實訓練（需依賴與資源）
    """

    def __init__(self, cache_root: Optional[str] = None):
        self.job_manager = TrainJobManager(cache_root)
        self.registry = ModelRegistry(cache_root)

    def run_lora_job(self, job_id: str, payload: Dict[str, Any]) -> None:
        job_type = str(payload.get("job_type") or payload.get("type") or "lora").strip().lower()
        simulate = payload.get("simulate")
        if simulate is None:
            simulate = os.getenv("TRAIN_SIMULATE", "1").lower() not in {"0", "false", "no"}

        if simulate:
            self._run_simulated(job_id, payload, job_type=job_type or "lora")
            return

        try:
            self.job_manager.update_job_status(job_id, "running", progress=0.0)
            if torch.cuda.is_available():
                from core.runtime import get_model_runtime

                runtime = get_model_runtime()
                with runtime.exclusive_gpu(reason=f"train.{job_type}", device="cuda"):
                    if "llm" in job_type:
                        self._run_real_llm_lora(job_id, payload)
                    else:
                        self._run_real_sdxl_lora(job_id, payload)
            else:
                if "llm" in job_type:
                    self._run_real_llm_lora(job_id, payload)
                else:
                    self._run_real_sdxl_lora(job_id, payload)
            self.job_manager.update_job_status(job_id, "completed", progress=100.0)
        except Exception as exc:  # noqa: BLE001
            logger.error("LoRA training失敗: %s", exc)
            self.job_manager.update_job_status(
                job_id, "failed", progress=0.0, result_path=None
            )

    def _run_real_sdxl_lora(self, job_id: str, payload: Dict[str, Any]) -> None:
        """Run SDXL LoRA training (UNet attn processors)."""
        from core.train.sdxl_lora_trainer import train_sdxl_lora

        cfg = payload.get("config") or {}
        base_model = str(payload.get("base_model") or payload.get("model_name") or cfg.get("base_model") or "").strip()
        dataset_path = str(payload.get("dataset_path") or cfg.get("dataset_path") or cfg.get("dataset") or "").strip()
        output_name = str(payload.get("output_name") or payload.get("output") or f"job_{job_id}").strip()

        if not base_model:
            raise ValueError("必須提供 base_model")
        if not dataset_path:
            raise ValueError("必須提供 dataset_path")

        cache = get_shared_cache()
        run_dir = Path(cache.get_path("OUTPUT_TRAINING")) / "lora_sdxl" / output_name
        lora_out_dir = Path(cache.get_path("MODELS_LORA_SDXL")) / output_name

        def _progress(step: int, total: int, loss: Optional[float]) -> None:
            pct = 0.0
            if total:
                pct = max(0.0, min(100.0, (step / total) * 100.0))
            self.job_manager.update_job(
                job_id,
                status="running",
                progress=round(pct, 2),
                current_step=int(step),
                total_steps=int(total),
                current_loss=float(loss) if loss is not None else None,
            )

        train_sdxl_lora(
            base_model=base_model,
            dataset_path=dataset_path,
            output_name=output_name,
            run_dir=run_dir,
            lora_out_dir=lora_out_dir,
            resolution=int(cfg.get("resolution", 1024) or 1024),
            train_batch_size=int(cfg.get("batch_size", 1) or 1),
            gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 4) or 4),
            max_train_steps=int(cfg.get("max_steps", 1000) or 1000),
            learning_rate=float(cfg.get("learning_rate", 1e-4) or 1e-4),
            lora_rank=int(cfg.get("lora_rank", 16) or 16),
            seed=int(cfg.get("seed", 42) or 42),
            mixed_precision=str(cfg.get("mixed_precision", "fp16") or "fp16"),
            save_steps=int(cfg.get("save_steps", 500) or 500),
            progress_cb=_progress,
        )

        safetensors = list(Path(lora_out_dir).glob("*.safetensors"))
        result_path = str(safetensors[0]) if safetensors else str(lora_out_dir)
        self.job_manager.update_job_status(job_id, "completed", progress=100.0, result_path=result_path)
        self.registry.add(
            output_name,
            {
                "model_type": "lora_sdxl",
                "base_model": base_model,
                "model_path": result_path,
                "run_dir": str(run_dir),
                "tags": payload.get("tags", []),
            },
        )

    def _run_real_llm_lora(self, job_id: str, payload: Dict[str, Any]) -> None:
        """Run LLM LoRA training (QLoRA-friendly)."""
        from core.train.llm_lora_trainer import LLMLoraTrainConfig, train_llm_lora

        cfg = payload.get("config") or {}
        base_model = str(payload.get("base_model") or payload.get("model_name") or cfg.get("base_model") or "").strip()
        dataset_path = str(payload.get("dataset_path") or cfg.get("dataset_path") or "").strip()
        output_name = str(payload.get("output_name") or payload.get("output") or f"job_{job_id}").strip()

        if not base_model:
            raise ValueError("必須提供 base_model")
        if not dataset_path:
            raise ValueError("必須提供 dataset_path")

        cache = get_shared_cache()
        run_dir = Path(cache.get_path("OUTPUT_TRAINING")) / "llm_lora" / output_name
        lora_out_dir = Path(cache.get_path("MODELS_LLM_LORA")) / output_name

        def _progress(step: int, total: int, loss: Optional[float]) -> None:
            pct = 0.0
            if total:
                pct = max(0.0, min(100.0, (step / total) * 100.0))
            self.job_manager.update_job(
                job_id,
                status="running",
                progress=round(pct, 2),
                current_step=int(step),
                total_steps=int(total),
                current_loss=float(loss) if loss is not None else None,
            )

        train_cfg = LLMLoraTrainConfig(
            base_model=base_model,
            dataset_path=dataset_path,
            output_name=output_name,
            max_length=int(cfg.get("max_length", 2048) or 2048),
            learning_rate=float(cfg.get("learning_rate", 2e-4) or 2e-4),
            batch_size=int(cfg.get("batch_size", 1) or 1),
            gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8) or 8),
            max_steps=int(cfg.get("max_steps", 500) or 500),
            warmup_steps=int(cfg.get("warmup_steps", 50) or 50),
            lr_scheduler=str(cfg.get("lr_scheduler", "cosine") or "cosine"),
            seed=int(cfg.get("seed", 42) or 42),
            lora_rank=int(cfg.get("lora_rank", 16) or 16),
            lora_alpha=int(cfg.get("lora_alpha", 32) or 32),
            lora_dropout=float(cfg.get("lora_dropout", 0.05) or 0.05),
            target_modules=list(cfg.get("target_modules")) if isinstance(cfg.get("target_modules"), list) else None,
            use_4bit=bool(cfg.get("use_4bit", True)),
        )
        train_llm_lora(config=train_cfg, run_dir=run_dir, lora_out_dir=lora_out_dir, progress_cb=_progress)

        result_path = str((Path(lora_out_dir) / "adapter_model.safetensors").resolve())
        if not Path(result_path).exists():
            # PEFT may save different names depending on version
            safetensors = list(Path(lora_out_dir).glob("*.safetensors"))
            if safetensors:
                result_path = str(safetensors[0])
            else:
                result_path = str(lora_out_dir)

        self.job_manager.update_job_status(job_id, "completed", progress=100.0, result_path=result_path)
        self.registry.add(
            output_name,
            {
                "model_type": "llm_lora",
                "base_model": base_model,
                "model_path": result_path,
                "run_dir": str(run_dir),
                "tags": payload.get("tags", []),
            },
        )

    def _run_simulated(self, job_id: str, payload: Dict[str, Any], job_type: str) -> None:
        """快速模擬訓練流程，避免消耗大量資源。"""
        artifact_dir = Path(self.job_manager.jobs_dir) / job_type
        artifact_dir.mkdir(parents=True, exist_ok=True)

        for step in range(1, 11):
            time.sleep(0.15)
            self.job_manager.update_job_status(job_id, "running", progress=step * 10)

        artifact_path = artifact_dir / f"{job_id}_lora.safetensors"
        artifact_path.write_text("placeholder lora weights", encoding="utf-8")
        self.job_manager.update_job_status(
            job_id, "completed", progress=100.0, result_path=str(artifact_path)
        )

        self.registry.add(
            payload.get("output_name", job_id),
            {
                "model_type": job_type,
                "base_model": payload.get("base_model", ""),
                "model_path": str(artifact_path),
                "tags": ["simulated"],
            },
        )
