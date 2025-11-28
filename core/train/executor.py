# core/train/executor.py
"""Training executor that bridges API jobs to real or simulated LoRA training."""

from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from core.train.job_manager import TrainJobManager
from core.train.registry import ModelRegistry

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
        simulate = payload.get("simulate")
        if simulate is None:
            simulate = os.getenv("TRAIN_SIMULATE", "1").lower() not in {"0", "false", "no"}

        if simulate:
            self._run_simulated(job_id, payload, job_type="lora")
            return

        try:
            self.job_manager.update_job_status(job_id, "running", progress=0.0)
            self._run_real_lora(job_id, payload)
            self.job_manager.update_job_status(job_id, "completed", progress=100.0)
        except Exception as exc:  # noqa: BLE001
            logger.error("LoRA training失敗: %s", exc)
            self.job_manager.update_job_status(
                job_id, "failed", progress=0.0, result_path=None
            )

    def _run_real_lora(self, job_id: str, payload: Dict[str, Any]) -> None:
        """
        嘗試使用 LoRATrainer 進行真實訓練。
        此處僅呼叫現有 Trainer；實際資源需求與環境由使用者自行確保。
        """
        try:
            from core.train.lora_trainer import train_lora_from_config
            from core.train.config import TrainingConfig
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"缺少訓練依賴，請確認環境：{exc}") from exc

        cfg_dict = payload.get("config") or {}
        base_model = payload.get("base_model") or cfg_dict.get("base_model_id")
        if not base_model:
            raise ValueError("必須提供 base_model")

        # 構造最小 TrainingConfig，允許覆蓋常用欄位
        config = TrainingConfig(
            name=payload.get("output_name", f"job_{job_id}"),
            description=payload.get("notes", ""),
            model={"base_model_id": base_model, "model_type": cfg_dict.get("model_type", "sd15")},
            resolution=cfg_dict.get("resolution", 512),
            train_batch_size=cfg_dict.get("batch_size", 1),
            max_train_steps=cfg_dict.get("max_steps", 100),
        )

        # 執行訓練（此函式內部應處理 LoRA 保存邏輯）
        artifact_path = train_lora_from_config(config)

        self.job_manager.update_job_status(job_id, "completed", progress=100.0, result_path=str(artifact_path))
        self.registry.add(
            payload.get("output_name", job_id),
            {
                "model_type": "lora",
                "base_model": base_model,
                "model_path": str(artifact_path),
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

