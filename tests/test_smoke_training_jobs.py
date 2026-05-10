import pytest


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_smoke_training_job_lifecycle_simulated(tmp_path, monkeypatch):
    from fastapi import BackgroundTasks

    from core.train.job_manager import TrainJobManager
    import api.routers.finetune as finetune_router
    import api.routers.jobs as jobs_router

    # Isolate job storage to a temp dir for tests
    isolated = TrainJobManager(cache_root=str(tmp_path))
    finetune_router.train_job_manager = isolated
    jobs_router.job_manager = isolated

    payload = {
        "job_type": "lora_sdxl",
        "simulate": True,
        "base_model": "/mnt/c/ai_models/stable-diffusion/xl/sdxl-base-1.0",
        "dataset_path": "/mnt/c/ai_datasets/smoke_dataset",
        "output_name": "smoke_lora",
        "config": {"max_steps": 10},
    }

    resp = await finetune_router.submit_lora_training(payload, background_tasks=BackgroundTasks())
    job_id = resp["job_id"]
    assert job_id
    assert resp["status"] in {"queued", "pending"}

    status = await jobs_router.get_job_status(job_id)
    assert status["job_id"] == job_id
    assert status["status"] in {"queued", "pending", "running", "completed"}
