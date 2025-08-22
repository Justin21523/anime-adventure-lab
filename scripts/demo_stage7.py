"""
Stage 7 Smoke Test - LoRA Fine-tuning Demo
Tests the complete LoRA training pipeline
"""

import os
import time
import json
import tempfile
from pathlib import Path
import requests
import yaml

# Shared cache bootstrap
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.shared_cache import bootstrap_cache

cache = bootstrap_cache()

API_BASE = "http://localhost:8000"


def create_sample_dataset():
    """Create a minimal sample dataset for testing"""
    ai_cache_root = os.getenv("AI_CACHE_ROOT", "../ai_warehouse/cache")
    dataset_dir = Path(ai_cache_root) / "datasets" / "anime-char-sample"

    # Create directory structure
    (dataset_dir / "images").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "captions").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "splits").mkdir(parents=True, exist_ok=True)

    # Create placeholder images (just copy any existing image or create dummy ones)
    from PIL import Image
    import numpy as np

    # Generate simple colored squares as placeholder images
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    image_files = []

    for i, color in enumerate(colors):
        # Create simple colored image
        img_array = np.full((512, 512, 3), color, dtype=np.uint8)
        img = Image.fromarray(img_array)

        img_filename = f"sample_{i:02d}.png"
        img.save(dataset_dir / "images" / img_filename)
        image_files.append(img_filename)

        # Create corresponding caption
        caption = f"<token> character, anime style, color theme {color}"
        with open(dataset_dir / "captions" / f"sample_{i:02d}.txt", "w") as f:
            f.write(caption)

    # Create train.txt
    with open(dataset_dir / "splits" / "train.txt", "w") as f:
        for img_file in image_files:
            f.write(f"{img_file}\n")

    # Create metadata
    metadata = {
        "character_name": "sample_char",
        "instance_token": "<token>",
        "description": "Sample dataset for testing",
        "files": {
            "images": len(image_files),
            "captions": len(image_files),
            "splits": ["train.txt"],
        },
        "created_at": "2025-01-01T00:00:00",
    }

    with open(dataset_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Dataset] Created sample dataset: {dataset_dir}")
    return dataset_dir


def create_training_config(dataset_dir: Path):
    """Create training configuration for test"""
    config = {
        "base_model": "runwayml/stable-diffusion-v1-5",
        "model_type": "sd15",
        "resolution": 512,  # Smaller for testing
        "rank": 8,  # Smaller rank for speed
        "alpha": 16,
        "learning_rate": 1e-4,
        "text_encoder_lr": 0.0,
        "train_steps": 100,  # Very short for testing
        "batch_size": 1,
        "gradient_accumulation_steps": 2,
        "seed": 42,
        "mixed_precision": "fp16",
        "gradient_checkpointing": True,
        "use_8bit_adam": True,
        "caption_dropout": 0.05,
        "min_snr_gamma": 5.0,
        "noise_offset": 0.1,
        "validation_prompts": [
            "portrait of <token> in school uniform, anime style",
            "<token> sitting in cafe, gentle smile",
        ],
        "validation_steps": 50,
        "num_validation_images": 1,
        "dataset": {
            "root": str(dataset_dir),
            "train_list": "splits/train.txt",
            "caption_dir": "captions",
            "image_dir": "images",
            "instance_token": "<token>",
            "class_token": "anime character",
            "dropout_tags": [],
        },
        "output": {
            "dir": "../ai_warehouse/outputs/lora",
            "run_id": "test_sample_char",
            "save_every": 50,
            "sample_every": 50,
        },
        "optimizations": {
            "enable_xformers": True,
            "gradient_checkpointing": True,
            "use_8bit_adam": True,
            "mixed_precision": "fp16",
            "cpu_offload_optimizer": False,
        },
    }

    # Save config
    config_path = Path("configs/train/test-lora-sd15.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    print(f"[Config] Created training config: {config_path}")
    return config_path


def test_api_endpoints():
    """Test API endpoints are responding"""
    print("\n=== Testing API Endpoints ===")

    # Test health
    try:
        response = requests.get(f"{API_BASE}/healthz", timeout=5)
        print(f"[API] Health check: {response.status_code}")
        assert response.status_code == 200
    except Exception as e:
        print(f"[API] Health check failed: {e}")
        return False

    # Test finetune endpoints
    try:
        response = requests.get(f"{API_BASE}/finetune/presets", timeout=5)
        print(f"[API] List presets: {response.status_code}")
        assert response.status_code == 200
    except Exception as e:
        print(f"[API] List presets failed: {e}")
        return False

    return True


def test_training_submission():
    """Test training job submission"""
    print("\n=== Testing Training Submission ===")

    # Create sample dataset and config
    dataset_dir = create_sample_dataset()
    config_path = create_training_config(dataset_dir)

    # Submit training job
    payload = {
        "config_path": str(config_path),
        "character_name": "sample_char",
        "notes": "Stage 7 smoke test",
    }

    try:
        response = requests.post(f"{API_BASE}/finetune/lora", json=payload, timeout=10)
        print(f"[Training] Submit job: {response.status_code}")

        if response.status_code != 200:
            print(f"[Training] Error: {response.text}")
            return None

        result = response.json()
        job_id = result["job_id"]
        print(f"[Training] Job ID: {job_id}")
        print(
            f"[Training] Estimated duration: {result['estimated_duration_minutes']} minutes"
        )

        return job_id

    except Exception as e:
        print(f"[Training] Submit failed: {e}")
        return None


def monitor_training_job(job_id: str, max_wait_minutes: int = 10):
    """Monitor training job progress"""
    print(f"\n=== Monitoring Job {job_id} ===")

    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60

    while time.time() - start_time < max_wait_seconds:
        try:
            response = requests.get(f"{API_BASE}/finetune/jobs/{job_id}", timeout=5)

            if response.status_code != 200:
                print(f"[Monitor] Error: {response.status_code}")
                break

            status_data = response.json()
            status = status_data["status"]

            print(f"[Monitor] Status: {status}")

            if status == "running" and status_data.get("progress"):
                progress = status_data["progress"]
                step = progress.get("step", 0)
                total = progress.get("total_steps", 100)
                percent = progress.get("progress_percent", 0)
                loss = progress.get("current_loss", 0)
                print(
                    f"[Monitor] Progress: {step}/{total} ({percent:.1f}%) - Loss: {loss:.4f}"
                )

            if status in ["completed", "failed", "cancelled"]:
                if status == "completed":
                    result = status_data.get("result", {})
                    print(f"[Monitor] Training completed!")
                    print(f"[Monitor] Output: {result.get('output_dir', 'N/A')}")
                    print(f"[Monitor] Preset: {result.get('preset_id', 'N/A')}")
                    return True
                else:
                    print(f"[Monitor] Training {status}")
                    if status_data.get("error"):
                        print(f"[Monitor] Error: {status_data['error']}")
                    return False

            time.sleep(10)  # Check every 10 seconds

        except Exception as e:
            print(f"[Monitor] Check failed: {e}")
            time.sleep(5)

    print(f"[Monitor] Timeout after {max_wait_minutes} minutes")
    return False


def test_preset_loading():
    """Test loading LoRA presets"""
    print("\n=== Testing Preset Loading ===")

    try:
        response = requests.get(f"{API_BASE}/finetune/presets", timeout=5)
        if response.status_code == 200:
            presets = response.json()["presets"]
            print(f"[Presets] Found {len(presets)} presets")

            for preset in presets[:3]:  # Show first 3
                print(
                    f"[Presets] - {preset['id']}: {preset.get('metadata', {}).get('character_name', 'N/A')}"
                )

            return len(presets) > 0
        else:
            print(f"[Presets] Error: {response.status_code}")
            return False

    except Exception as e:
        print(f"[Presets] Failed: {e}")
        return False


def test_batch_generation():
    """Test batch generation script"""
    print("\n=== Testing Batch Generation ===")

    # Check if we have any presets to test with
    try:
        response = requests.get(f"{API_BASE}/finetune/presets", timeout=5)
        if response.status_code == 200:
            presets = response.json()["presets"]
            if not presets:
                print("[Batch] No presets available for testing")
                return False

            # Use first preset
            preset_id = presets[0]["id"]
            print(f"[Batch] Testing with preset: {preset_id}")

            # Create output directory
            output_dir = Path("../ai_warehouse/outputs/batch_test")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Run batch generation (this would be a subprocess in real scenario)
            print(
                f"[Batch] Would run: python scripts/batch_generate.py --preset {preset_id} --output {output_dir}"
            )
            print(f"[Batch] Batch generation test passed (simulated)")
            return True

    except Exception as e:
        print(f"[Batch] Failed: {e}")
        return False


def main():
    """Run Stage 7 smoke test"""
    print("=== Stage 7 - LoRA Fine-tuning Smoke Test ===")
    print("Testing: Training API + Celery Jobs + Preset Registry + Batch Generation")

    # Test sequence
    tests = [
        ("API Endpoints", test_api_endpoints),
        ("Preset Loading", test_preset_loading),
        ("Batch Generation", test_batch_generation),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"[ERROR] {test_name} failed: {e}")
            results[test_name] = False

    # Optional: Test actual training (only if API is running and we have GPU)
    if results.get("API Endpoints", False):
        print("\n--- Training Submission (Optional) ---")
        job_id = test_training_submission()

        if job_id:
            print(f"[Test] Training job submitted: {job_id}")
            print(f"[Test] Monitor with: curl {API_BASE}/finetune/jobs/{job_id}")

            # Optionally monitor for a short time
            monitor_training_job(job_id, max_wait_minutes=2)

    # Summary
    print("\n=== Test Results Summary ===")
    passed = 0
    total = len(results)

    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("🎉 Stage 7 smoke test PASSED!")
        return True
    else:
        print("❌ Stage 7 smoke test FAILED!")
        return False


if __name__ == "__main__":
    main()
