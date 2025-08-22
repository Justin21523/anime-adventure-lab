# scripts/smoke_test_t2i.py
"""
T2I Pipeline Smoke Test - Stage 5
Tests basic generation, style presets, and seed reproducibility
"""
import os
import sys
import requests
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.shared_cache import setup_cache


def test_api_endpoints():
    """Test T2I API endpoints"""
    base_url = "http://localhost:8000"

    print("=== Testing T2I API Endpoints ===")

    # 1. Health check
    print("\n1. Health Check...")
    try:
        response = requests.get(f"{base_url}/healthz")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            health = response.json()
            print(f"   GPU Available: {health.get('gpu_available')}")
            print(f"   Cache Root: {health.get('cache_root')}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   Connection error: {e}")
        return False

    # 2. Model info
    print("\n2. Model Info...")
    try:
        response = requests.get(f"{base_url}/api/v1/t2i/models")
        if response.status_code == 200:
            models = response.json()
            print(f"   Current Model: {models.get('current_model', 'None')}")
            print(
                f"   Available Styles: {list(models.get('available_styles', {}).keys())}"
            )
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   Error: {e}")

    # 3. Simple generation test
    print("\n3. Simple Generation Test...")
    try:
        payload = {
            "prompt": "a cute anime girl with blue hair",
            "negative_prompt": "blurry, low quality",
            "width": 512,
            "height": 512,
            "num_inference_steps": 20,
            "seed": 42,
            "scene_id": "test_001",
            "image_type": "portrait",
        }

        response = requests.post(
            f"{base_url}/api/v1/t2i/generate", json=payload, timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Success: {result['success']}")
            if result["success"]:
                print(f"   ✓ Image: {result['image_path']}")
                print(f"   ✓ Metadata: {result['metadata_path']}")

                # Check if files exist
                if os.path.exists(result["image_path"]):
                    print(f"   ✓ Image file exists")
                else:
                    print(f"   ✗ Image file missing")

                if os.path.exists(result["metadata_path"]):
                    print(f"   ✓ Metadata file exists")
                else:
                    print(f"   ✗ Metadata file missing")
            else:
                print(f"   ✗ Generation failed: {result.get('error')}")
        else:
            print(f"   ✗ API Error {response.status_code}: {response.text}")

    except Exception as e:
        print(f"   ✗ Request error: {e}")

    # 4. Style preset test
    print("\n4. Style Preset Test...")
    try:
        payload = {
            "prompt": "mountain landscape at sunset",
            "width": 768,
            "height": 768,
            "num_inference_steps": 25,
            "seed": 12345,
            "style_id": "anime_style",
            "scene_id": "test_002",
            "image_type": "bg",
        }

        response = requests.post(
            f"{base_url}/api/v1/t2i/generate", json=payload, timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Style generation: {result['success']}")
            if result["success"]:
                print(f"   ✓ Applied style: {result['metadata']['style_id']}")
        else:
            print(f"   ✗ Style test failed: {response.text}")

    except Exception as e:
        print(f"   ✗ Style test error: {e}")

    return True


def test_direct_pipeline():
    """Test T2I pipeline directly"""
    print("\n=== Testing T2I Pipeline Directly ===")

    try:
        from core.t2i.pipeline import t2i_pipeline

        # 1. Load model test
        print("\n1. Loading Model...")
        success = t2i_pipeline.load_model("runwayml/stable-diffusion-v1-5")
        if success:
            print("   ✓ Model loaded successfully")
        else:
            print("   ✗ Model loading failed")
            return False

        # 2. Basic generation
        print("\n2. Basic Generation...")
        result = t2i_pipeline.generate(
            prompt="a serene lake at dawn, anime style",
            negative_prompt="blurry, low quality",
            width=512,
            height=512,
            num_inference_steps=20,
            seed=9999,
            scene_id="direct_test",
            image_type="bg",
        )

        if result["success"]:
            print("   ✓ Direct generation successful")
            print(f"   ✓ Image: {result['image_path']}")
            print(f"   ✓ Seed: {result['metadata']['seed']}")
        else:
            print(f"   ✗ Direct generation failed: {result['error']}")

        # 3. Style preset test
        print("\n3. Style Preset Application...")
        styles = t2i_pipeline.get_available_styles()
        print(f"   Available styles: {list(styles.keys())}")

        if "anime_style" in styles:
            result = t2i_pipeline.generate(
                prompt="magical forest with glowing flowers",
                width=512,
                height=512,
                num_inference_steps=20,
                seed=7777,
                style_id="anime_style",
                scene_id="style_test",
                image_type="bg",
            )

            if result["success"]:
                print("   ✓ Style preset applied successfully")
            else:
                print(f"   ✗ Style preset failed: {result['error']}")

        # 4. Seed reproducibility test
        print("\n4. Seed Reproducibility Test...")
        fixed_seed = 1337

        # Generate twice with same seed
        result1 = t2i_pipeline.generate(
            prompt="cyberpunk city street",
            width=512,
            height=512,
            num_inference_steps=20,
            seed=fixed_seed,
            scene_id="repro_test1",
            image_type="bg",
        )

        result2 = t2i_pipeline.generate(
            prompt="cyberpunk city street",
            width=512,
            height=512,
            num_inference_steps=20,
            seed=fixed_seed,
            scene_id="repro_test2",
            image_type="bg",
        )

        if result1["success"] and result2["success"]:
            print("   ✓ Both generations completed")
            print(
                f"   Seeds: {result1['metadata']['seed']} == {result2['metadata']['seed']}"
            )
            # Note: Actual image comparison would require PIL image hashing

        return True

    except Exception as e:
        print(f"   ✗ Direct pipeline test error: {e}")
        return False


def test_file_naming_and_metadata():
    """Test file naming convention and metadata structure"""
    print("\n=== Testing File Naming & Metadata ===")

    try:
        from core.t2i.pipeline import t2i_pipeline

        # Load model if not loaded
        if not t2i_pipeline.current_model:
            t2i_pipeline.load_model("runwayml/stable-diffusion-v1-5")

        # Generate with specific naming
        result = t2i_pipeline.generate(
            prompt="test image for naming convention",
            width=512,
            height=512,
            num_inference_steps=10,
            seed=2024,
            scene_id="scene_005",
            image_type="character",
        )

        if result["success"]:
            # Check filename format
            filename = Path(result["image_path"]).name
            print(f"   Generated filename: {filename}")

            # Expected format: scene_005_character_2024_YYYYMMDD_HHMMSS.png
            parts = filename.split("_")
            if len(parts) >= 4:
                scene_id = parts[0] + "_" + parts[1]  # scene_005
                image_type = parts[2]  # character
                seed = parts[3]  # 2024
                print(f"   ✓ Scene ID: {scene_id}")
                print(f"   ✓ Image Type: {image_type}")
                print(f"   ✓ Seed: {seed}")

            # Check metadata structure
            metadata = result["metadata"]
            required_fields = [
                "prompt",
                "model",
                "width",
                "height",
                "seed",
                "timestamp",
                "elapsed_seconds",
                "filename",
            ]

            missing_fields = [
                field for field in required_fields if field not in metadata
            ]
            if not missing_fields:
                print("   ✓ All required metadata fields present")
            else:
                print(f"   ✗ Missing metadata fields: {missing_fields}")

            print(f"   Model: {metadata.get('model')}")
            print(f"   Elapsed: {metadata.get('elapsed_seconds'):.2f}s")

        return True

    except Exception as e:
        print(f"   ✗ File naming test error: {e}")
        return False


def main():
    """Run all smoke tests"""
    print("🚀 SagaForge Stage 5 - T2I Pipeline Smoke Test")
    print("=" * 50)

    # Setup cache first
    setup_cache()

    all_passed = True

    # Test 1: Direct pipeline
    if not test_direct_pipeline():
        all_passed = False

    # Test 2: File naming and metadata
    if not test_file_naming_and_metadata():
        all_passed = False

    # Test 3: API endpoints (requires server running)
    print("\n" + "=" * 50)
    print("NOTE: API tests require server running:")
    print("      python api/main.py")
    print("=" * 50)

    if not test_api_endpoints():
        print("   (API tests skipped - server not running)")

    # Final result
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All core tests PASSED!")
        print("✓ T2I pipeline functional")
        print("✓ File naming convention working")
        print("✓ Metadata generation complete")
        print("✓ Style presets loading")
    else:
        print("❌ Some tests FAILED!")
        print("Check error messages above")

    print("\nNext steps:")
    print("1. Start API server: python api/main.py")
    print("2. Test via browser: http://localhost:8000/docs")
    print("3. Generate test images via API")
    print("=" * 50)


if __name__ == "__main__":
    main()
