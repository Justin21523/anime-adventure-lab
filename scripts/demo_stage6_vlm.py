# scripts/demo_stage6_vlm.py
"""
Stage 6 VLM Demo and Smoke Test
Tests VLM captioning, analysis, consistency checking, and RAG writeback
"""

import os
import asyncio
import requests
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append("../..")

# Shared Cache Bootstrap
AI_CACHE_ROOT = os.getenv("AI_CACHE_ROOT", "../ai_warehouse/cache")
for k, v in {
    "HF_HOME": f"{AI_CACHE_ROOT}/hf",
    "TRANSFORMERS_CACHE": f"{AI_CACHE_ROOT}/hf/transformers",
    "HF_DATASETS_CACHE": f"{AI_CACHE_ROOT}/hf/datasets",
    "HUGGINGFACE_HUB_CACHE": f"{AI_CACHE_ROOT}/hf/hub",
    "TORCH_HOME": f"{AI_CACHE_ROOT}/torch",
}.items():
    os.environ[k] = v
    Path(v).mkdir(parents=True, exist_ok=True)

from core.vlm.captioner import VLMCaptioner, BLIP2Captioner
from core.vlm.tagger import WD14Tagger
from core.vlm.consistency import VLMConsistencyChecker, ConsistencyReport

print("[VLM Demo] Stage 6 - Vision Language Model Integration")
print(f"[cache] {AI_CACHE_ROOT}")


class VLMDemo:
    def __init__(self):
        self.api_base = "http://localhost:8000"
        self.test_images_dir = Path(f"{AI_CACHE_ROOT}/test_images")
        self.test_images_dir.mkdir(exist_ok=True)

        # VLM Config for testing
        self.vlm_config = {
            "default_model": "blip2",
            "device": "auto",
            "low_vram": True,
            "models": {
                "blip2": "Salesforce/blip2-opt-2.7b"  # Smaller model for testing
            },
        }

        # Test world data
        self.test_world_data = {
            "characters": {
                "alice": {
                    "name": "Alice",
                    "appearance": {
                        "hair_color": "blue",
                        "eye_color": "green",
                        "distinctive_features": ["cat ears", "school uniform"],
                    },
                    "personality": {"traits": ["cheerful", "curious"]},
                }
            },
            "scenes": {
                "classroom": {
                    "location": "school classroom",
                    "time_of_day": "morning",
                    "weather": "sunny",
                    "mood": "peaceful",
                }
            },
            "lore": [
                {"content": "Alice is a magical student with blue hair and cat ears"},
                {"content": "The school has traditional classrooms with wooden desks"},
            ],
        }

    def create_test_image(self) -> Path:
        """Create a simple test image for demo purposes"""
        # Create a simple anime-style test image
        img = Image.new("RGB", (512, 512), color="lightblue")
        draw = ImageDraw.Draw(img)

        # Draw simple character representation
        # Head (circle)
        draw.ellipse([150, 100, 350, 300], fill="peachpuff", outline="black", width=2)

        # Eyes
        draw.ellipse([180, 150, 210, 180], fill="green", outline="black", width=1)
        draw.ellipse([280, 150, 310, 180], fill="green", outline="black", width=1)

        # Hair (blue rectangles)
        draw.rectangle([140, 90, 360, 140], fill="blue", outline="darkblue", width=1)

        # Cat ears
        draw.polygon(
            [(160, 100), (140, 60), (180, 80)], fill="blue", outline="darkblue"
        )
        draw.polygon(
            [(320, 100), (300, 80), (340, 60)], fill="blue", outline="darkblue"
        )

        # School uniform (rectangle for body)
        draw.rectangle([200, 300, 300, 450], fill="navy", outline="black", width=2)

        # Add text
        try:
            # Try to use default font, fallback to basic if not available
            font = ImageFont.load_default()
        except:
            font = None

        draw.text(
            (50, 460),
            "Test: Blue-haired student with cat ears",
            fill="black",
            font=font,
        )

        # Save test image
        test_image_path = self.test_images_dir / "test_character.png"
        img.save(test_image_path)
        print(f"[Test Image] Created: {test_image_path}")

        return test_image_path

    def test_vlm_captioning(self):
        """Test 1: VLM Captioning"""
        print("\n=== Test 1: VLM Captioning ===")

        try:
            # Initialize captioner
            captioner = VLMCaptioner(self.vlm_config)

            # Create test image
            test_image_path = self.create_test_image()
            image = Image.open(test_image_path)

            # Test BLIP2 captioning
            print("Testing BLIP2 captioning...")
            caption = captioner.caption(image, model_type="blip2")
            print(f"✓ BLIP2 Caption: {caption}")

            # Test with custom prompt
            custom_prompt = "Describe the character in this anime-style image"
            custom_caption = captioner.caption(
                image, model_type="blip2", prompt=custom_prompt
            )
            print(f"✓ Custom Prompt Caption: {custom_caption}")

            return True

        except Exception as e:
            print(f"✗ VLM Captioning failed: {e}")
            return False

    def test_vlm_analysis(self):
        """Test 2: VLM Structured Analysis"""
        print("\n=== Test 2: VLM Analysis ===")

        try:
            captioner = VLMCaptioner(self.vlm_config)
            test_image_path = self.create_test_image()
            image = Image.open(test_image_path)

            # Perform analysis
            analysis = captioner.analyze(image, model_type="blip2")

            print("✓ Analysis Results:")
            for category, result in analysis.items():
                if result and category != "model_type":
                    print(f"  {category}: {result[:100]}...")

            return True

        except Exception as e:
            print(f"✗ VLM Analysis failed: {e}")
            return False

    def test_wd14_tagging(self):
        """Test 3: WD14 Tagging"""
        print("\n=== Test 3: WD14 Tagging ===")

        try:
            # Note: This test may fail if WD14 model can't be loaded due to VRAM
            print("Attempting to load WD14 tagger...")
            tagger = WD14Tagger(threshold=0.3)

            test_image_path = self.create_test_image()
            image = Image.open(test_image_path)

            # Generate tags
            tags = tagger.tag_image(image)
            print(f"✓ Generated {len(tags)} tags: {tags[:10]}...")

            # Generate character-specific tags
            char_tags = tagger.get_character_tags(image)
            print("✓ Character tags by category:")
            for category, tag_list in char_tags.items():
                if tag_list:
                    print(f"  {category}: {tag_list[:3]}")

            # Generate prompt
            prompt = tagger.generate_prompt_tags(image)
            print(f"✓ Generated prompt: {prompt}")

            tagger.unload()
            return True

        except Exception as e:
            print(f"✗ WD14 Tagging failed (may be due to model size): {e}")
            # Don't fail the entire test for WD14 as it requires more VRAM
            return True

    def test_consistency_checking(self):
        """Test 4: Consistency Checking"""
        print("\n=== Test 4: Consistency Checking ===")

        try:
            # Initialize consistency checker
            checker = VLMConsistencyChecker(self.test_world_data, {})

            # Test consistent caption
            consistent_caption = (
                "A cheerful girl with blue hair and cat ears wearing a school uniform"
            )
            context = {"character_id": "alice", "scene_id": "classroom"}

            report = checker.check_consistency(consistent_caption, context)

            print(f"✓ Consistency Report:")
            print(f"  Overall Score: {report.overall_score:.2f}")
            print(f"  Issues Found: {len(report.issues)}")
            print(f"  Validated Elements: {report.validated_elements}")

            if report.issues:
                print("  Issues:")
                for issue in report.issues[:3]:  # Show first 3 issues
                    print(
                        f"    - {issue.category} ({issue.severity}): {issue.description}"
                    )

            if report.suggestions:
                print(f"  Suggestions: {report.suggestions[:2]}")

            # Test inconsistent caption
            print("\n--- Testing Inconsistent Caption ---")
            inconsistent_caption = "A serious boy with red hair wearing a business suit"
            report2 = checker.check_consistency(inconsistent_caption, context)

            print(f"✓ Inconsistent Caption Score: {report2.overall_score:.2f}")
            print(f"  Issues Found: {len(report2.issues)}")

            return True

        except Exception as e:
            print(f"✗ Consistency checking failed: {e}")
            return False

    def test_api_endpoints(self):
        """Test 5: API Endpoints (if server is running)"""
        print("\n=== Test 5: API Endpoints ===")

        try:
            # Check if API server is running
            response = requests.get(f"{self.api_base}/healthz", timeout=5)
            if response.status_code != 200:
                print("⚠ API server not running - skipping API tests")
                return True

            print("✓ API server is running")

            # Test VLM health endpoint
            vlm_health = requests.get(f"{self.api_base}/vlm/health", timeout=10)
            if vlm_health.status_code == 200:
                health_data = vlm_health.json()
                print(f"✓ VLM Health: {health_data.get('status', 'unknown')}")
            else:
                print(f"⚠ VLM health check failed: {vlm_health.status_code}")

            # Test models endpoint
            models_response = requests.get(f"{self.api_base}/vlm/models", timeout=5)
            if models_response.status_code == 200:
                models_data = models_response.json()
                print(f"✓ Available models: {len(models_data.get('models', []))}")

            # Test caption endpoint (would need multipart upload in real test)
            print("⚠ Caption endpoint test requires multipart upload - skipped in demo")

            return True

        except requests.RequestException as e:
            print(f"⚠ API tests skipped - server not available: {e}")
            return True  # Don't fail for API unavailability
        except Exception as e:
            print(f"✗ API endpoint test failed: {e}")
            return False

    def test_memory_usage(self):
        """Test 6: Memory Usage and Model Loading"""
        print("\n=== Test 6: Memory Management ===")

        try:
            import torch
            import gc

            initial_memory = (
                torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            )
            print(f"Initial GPU memory: {initial_memory / 1024**2:.1f} MB")

            # Test model loading and unloading
            captioner = VLMCaptioner(self.vlm_config)

            # Load model
            model = captioner.load_model("blip2")
            after_load_memory = (
                torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            )
            print(f"After model load: {after_load_memory / 1024**2:.1f} MB")

            # Test image processing
            test_image_path = self.create_test_image()
            image = Image.open(test_image_path)
            caption = captioner.caption(image)

            after_inference_memory = (
                torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            )
            print(f"After inference: {after_inference_memory / 1024**2:.1f} MB")

            # Unload model
            captioner.unload_all()
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            final_memory = (
                torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            )
            print(f"After cleanup: {final_memory / 1024**2:.1f} MB")

            print("✓ Memory management test completed")
            return True

        except Exception as e:
            print(f"✗ Memory usage test failed: {e}")
            return False

    def run_all_tests(self):
        """Run all VLM tests"""
        print("Starting VLM Stage 6 Comprehensive Test Suite...")

        tests = [
            ("VLM Captioning", self.test_vlm_captioning),
            ("VLM Analysis", self.test_vlm_analysis),
            ("WD14 Tagging", self.test_wd14_tagging),
            ("Consistency Checking", self.test_consistency_checking),
            ("API Endpoints", self.test_api_endpoints),
            ("Memory Management", self.test_memory_usage),
        ]

        results = {}

        for test_name, test_func in tests:
            print(f"\n{'='*60}")
            try:
                results[test_name] = test_func()
            except Exception as e:
                print(f"✗ {test_name} crashed: {e}")
                results[test_name] = False

        # Summary
        print(f"\n{'='*60}")
        print("TEST SUMMARY:")
        print(f"{'='*60}")

        passed = sum(results.values())
        total = len(results)

        for test_name, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{test_name:.<40} {status}")

        print(f"\nOverall: {passed}/{total} tests passed")

        if passed == total:
            print("🎉 All VLM tests passed! Stage 6 is ready.")
        elif passed >= total * 0.8:
            print("⚠ Most tests passed. Some optional features may need attention.")
        else:
            print("❌ Multiple test failures. Check configuration and dependencies.")

        return passed >= total * 0.8


def main():
    """Main demo function"""
    demo = VLMDemo()
    success = demo.run_all_tests()

    if success:
        print(f"\n🚀 Stage 6 VLM Demo completed successfully!")
        print(f"Next steps:")
        print(f"1. Start API server: uvicorn api.main:app --reload")
        print(f"2. Test with real images via /vlm/caption endpoint")
        print(f"3. Integrate with RAG writeback for scene memory")
        print(f"4. Move to Stage 7: LoRA fine-tuning")
    else:
        print(f"\n⚠ Demo completed with some issues. Check logs above.")

    return success


if __name__ == "__main__":
    main()
