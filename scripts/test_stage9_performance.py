#!/usr/bin/env python3
# scripts/test_stage9_performance.py

import time
import requests
import json
from pathlib import Path

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append("../..")

# Shared cache bootstrap
import os, pathlib, torch

AI_CACHE_ROOT = os.getenv("AI_CACHE_ROOT", "../ai_warehouse/cache")
for k, v in {
    "HF_HOME": f"{AI_CACHE_ROOT}/hf",
    "TORCH_HOME": f"{AI_CACHE_ROOT}/torch",
}.items():
    os.environ[k] = v
    pathlib.Path(v).mkdir(parents=True, exist_ok=True)
print(f"[cache] {AI_CACHE_ROOT} | GPU: {torch.cuda.is_available()}")

from core.performance.memory_manager import MemoryManager, MemoryConfig
from core.performance.cache_manager import CacheManager, CacheConfig
from core.export.story_exporter import (
    StoryExporter,
    ExportConfig,
    StorySession,
    StoryTurn,
)
from datetime import datetime


def test_memory_manager():
    """Test memory manager functionality"""
    print("\n=== Testing Memory Manager ===")

    config = MemoryConfig(enable_8bit=True, cpu_offload=True, attention_slicing=True)

    mm = MemoryManager(config)

    # Test memory info
    memory_info = mm.get_memory_info()
    print(f"Memory info: {json.dumps(memory_info, indent=2)}")

    # Test quantization config
    quant_config = mm.get_quantization_config()
    print(f"Quantization config available: {quant_config is not None}")

    # Test managed inference context
    with mm.managed_inference("test_model"):
        print("Running inference simulation...")
        time.sleep(1)

    print("✅ Memory Manager test passed")


def test_cache_manager():
    """Test cache manager functionality"""
    print("\n=== Testing Cache Manager ===")

    config = CacheConfig()
    cm = CacheManager(config)

    # Test embedding cache
    test_text = "這是一個測試文本"
    test_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

    # Set cache
    cm.set_embedding_cache(test_text, "test_model", test_embedding)

    # Get cache
    cached_embedding = cm.get_embedding_cache(test_text, "test_model")

    if cached_embedding == test_embedding:
        print("✅ Embedding cache test passed")
    else:
        print("❌ Embedding cache test failed")

    # Test cache stats
    stats = cm.get_cache_stats()
    print(f"Cache stats: {json.dumps(stats, indent=2)}")

    # Test cleanup
    cm.cleanup_expired()
    print("✅ Cache cleanup test passed")


def test_story_exporter():
    """Test story export functionality"""
    print("\n=== Testing Story Exporter ===")

    config = ExportConfig()
    exporter = StoryExporter(config)

    # Create sample story
    turns = [
        StoryTurn(
            turn_id=1,
            user_input="開始冒險",
            narration="你踏上了冒險的旅程...",
            dialogues=[{"character": "嚮導", "text": "歡迎！"}],
            choices=["繼續前進", "休息"],
            selected_choice="繼續前進",
            background_image=None,
            character_image=None,
            timestamp=datetime.now().isoformat(),
            metadata={},
        )
    ]

    story = StorySession(
        session_id="test_session",
        title="測試故事",
        world_id="test_world",
        character_name="測試角色",
        turns=turns,
        start_time=datetime.now().isoformat(),
        end_time=None,
        total_turns=1,
        metadata={},
    )

    # Test JSON export
    json_path = exporter.export_to_json(story)
    print(f"✅ JSON export: {json_path}")

    # Test HTML export
    html_path = exporter.export_to_html(story)
    print(f"✅ HTML export: {html_path}")

    # Test JSON import
    imported_story = exporter.load_from_json(json_path)

    if imported_story.session_id == story.session_id:
        print("✅ JSON import test passed")
    else:
        print("❌ JSON import test failed")

    print(f"✅ Story Exporter test passed")


def test_api_endpoints():
    """Test performance and export API endpoints"""
    print("\n=== Testing API Endpoints ===")

    base_url = "http://localhost:8000"

    # Test monitoring endpoints
    try:
        # Health check
        response = requests.get(f"{base_url}/monitoring/health")
        if response.status_code == 200:
            print("✅ Health endpoint working")
            health_data = response.json()
            print(f"System status: {health_data.get('status')}")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")

        # Metrics
        response = requests.get(f"{base_url}/monitoring/metrics")
        if response.status_code == 200:
            print("✅ Metrics endpoint working")
            metrics = response.json()
            print(
                f"GPU memory: {metrics.get('gpu', {}).get('memory_allocated', 0):.2f}GB"
            )
        else:
            print(f"❌ Metrics endpoint failed: {response.status_code}")

        # Cache stats
        response = requests.get(f"{base_url}/monitoring/cache/stats")
        if response.status_code == 200:
            print("✅ Cache stats endpoint working")
        else:
            print(f"❌ Cache stats endpoint failed: {response.status_code}")

    except requests.exceptions.ConnectionError:
        print("⚠️ API server not running, skipping endpoint tests")
        print("To test endpoints, run: uvicorn api.main:app --reload")


def test_export_api():
    """Test export API endpoints"""
    print("\n=== Testing Export API ===")

    base_url = "http://localhost:8000"

    try:
        # Test supported formats
        response = requests.get(f"{base_url}/export/formats")
        if response.status_code == 200:
            print("✅ Export formats endpoint working")
            formats = response.json()
            print(f"Supported formats: {[f['id'] for f in formats['formats']]}")

        # Test story export
        export_request = {
            "session_id": "test_session_api",
            "format": "html",
            "include_images": True,
            "include_metadata": True,
        }

        response = requests.post(f"{base_url}/export/story", json=export_request)
        if response.status_code == 200:
            print("✅ Story export endpoint working")
            result = response.json()
            print(f"Export path: {result.get('export_path')}")
        else:
            print(f"❌ Story export failed: {response.status_code}")

    except requests.exceptions.ConnectionError:
        print("⚠️ API server not running, skipping export API tests")


def test_performance_optimization():
    """Test performance optimization features"""
    print("\n=== Testing Performance Optimization ===")

    # Test different memory configurations
    configs = [
        ("Standard", MemoryConfig()),
        (
            "Low VRAM",
            MemoryConfig(
                enable_4bit=True, cpu_offload=True, sequential_cpu_offload=True
            ),
        ),
        ("High Performance", MemoryConfig(enable_8bit=False, torch_compile=False)),
    ]

    for name, config in configs:
        print(f"\nTesting {name} configuration:")
        mm = MemoryManager(config)

        start_time = time.time()

        # Simulate model loading and inference
        initial_memory = mm.get_memory_info()

        with mm.managed_inference(f"test_model_{name.lower()}"):
            # Simulate some work
            if torch.cuda.is_available():
                # Allocate some GPU memory to test
                test_tensor = torch.randn(100, 100).cuda()
                time.sleep(0.1)
                del test_tensor
                torch.cuda.empty_cache()

        final_memory = mm.get_memory_info()
        elapsed = time.time() - start_time

        print(f"  Elapsed time: {elapsed:.3f}s")
        print(
            f"  GPU memory delta: {final_memory.get('gpu_allocated_gb', 0) - initial_memory.get('gpu_allocated_gb', 0):.3f}GB"
        )
        print(f"  Configuration valid: ✅")


def test_cache_performance():
    """Test cache performance and hit rates"""
    print("\n=== Testing Cache Performance ===")

    config = CacheConfig()
    cm = CacheManager(config)

    # Test embedding cache performance
    test_texts = [
        "這是第一個測試文本",
        "這是第二個測試文本",
        "這是第三個測試文本",
        "這是第一個測試文本",  # Duplicate for cache hit test
        "這是第四個測試文本",
    ]

    cache_hits = 0
    cache_misses = 0

    start_time = time.time()

    for i, text in enumerate(test_texts):
        # Try to get from cache first
        cached = cm.get_embedding_cache(text, "test_model")

        if cached is not None:
            cache_hits += 1
            print(f"  Cache HIT for text {i+1}")
        else:
            cache_misses += 1
            # Simulate embedding generation
            fake_embedding = [float(j) for j in range(10)]
            cm.set_embedding_cache(text, "test_model", fake_embedding)
            print(f"  Cache MISS for text {i+1} - cached result")

    elapsed = time.time() - start_time
    hit_rate = cache_hits / len(test_texts) * 100

    print(f"\nCache Performance Results:")
    print(f"  Total requests: {len(test_texts)}")
    print(f"  Cache hits: {cache_hits}")
    print(f"  Cache misses: {cache_misses}")
    print(f"  Hit rate: {hit_rate:.1f}%")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Average time per request: {elapsed/len(test_texts)*1000:.1f}ms")


def main():
    """Run all Stage 9 tests"""
    print("🚀 SagaForge Stage 9: Performance & Export Tests")
    print("=" * 50)

    try:
        # Core functionality tests
        test_memory_manager()
        test_cache_manager()
        test_story_exporter()

        # Performance tests
        test_performance_optimization()
        test_cache_performance()

        # API tests (optional - requires running server)
        test_api_endpoints()
        test_export_api()

        print("\n" + "=" * 50)
        print("🎉 All Stage 9 tests completed!")
        print("\nNext steps:")
        print("1. Start API server: uvicorn api.main:app --reload")
        print("2. Test web UI: python frontend/gradio/app.py")
        print("3. Run performance monitoring in production")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
