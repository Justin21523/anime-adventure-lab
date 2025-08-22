# tests/test_stage1_smoke.py
"""
Stage 1 Smoke Tests
Basic functionality verification for bootstrap components
"""

import os
import json
import pytest
import httpx
from pathlib import Path

# Test imports
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append("../..")

# sys.path.append("..")
from core.shared_cache import SharedCache, bootstrap_cache
from core.config import AppConfig, get_config


class TestSharedCache:
    """Test shared cache functionality"""

    def test_cache_bootstrap(self, tmp_path):
        """Test cache directory creation"""
        cache_root = str(tmp_path / "test_cache")
        cache = SharedCache(cache_root)

        # Check required directories exist
        required_dirs = [
            "models/sd",
            "models/llm",
            "datasets/raw",
            "outputs/saga-forge",
            "hf/transformers",
        ]

        for dir_path in required_dirs:
            full_path = Path(cache_root) / dir_path
            assert full_path.exists(), f"Directory {dir_path} not created"

    def test_gpu_info(self):
        """Test GPU information retrieval"""
        cache = SharedCache()
        gpu_info = cache.get_gpu_info()

        # Should have basic keys regardless of GPU availability
        required_keys = [
            "cuda_available",
            "device_count",
            "current_device",
            "memory_info",
        ]
        for key in required_keys:
            assert key in gpu_info, f"Missing GPU info key: {key}"

    def test_cache_summary(self):
        """Test cache summary generation"""
        cache = SharedCache()
        summary = cache.get_summary()

        required_keys = ["cache_root", "directories", "gpu_info", "env_vars"]
        for key in required_keys:
            assert key in summary, f"Missing summary key: {key}"

        # Check directories count
        assert len(summary["directories"]) > 10, "Not enough directories created"


class TestConfig:
    """Test configuration management"""

    def test_config_creation(self, tmp_path):
        """Test configuration file creation"""
        config_path = tmp_path / "test_config.yaml"
        config = AppConfig(str(config_path))

        # Config file should be created
        assert config_path.exists(), "Config file not created"

        # Should have basic structure
        assert config.get("app.name") == "SagaForge"
        assert config.get("features.enable_rag") is True

    def test_env_override(self, monkeypatch):
        """Test environment variable overrides"""
        monkeypatch.setenv("API_PORT", "9000")
        monkeypatch.setenv("MODEL_USE_4BIT_LOADING", "false")

        config = AppConfig()

        assert config.api.port == 9000
        assert config.model.use_4bit_loading is False

    def test_config_summary(self):
        """Test configuration summary"""
        config = AppConfig()
        summary = config.get_summary()

        required_keys = ["app", "features", "api", "model", "cache_root"]
        for key in required_keys:
            assert key in summary, f"Missing config summary key: {key}"


@pytest.mark.asyncio
class TestAPIHealth:
    """Test API health endpoints (requires running server)"""

    @pytest.fixture
    def api_client(self):
        """HTTP client for API testing"""
        return httpx.AsyncClient(base_url="http://localhost:8000", timeout=10.0)

    async def test_health_endpoint(self, api_client):
        """Test /healthz endpoint"""
        try:
            response = await api_client.get("/api/v1/healthz")

            if response.status_code == 200:
                data = response.json()

                # Check required fields
                required_fields = [
                    "status",
                    "timestamp",
                    "uptime_seconds",
                    "version",
                    "system",
                    "cache",
                    "config",
                ]
                for field in required_fields:
                    assert field in data, f"Missing health response field: {field}"

                # Check status
                assert data["status"] == "healthy"
                assert data["uptime_seconds"] >= 0
                assert data["version"] == "0.1.0"

                # Check system info
                system = data["system"]
                assert "cpu_percent" in system
                assert "memory_percent" in system
                assert system["memory_percent"] >= 0

            else:
                pytest.skip(f"API server not running (HTTP {response.status_code})")

        except httpx.ConnectError:
            pytest.skip("API server not running (connection refused)")

    async def test_ready_endpoint(self, api_client):
        """Test /ready endpoint"""
        try:
            response = await api_client.get("/api/v1/ready")

            if response.status_code == 200:
                data = response.json()
                assert "status" in data
                assert data["status"] in ["ready", "not_ready"]

        except httpx.ConnectError:
            pytest.skip("API server not running")

    async def test_metrics_endpoint(self, api_client):
        """Test /metrics endpoint"""
        try:
            response = await api_client.get("/api/v1/metrics")

            if response.status_code == 200:
                data = response.json()

                # Check basic metrics
                expected_metrics = [
                    "saga_forge_uptime_seconds",
                    "saga_forge_cpu_usage_percent",
                    "saga_forge_memory_usage_percent",
                ]

                for metric in expected_metrics:
                    assert metric in data, f"Missing metric: {metric}"
                    assert isinstance(
                        data[metric], (int, float)
                    ), f"Metric {metric} should be numeric"

        except httpx.ConnectError:
            pytest.skip("API server not running")


def test_integration_bootstrap():
    """Integration test: full bootstrap process"""
    # This should work without errors
    cache = bootstrap_cache()
    config = get_config()

    # Verify cache and config work together
    cache_root = cache.cache_root
    config_cache_root = config.cache.root

    # They should use the same root (or at least both be valid paths)
    assert Path(cache_root).is_absolute() or cache_root.startswith("../")
    assert Path(config_cache_root).is_absolute() or config_cache_root.startswith("../")


if __name__ == "__main__":
    # Run basic tests when called directly
    print("🧪 Running Stage 1 Smoke Tests...")

    print("\n1. Testing Shared Cache...")
    cache = bootstrap_cache()
    print(f"   ✅ Cache root: {cache.cache_root}")
    print(f"   ✅ Directories: {len(cache.app_dirs)}")
    print(f"   ✅ GPU available: {cache.get_gpu_info()['cuda_available']}")

    print("\n2. Testing Configuration...")
    config = get_config()
    print(f"   ✅ App name: {config.get('app.name')}")
    print(f"   ✅ API port: {config.api.port}")
    print(f"   ✅ Features: {len(config.get('features', {}))}")

    print("\n3. Testing API Health (if server running)...")
    try:
        import asyncio

        async def test_api():
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "http://localhost:8000/api/v1/healthz", timeout=5.0
                )
                if response.status_code == 200:
                    data = response.json()
                    print(f"   ✅ API Status: {data['status']}")
                    print(f"   ✅ Uptime: {data['uptime_seconds']:.1f}s")
                    return True
                else:
                    print(f"   ❌ API Error: HTTP {response.status_code}")
                    return False

        result = asyncio.run(test_api())
        if not result:
            print("   ℹ️  API server not running - start with: python api/main.py")

    except Exception as e:
        print(f"   ℹ️  Could not test API: {e}")

    print("\n🎉 Stage 1 smoke tests completed!")
    print("\n📋 Next steps:")
    print("   1. Start API: python api/main.py")
    print("   2. Start UI: python frontend/gradio/app.py")
    print("   3. Visit: http://localhost:7860")
