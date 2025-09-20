# tests/conftest.py
"""
測試配置和共用 fixtures
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os
import sys

# Add backend to path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

# Mock environment variables before importing app
test_env = {
    "AI_CACHE_ROOT": "/tmp/test_cache",
    "API_PREFIX": "/api/v1",
    "ALLOWED_ORIGINS": "http://localhost:3000",
    "DEVICE": "cpu",
    "MAX_WORKERS": "1",
    "MAX_BATCH_SIZE": "2",
    "DEBUG": "true",
}

for k, v in test_env.items():
    os.environ[k] = v

from api.main import app


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_cache_dir():
    """Create temporary cache directory"""
    cache_dir = Path(tempfile.mkdtemp(prefix="test_cache_"))

    # Create required subdirectories
    subdirs = [
        "hf/transformers",
        "hf/datasets",
        "hf/hub",
        "torch",
        "models/lora",
        "models/blip2",
        "models/qwen",
        "models/llava",
        "models/embeddings",
        "datasets/raw",
        "datasets/processed",
        "datasets/metadata",
        "outputs/multi-modal-lab",
    ]

    for subdir in subdirs:
        (cache_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Update environment
    os.environ["AI_CACHE_ROOT"] = str(cache_dir)

    yield cache_dir

    # Cleanup
    shutil.rmtree(cache_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def client(test_cache_dir):
    """FastAPI test client"""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_models():
    """Mock all model loading to avoid GPU requirements"""
    with (
        patch("core.llm.adapter.get_llm_adapter") as mock_llm,
        patch("core.vlm.engine.get_vlm_engine") as mock_vlm,
        patch("torch.cuda.is_available", return_value=False),
    ):

        # Mock LLM adapter
        mock_llm_instance = MagicMock()
        mock_llm_instance.generate.return_value = "Mock LLM response"
        mock_llm_instance.list_loaded_models.return_value = ["mock-llm"]
        mock_llm_instance.unload_all.return_value = True
        mock_llm.return_value = mock_llm_instance

        # Mock VLM engine
        mock_vlm_instance = MagicMock()
        mock_vlm_instance.caption.return_value = "Mock caption"
        mock_vlm_instance.vqa.return_value = "Mock VQA answer"
        mock_vlm_instance.get_status.return_value = {"loaded": ["mock-vlm"]}
        mock_vlm_instance.unload_models.return_value = True
        mock_vlm.return_value = mock_vlm_instance

        yield {"llm": mock_llm_instance, "vlm": mock_vlm_instance}


@pytest.fixture
def sample_image_data():
    """Sample image data for testing"""
    from PIL import Image
    import io

    # Create a small test image
    img = Image.new("RGB", (100, 100), color="red")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    return img_bytes.getvalue()


@pytest.fixture
def sample_documents():
    """Sample documents for RAG testing"""
    return [
        {
            "content": "人工智慧是電腦科學的一個分支，旨在創造能夠模擬人類智能的機器。",
            "metadata": {"title": "AI介紹", "source": "test1.txt"},
        },
        {
            "content": "機器學習是人工智慧的子領域，使用統計方法讓電腦從資料中學習。",
            "metadata": {"title": "ML基礎", "source": "test2.txt"},
        },
        {
            "content": "深度學習使用神經網路來處理複雜的模式識別任務。",
            "metadata": {"title": "DL概念", "source": "test3.txt"},
        },
    ]
