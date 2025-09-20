# tests/mocks/api_mocks.py
"""
API相關的Mock工具
"""

from unittest.mock import patch, MagicMock
from contextlib import contextmanager
from typing import Generator, Any


@contextmanager
def mock_all_models() -> Generator[dict, None, None]:
    """一次性Mock所有模型"""
    with (
        patch("core.llm.adapter.get_llm_adapter") as mock_llm,
        patch("core.vlm.engine.get_vlm_engine") as mock_vlm,
        patch("core.rag.engine.RAGEngine") as mock_rag,
        patch("torch.cuda.is_available", return_value=False),
    ):

        # Setup mocks
        llm_mock = MockLLMAdapter()
        vlm_mock = MockVLMEngine()
        rag_mock = MockRAGEngine()

        mock_llm.return_value = llm_mock
        mock_vlm.return_value = vlm_mock
        mock_rag.return_value = rag_mock

        yield {"llm": llm_mock, "vlm": vlm_mock, "rag": rag_mock}


@contextmanager
def mock_file_operations():
    """Mock文件操作"""
    with (
        patch("builtins.open", create=True) as mock_open,
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.mkdir"),
        patch("shutil.copy2"),
        patch("os.listdir", return_value=["test1.txt", "test2.pdf"]),
    ):

        # Mock文件內容
        mock_open.return_value.__enter__.return_value.read.return_value = (
            "Mock file content"
        )
        yield mock_open


def create_mock_image(width: int = 100, height: int = 100, color: str = "red") -> Any:
    """創建Mock圖片對象"""
    from PIL import Image
    import io

    # 創建測試圖片
    img = Image.new("RGB", (width, height), color=color)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    return img_bytes


def create_mock_upload_file(
    filename: str, content: bytes, content_type: str = "image/jpeg"
):
    """創建Mock上傳文件"""
    from fastapi import UploadFile
    import io

    file_obj = io.BytesIO(content)
    return UploadFile(
        filename=filename, file=file_obj, headers={"content-type": content_type}
    )
