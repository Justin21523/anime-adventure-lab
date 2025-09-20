# tests/test_core_modules.py
"""
Core 模組單元測試
"""

import pytest
from unittest.mock import patch, MagicMock
import tempfile
import json


class TestSharedCache:
    """測試共享快取模組"""

    def test_cache_bootstrap(self, test_cache_dir):
        """測試快取初始化"""
        from core.shared_cache import bootstrap_cache, get_shared_cache

        cache = bootstrap_cache()
        assert cache.cache_root == str(test_cache_dir)
        assert cache.get_cache_stats()["cache_root"] == str(test_cache_dir)

    def test_cache_paths(self, test_cache_dir):
        """測試快取路徑建立"""
        from core.shared_cache import get_shared_cache

        cache = get_shared_cache()

        # 檢查關鍵路徑是否存在
        assert (test_cache_dir / "models" / "lora").exists()
        assert (test_cache_dir / "datasets" / "raw").exists()
        assert (test_cache_dir / "outputs" / "multi-modal-lab").exists()


class TestConfig:
    """測試配置模組"""

    def test_config_loading(self):
        """測試配置載入"""
        from core.config import get_config

        config = get_config()
        assert config.api.prefix == "/api/v1"
        assert config.api.debug is True
        assert config.models.device == "cpu"

    def test_logging_setup(self):
        """測試日誌設定"""
        from core.config import setup_logging, get_config

        config = get_config()
        # Should not raise exception
        setup_logging(config)


class TestLLMAdapter:
    """測試 LLM 適配器"""

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    def test_llm_adapter_init(self, mock_model, mock_tokenizer):
        """測試 LLM 適配器初始化"""
        from core.llm.adapter import LLMAdapter

        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()

        adapter = LLMAdapter()
        assert adapter is not None

    def test_llm_generate_mock(self, mock_models):
        """測試 LLM 生成 (使用 mock)"""
        from core.llm.adapter import get_llm_adapter

        adapter = get_llm_adapter()
        result = adapter.generate("測試提示")
        assert result == "Mock LLM response"


class TestVLMEngine:
    """測試視覺語言模型引擎"""

    def test_vlm_engine_mock(self, mock_models):
        """測試 VLM 引擎 (使用 mock)"""
        from core.vlm.engine import get_vlm_engine

        engine = get_vlm_engine()

        # Test caption
        caption = engine.caption(None)  # Mock doesn't need real image
        assert caption == "Mock caption"

        # Test VQA
        answer = engine.vqa(None, "What is this?")
        assert answer == "Mock VQA answer"


class TestRAGEngine:
    """測試 RAG 引擎"""

    def test_rag_engine_init(self, test_cache_dir):
        """測試 RAG 引擎初始化"""
        from core.rag.engine import RAGEngine

        engine = RAGEngine()
        assert engine is not None
        assert engine.vector_store is not None

    def test_rag_add_documents(self, sample_documents):
        """測試文檔添加"""
        from core.rag.engine import RAGEngine

        engine = RAGEngine()

        for doc in sample_documents:
            doc_id = engine.add_document(doc["content"], doc["metadata"])
            assert doc_id is not None

    def test_rag_search(self, sample_documents):
        """測試文檔搜尋"""
        from core.rag.engine import RAGEngine

        engine = RAGEngine()

        # Add documents
        for doc in sample_documents:
            engine.add_document(doc["content"], doc["metadata"])

        # Search
        results = engine.search("人工智慧", top_k=2)
        assert len(results) <= 2
        assert all("score" in result for result in results)
