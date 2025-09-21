#!/usr/bin/env python3
# scripts/minimal_test.py
"""
最小化測試 - 避免複雜依賴的基礎功能測試 (already pass)
"""

import os
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Set up minimal environment
os.environ["AI_CACHE_ROOT"] = str(ROOT_DIR.parent / "ai_warehouse" / "cache")


def test_basic_python():
    """測試基本 Python 功能"""
    print("🐍 測試基本 Python 功能...")

    try:
        import json
        import logging
        import hashlib
        from typing import Dict, Any, Optional
        from pathlib import Path
        from dataclasses import dataclass

        print("   ✅ 基本 Python 模組正常")
        return True
    except Exception as e:
        print(f"   ❌ 基本 Python 模組失敗: {e}")
        return False


def test_optional_imports():
    """測試可選的重型依賴"""
    print("\n📦 測試可選依賴...")

    results = {}

    # Test numpy
    try:
        import numpy as np

        print(f"   ✅ numpy {np.__version__}")
        results["numpy"] = True
    except Exception as e:
        print(f"   ❌ numpy 導入失敗: {e}")
        results["numpy"] = False

    # Test torch
    try:
        import torch

        print(f"   ✅ torch {torch.__version__}")
        results["torch"] = True
    except Exception as e:
        print(f"   ❌ torch 導入失敗: {e}")
        results["torch"] = False

    # Test transformers
    try:
        import transformers

        print(f"   ✅ transformers {transformers.__version__}")
        results["transformers"] = True
    except Exception as e:
        print(f"   ❌ transformers 導入失敗: {e}")
        results["transformers"] = False

    return results


def create_mock_dependencies():
    """創建 mock 版本的重型依賴"""
    print("\n🎭 創建 Mock 依賴...")

    # Create mock torch if not available
    if "torch" not in sys.modules:
        import types

        # Mock torch module
        torch_mock = types.ModuleType("torch")
        torch_mock.cuda = types.ModuleType("cuda")
        torch_mock.cuda.is_available = lambda: False
        torch_mock.cuda.empty_cache = lambda: None
        torch_mock.float16 = "torch.float16"
        torch_mock.float32 = "torch.float32"
        torch_mock.bfloat16 = "torch.bfloat16"

        sys.modules["torch"] = torch_mock
        print("   ✅ 創建 torch mock")

    # Mock transformers if needed
    if "transformers" not in sys.modules:
        import types

        transformers_mock = types.ModuleType("transformers")

        # Mock AutoTokenizer
        class MockAutoTokenizer:
            @staticmethod
            def from_pretrained(model_name, **kwargs):
                mock_tokenizer = types.SimpleNamespace()
                mock_tokenizer.pad_token = "[PAD]"
                mock_tokenizer.eos_token = "[EOS]"
                mock_tokenizer.encode = lambda text: [1, 2, 3, 4, 5]  # Mock encoding
                mock_tokenizer.decode = lambda tokens, **kwargs: "mock decoded text"
                return mock_tokenizer

        # Mock AutoModelForCausalLM
        class MockAutoModelForCausalLM:
            @staticmethod
            def from_pretrained(model_name, **kwargs):
                mock_model = types.SimpleNamespace()
                mock_model.generate = lambda **kwargs: [
                    [1, 2, 3, 4, 5, 6]
                ]  # Mock generation
                mock_model.parameters = lambda: [types.SimpleNamespace(device="cpu")]
                return mock_model

        transformers_mock.AutoTokenizer = MockAutoTokenizer
        transformers_mock.AutoModelForCausalLM = MockAutoModelForCausalLM

        sys.modules["transformers"] = transformers_mock
        print("   ✅ 創建 transformers mock")


def test_core_logic_only():
    """僅測試核心邏輯，不依賴重型庫"""
    print("\n🧠 測試核心邏輯...")

    try:
        # Test config loading (should work without heavy deps)
        from core.config import get_config

        config = get_config()
        print("   ✅ 配置模組載入成功")

        # Test shared cache (basic file operations)
        from core.shared_cache import bootstrap_cache

        cache = bootstrap_cache()
        print("   ✅ 共享快取初始化成功")

        # Test exceptions (pure Python)
        from core.exceptions import ModelLoadError, ValidationError

        print("   ✅ 異常類別載入成功")

        return True

    except Exception as e:
        print(f"   ❌ 核心邏輯測試失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_llm_with_mocks():
    """使用 mock 測試 LLM 邏輯"""
    print("\n🤖 測試 LLM 邏輯（使用 mock）...")

    try:
        # Test base classes (no heavy deps)
        from core.llm.base import ChatMessage, LLMResponse

        message = ChatMessage(role="user", content="test")
        print("   ✅ ChatMessage 創建成功")

        # Test with mock adapter
        from core.llm.adapter import MockLLMAdapter

        mock_llm = MockLLMAdapter("test-model")
        response = mock_llm.generate("Hello world")
        print(f"   ✅ Mock 生成: {response[:50]}...")

        # Test chat
        messages = [ChatMessage(role="user", content="Hello")]
        chat_response = mock_llm.chat(messages)
        print(f"   ✅ Mock 對話: {chat_response.content[:50]}...")

        return True

    except Exception as e:
        print(f"   ❌ LLM 邏輯測試失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """主測試函數"""
    print("🔬 最小化環境測試\n")

    # Test basic functionality first
    basic_ok = test_basic_python()
    if not basic_ok:
        print("\n❌ 基本 Python 環境有問題，請檢查 Python 安裝")
        return False

    # Test optional heavy dependencies
    deps = test_optional_imports()

    # Create mocks for missing dependencies
    create_mock_dependencies()

    # Test core logic without heavy deps
    core_ok = test_core_logic_only()

    # Test LLM logic with mocks
    llm_ok = test_llm_with_mocks()

    print("\n" + "=" * 50)
    print("📊 最小化測試結果:")
    print("=" * 50)
    print(f"基本 Python:     {'✅' if basic_ok else '❌'}")
    print(f"NumPy:           {'✅' if deps.get('numpy') else '❌'}")
    print(f"PyTorch:         {'✅' if deps.get('torch') else '❌'}")
    print(f"Transformers:    {'✅' if deps.get('transformers') else '❌'}")
    print(f"核心邏輯:        {'✅' if core_ok else '❌'}")
    print(f"LLM 邏輯:        {'✅' if llm_ok else '❌'}")

    if core_ok and llm_ok:
        print("\n✨ 核心功能正常！可以繼續開發")
        print("\n建議:")
        if not all(deps.values()):
            print("- 修復重型依賴安裝問題")
            print("- 暫時可以使用 mock 模式開發")
        return True
    else:
        print("\n⚠️ 核心功能有問題，需要修復")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 測試過程發生錯誤: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
