#!/usr/bin/env python3
# scripts/fix_llm_modules.py
"""
Fix and Test LLM Modules (already pass)
完整修復 LLM 模組的導入衝突和實作問題
"""

import os
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Set up environment
os.environ.setdefault("AI_CACHE_ROOT", "/mnt/c/ai_cache")
os.environ.setdefault("AI_MODELS_ROOT", "/mnt/c/ai_models")
os.environ.setdefault(
    "AI_OUTPUT_ROOT", "/mnt/c/ai_output/anime-adventure-lab"
)


def setup_test_environment():
    """設置測試環境"""
    print("🔧 設置測試環境...")

    # Create cache directories
    cache_root = Path(os.environ["AI_CACHE_ROOT"])
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "huggingface").mkdir(parents=True, exist_ok=True)
    (cache_root / "torch").mkdir(parents=True, exist_ok=True)

    # Bootstrap shared cache
    try:
        from core.shared_cache import bootstrap_cache

        cache = bootstrap_cache()
        print(f"   ✅ 共享快取初始化: {cache.cache_root}")
    except Exception as e:
        print(f"   ❌ 快取初始化失敗: {e}")
        return False

    return True


def test_core_imports():
    """測試核心模組導入"""
    print("\n📦 測試核心模組導入...")

    try:
        # Test basic exceptions
        from core.exceptions import (
            ModelLoadError,
            ModelNotFoundError,
            CUDAOutOfMemoryError,
            SessionNotFoundError,
            ContextLengthExceededError,
            ValidationError,
        )

        print("   ✅ 核心異常類別導入成功")

        # Test config
        from core.config import get_config

        config = get_config()
        print("   ✅ 配置模組導入成功")

        # Test shared cache
        from core.shared_cache import get_shared_cache

        cache = get_shared_cache()
        print("   ✅ 共享快取模組導入成功")

        return True

    except ImportError as e:
        print(f"   ❌ 核心模組導入失敗: {e}")
        return False


def test_llm_imports():
    """測試 LLM 模組導入"""
    print("\n🤖 測試 LLM 模組導入...")

    try:
        # Test base classes
        from core.llm.base import BaseLLM, ChatMessage, LLMResponse

        print("   ✅ LLM 基礎類別導入成功")

        # Test model loader
        from core.llm.model_loader import ModelLoader, ModelLoadConfig, get_model_loader

        print("   ✅ 模型載入器導入成功")

        # Test chat manager
        from core.llm.chat_manager import ChatManager, ChatSession, get_chat_manager

        print("   ✅ 對話管理器導入成功")

        # Test context manager
        from core.llm.context_manager import (
            ContextManager,
            ContextWindow,
            get_context_manager,
        )

        print("   ✅ 上下文管理器導入成功")

        # Test adapter
        from core.llm.adapter import (
            EnhancedTransformersLLM,
            MockLLMAdapter,
            get_llm_adapter,
        )

        print("   ✅ LLM 適配器導入成功")

        # Test unified imports via __init__.py
        from core.llm import (
            get_llm_adapter,
            get_model_loader,
            get_chat_manager,
            get_context_manager,
            ChatMessage,
            LLMResponse,
            ModelLoadConfig,
        )

        print("   ✅ 統一接口導入成功")

        return True

    except ImportError as e:
        print(f"   ❌ LLM 模組導入失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_model_loader():
    """測試模型載入器功能"""
    print("\n⚙️ 測試模型載入器功能...")

    try:
        from core.llm import get_model_loader, ModelLoadConfig

        # Create loader
        loader = get_model_loader()
        print("   ✅ ModelLoader 實例創建成功")

        # Test config creation
        config = ModelLoadConfig(
            model_name="test/model",
            device_map="cpu",
            torch_dtype="float32",
            use_quantization=False,
        )
        print("   ✅ ModelLoadConfig 創建成功")

        # Test config serialization
        config_dict = config.to_dict()
        cache_key = config.get_cache_key()
        print(f"   ✅ 配置序列化成功，快取鍵: {cache_key[:8]}...")

        # Test memory stats
        memory_stats = loader.get_memory_usage()
        print(f"   ✅ 記憶體統計: {memory_stats}")

        return True

    except Exception as e:
        print(f"   ❌ 模型載入器測試失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_chat_manager():
    """測試對話管理器功能"""
    print("\n💬 測試對話管理器功能...")

    try:
        from core.llm import get_chat_manager, ChatMessage
        from core.exceptions import SessionNotFoundError

        # Create manager (with persistence disabled for testing)
        manager = get_chat_manager(persist_sessions=False)
        print("   ✅ ChatManager 實例創建成功")

        # Test session creation
        session_id = manager.create_session(
            system_prompt="You are a helpful assistant", metadata={"test": True}
        )
        print(f"   ✅ 會話創建成功: {session_id}")

        # Test message addition
        manager.add_message(session_id, "user", "Hello, how are you?")
        manager.add_message(session_id, "assistant", "I'm doing well, thank you!")
        print("   ✅ 訊息添加成功")

        # Test session retrieval
        session = manager.get_session(session_id)
        print(f"   ✅ 會話檢索成功，訊息數量: {session.get_message_count()}")

        # Test invalid session handling
        try:
            manager.get_session("invalid-session-id")
            print("   ❌ 應該拋出 SessionNotFoundError")
            return False
        except SessionNotFoundError:
            print("   ✅ SessionNotFoundError 正確拋出")

        # Test session listing
        sessions = manager.list_sessions()
        print(f"   ✅ 會話列表: {len(sessions)} 個會話")

        return True

    except Exception as e:
        print(f"   ❌ 對話管理器測試失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_context_manager():
    """測試上下文管理器功能"""
    print("\n📝 測試上下文管理器功能...")

    try:
        from core.llm import get_context_manager, ChatMessage

        # Create manager
        manager = get_context_manager()
        print("   ✅ ContextManager 實例創建成功")

        # Test context window retrieval
        context_window = manager.get_context_window("microsoft/DialoGPT-medium")
        print(f"   ✅ 上下文窗口: {context_window.max_context_length} tokens")

        # Test token counting
        test_text = "Hello world, this is a test message for token counting."
        token_count = manager.count_tokens(test_text, "microsoft/DialoGPT-medium")
        print(f"   ✅ Token 計數: {token_count} tokens")

        # Test message preparation
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Hello!"),
            ChatMessage(role="assistant", content="Hi there!"),
            ChatMessage(role="user", content="How are you doing today?"),
        ]

        prepared_messages, token_usage = manager.prepare_context(
            messages, "microsoft/DialoGPT-medium", max_new_tokens=100
        )

        print(
            f"   ✅ 上下文準備: {len(prepared_messages)} 訊息, {token_usage.prompt_tokens} tokens"
        )

        # Test context stats
        stats = manager.get_context_stats(messages, "microsoft/DialoGPT-medium")
        print(f"   ✅ 上下文統計: {stats['utilization']:.2%} 使用率")

        return True

    except Exception as e:
        print(f"   ❌ 上下文管理器測試失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_llm_adapter():
    """測試 LLM 適配器功能"""
    print("\n🎯 測試 LLM 適配器功能...")

    try:
        from core.llm import get_llm_adapter, ChatMessage

        # Test mock adapter (always works)
        mock_adapter = get_llm_adapter(model_name="mock", use_mock=True)
        print("   ✅ Mock LLM 適配器創建成功")

        # Test mock generation
        mock_response = mock_adapter.generate("Hello, world!")
        print(f"   ✅ Mock 生成: {mock_response[:50]}...")

        # Test mock chat
        messages = [ChatMessage(role="user", content="Hello!")]
        chat_response = mock_adapter.chat(messages)
        print(f"   ✅ Mock 對話: {chat_response.content[:50]}...")
        print(f"   ✅ Token 使用: {chat_response.usage}")

        # Test model info
        model_info = mock_adapter.get_model_info()
        print(f"   ✅ 模型資訊: {model_info}")

        return True

    except Exception as e:
        print(f"   ❌ LLM 適配器測試失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_integration():
    """測試整合功能"""
    print("\n🔗 測試整合功能...")

    try:
        from core.llm import (
            get_llm_adapter,
            get_chat_manager,
            get_context_manager,
            ChatMessage,
        )

        # Create components
        adapter = get_llm_adapter(model_name="mock", use_mock=True)
        chat_manager = get_chat_manager(persist_sessions=False)
        context_manager = get_context_manager()

        # Create a conversation
        session_id = chat_manager.create_session(
            system_prompt="You are a helpful AI assistant."
        )

        # Add some conversation history
        chat_manager.add_message(session_id, "user", "What is machine learning?")
        chat_manager.add_message(
            session_id, "assistant", "Machine learning is a subset of AI..."
        )
        chat_manager.add_message(session_id, "user", "Can you give me an example?")

        # Get session messages
        session = chat_manager.get_session(session_id)
        messages = session.get_messages()

        # Prepare context
        prepared_messages, token_usage = context_manager.prepare_context(
            messages, adapter.model_name, max_new_tokens=150
        )

        # Generate response
        response = adapter.chat(prepared_messages, max_length=150)

        # Add response to session
        chat_manager.add_message(session_id, "assistant", response.content)

        print("   ✅ 完整對話流程測試成功")
        print(f"   ✅ 最終回應: {response.content[:100]}...")
        print(f"   ✅ Token 使用: {response.usage}")

        return True

    except Exception as e:
        print(f"   ❌ 整合測試失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_all_tests():
    """執行所有測試"""
    print("🚀 開始 LLM 模組修復與測試\n")

    tests = [
        ("環境設置", setup_test_environment),
        ("核心模組導入", test_core_imports),
        ("LLM 模組導入", test_llm_imports),
        ("模型載入器", test_model_loader),
        ("對話管理器", test_chat_manager),
        ("上下文管理器", test_context_manager),
        ("LLM 適配器", test_llm_adapter),
        ("整合測試", test_integration),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} 發生未預期錯誤: {e}")
            results.append((test_name, False))

    # Print summary
    print("\n" + "=" * 60)
    print("📊 測試結果摘要:")
    print("=" * 60)

    passed = 0
    failed = 0

    for test_name, result in results:
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"{test_name:20} {status}")

        if result:
            passed += 1
        else:
            failed += 1

    print("=" * 60)
    print(f"總計: {passed} 通過, {failed} 失敗")

    if failed == 0:
        print("🎉 所有測試通過！LLM 模組修復成功！")
        return True
    else:
        print("⚠️ 部分測試失敗，需要進一步修復")
        return False


def cleanup_and_exit(success: bool):
    """清理並退出"""
    print("\n🧹 清理測試環境...")

    try:
        # Reset global instances
        from core.llm import shutdown_llm_modules

        shutdown_llm_modules()
        print("   ✅ LLM 模組清理完成")
    except Exception as e:
        print(f"   ⚠️ 清理過程中出現警告: {e}")

    print("\n✨ 測試完成")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    try:
        success = run_all_tests()
        cleanup_and_exit(success)
    except KeyboardInterrupt:
        print("\n\n⏹️ 測試被用戶中斷")
        cleanup_and_exit(False)
    except Exception as e:
        print(f"\n\n💥 測試過程中發生嚴重錯誤: {e}")
        import traceback

        traceback.print_exc()
        cleanup_and_exit(False)
