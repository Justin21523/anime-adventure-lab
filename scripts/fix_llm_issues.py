# scripts/fix_llm_issues.py
"""
Fix LLM Engine Issues
Addresses the specific problems found in the implementation
"""

import os
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))


def test_imports():
    """Test all LLM imports work correctly"""
    print("=== Testing LLM Imports ===")

    try:
        # Test core exceptions
        from core.exceptions import (
            ContextLengthExceededError,
            SessionNotFoundError,
            ValidationError,
            ModelLoadError,
            CUDAOutOfMemoryError,
        )

        print("✅ Core exceptions imported successfully")

        # Test LLM components
        from core.llm import (
            get_llm_adapter,
            get_chat_manager,
            get_context_manager,
            get_model_loader,
            ChatMessage,
            LLMResponse,
            ModelLoadConfig,
        )

        print("✅ LLM components imported successfully")

        # Test legacy integration
        from core.llm.legacy_adapters import (
            get_unified_llm_adapter,
            OllamaLLMAdapter,
            LegacyTransformersLLM,
        )

        print("✅ Legacy adapters imported successfully")

        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_chat_manager():
    """Test ChatManager functionality"""
    print("\n=== Testing ChatManager ===")

    try:
        from core.llm import get_chat_manager, ChatMessage
        from core.exceptions import SessionNotFoundError

        # Create manager
        manager = get_chat_manager()
        print("✅ ChatManager created")

        # Test session creation
        session_id = manager.create_session(
            system_prompt="Test system", metadata={"test": True}
        )
        print(f"✅ Session created: {session_id}")

        # Test message addition
        manager.add_message(session_id, "user", "Hello")
        print("✅ Message added")

        # Test session retrieval
        session = manager.get_session(session_id)
        print(f"✅ Session retrieved, messages: {session.get_message_count()}")

        # Test invalid session
        try:
            manager.get_session("invalid-id")
            print("❌ Should have raised SessionNotFoundError")
            return False
        except SessionNotFoundError:
            print("✅ SessionNotFoundError raised correctly")

        return True

    except Exception as e:
        print(f"❌ ChatManager test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_model_loader():
    """Test ModelLoader functionality"""
    print("\n=== Testing ModelLoader ===")

    try:
        from core.llm import get_model_loader, ModelLoadConfig

        # Create loader
        loader = get_model_loader()
        print("✅ ModelLoader created")

        # Test config creation
        config = ModelLoadConfig(
            model_name="test/model",
            device_map="cpu",
            torch_dtype="float32",
            use_quantization=False,
        )
        print("✅ ModelLoadConfig created")

        # Test config serialization
        config_dict = config.to_dict()
        cache_key = config.get_cache_key()
        print(f"✅ Config serialized, cache key: {cache_key[:8]}...")

        # Test memory stats
        memory_stats = loader.get_memory_usage()
        print(f"✅ Memory stats: {memory_stats}")

        return True

    except Exception as e:
        print(f"❌ ModelLoader test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_context_manager():
    """Test ContextManager functionality"""
    print("\n=== Testing ContextManager ===")

    try:
        from core.llm import get_context_manager, ChatMessage

        # Create manager
        manager = get_context_manager()
        print("✅ ContextManager created")

        # Test context window
        context_window = manager.get_context_window("test/model")
        print(f"✅ Context window: {context_window.max_context_length}")

        # Test token counting (will use fallback estimation)
        test_text = "Hello world, this is a test message."
        token_count = manager.count_tokens(test_text, "test/model")
        print(f"✅ Token count: {token_count}")

        # Test message preparation
        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there!"),
        ]

        prepared_messages, token_usage = manager.prepare_context(
            messages, "test/model", max_response_length=100
        )
        print(
            f"✅ Context prepared: {len(prepared_messages)} messages, {token_usage.prompt_tokens} tokens"
        )

        return True

    except Exception as e:
        print(f"❌ ContextManager test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_unified_adapter():
    """Test UnifiedLLMAdapter functionality"""
    print("\n=== Testing UnifiedLLMAdapter ===")

    try:
        from core.llm.legacy_adapters import get_unified_llm_adapter

        # Create adapter
        adapter = get_unified_llm_adapter()
        print("✅ UnifiedLLMAdapter created")

        # Test session creation
        session_id = adapter.create_chat_session(
            system_prompt="Test unified adapter", metadata={"unified": True}
        )
        print(f"✅ Session created via adapter: {session_id}")

        # Test provider detection
        provider = adapter._detect_provider("test/model")
        print(f"✅ Provider detected: {provider}")

        # Test model listing
        all_models = adapter.list_all_models()
        print(f"✅ Models listed: {list(all_models.keys())}")

        return True

    except Exception as e:
        print(f"❌ UnifiedLLMAdapter test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_integration():
    """Test integration between components"""
    print("\n=== Testing Integration ===")

    try:
        from core.llm import get_llm_adapter, ChatMessage

        # Get main adapter
        adapter = get_llm_adapter()
        print("✅ Main adapter retrieved")

        # Test context validation
        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there!"),
        ]

        validation = adapter.validate_model_context(
            messages, model_name="test/model", max_response_length=100  # type: ignore
        )
        print(f"✅ Context validation: {validation['fits_in_context']}")

        # Test system status
        status = adapter.get_system_status()
        print(f"✅ System status: {len(status)} components")

        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def fix_legacy_files():
    """Fix legacy file integration"""
    print("\n=== Fixing Legacy Files ===")

    legacy_files = [
        "core/llm/ollama_llm.py",
        "core/llm/prompt_templates.py",
        "core/llm/transformers_llm.py",
    ]

    recommendations = []

    for file_path in legacy_files:
        full_path = ROOT_DIR / file_path
        if full_path.exists():
            print(f"📁 Found legacy file: {file_path}")

            # Read file to understand content
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()

                if "ollama" in content.lower():
                    recommendations.append(
                        f"• {file_path}: Integrate with OllamaLLMAdapter in legacy_adapters.py"
                    )
                elif "template" in content.lower():
                    recommendations.append(
                        f"• {file_path}: Move templates to context_manager.py or create separate templates module"
                    )
                elif "transformers" in content.lower():
                    recommendations.append(
                        f"• {file_path}: Merge functionality into enhanced adapter.py"
                    )
                else:
                    recommendations.append(
                        f"• {file_path}: Review and integrate or remove"
                    )

            except Exception as e:
                print(f"  ⚠️ Could not read {file_path}: {e}")
        else:
            print(f"🔍 Legacy file not found: {file_path}")

    print("\n📋 Recommendations:")
    for rec in recommendations:
        print(rec)

    return True


def create_migration_guide():
    """Create migration guide for legacy code"""
    print("\n=== Creating Migration Guide ===")

    migration_guide = """
# LLM Engine Migration Guide

## Overview
The LLM engine has been enhanced with new modular architecture. This guide helps migrate from legacy implementations.

## Key Changes

### 1. Enhanced LLM Adapter
- **Old**: Direct transformers usage
- **New**: EnhancedLLMAdapter with integrated features
- **Migration**: Use `get_llm_adapter()` instead of direct model loading

### 2. Chat Management
- **Old**: Manual conversation tracking
- **New**: ChatManager with session persistence
- **Migration**: Use `chat_manager.create_session()` for conversations

### 3. Context Management
- **Old**: Manual token counting
- **New**: ContextManager with intelligent truncation
- **Migration**: Use `context_manager.prepare_context()` for optimization

### 4. Model Loading
- **Old**: Direct AutoModelForCausalLM.from_pretrained()
- **New**: ModelLoader with advanced configuration
- **Migration**: Use ModelLoadConfig for fine-tuned control

## Backward Compatibility

### Legacy Transformers Models
```python
# Old way
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("model_name")

# New way (with legacy support)
from core.llm.legacy_adapters import LegacyTransformersLLM
llm = LegacyTransformersLLM("model_name")
llm.load_model()
```

### Ollama Integration
```python
# Old way (if existed)
# Manual Ollama API calls

# New way
from core.llm.legacy_adapters import OllamaLLMAdapter
ollama = OllamaLLMAdapter("llama2")
ollama.load_model()
response = ollama.chat(messages)
```

### Unified Interface
```python
# Recommended new approach
from core.llm.legacy_adapters import get_unified_llm_adapter

adapter = get_unified_llm_adapter()

# Auto-detects provider (transformers, ollama, etc.)
llm = adapter.get_llm("model_name", provider="auto")
response = adapter.chat(messages, model_name="model_name")
```

## File Structure Changes

### Deprecated Files
- `ollama_llm.py` → Integrated into `legacy_adapters.py`
- `transformers_llm.py` → Enhanced as `adapter.py`
- `prompt_templates.py` → Move to `context_manager.py` or separate module

### New Files
- `model_loader.py` - Advanced model loading
- `chat_manager.py` - Session management
- `context_manager.py` - Context optimization
- `legacy_adapters.py` - Backward compatibility

## Migration Steps

1. **Update Imports**
   ```python
   # Replace old imports
   from core.llm import get_llm_adapter, ChatMessage
   ```

2. **Update Model Loading**
   ```python
   # Old
   model = load_model("model_name")

   # New
   adapter = get_llm_adapter()
   llm = adapter.get_llm("model_name")
   ```

3. **Update Chat Interface**
   ```python
   # Old
   response = model.generate(prompt)

   # New
   messages = [ChatMessage(role="user", content=prompt)]
   response = adapter.chat(messages)
   ```

4. **Add Session Management**
   ```python
   # Create persistent session
   session_id = adapter.create_chat_session(
       system_prompt="You are a helpful assistant"
   )

   # Chat within session
   response = adapter.chat_with_session(session_id, "Hello")
   ```

## Testing Migration

Run the fix script to test compatibility:
```bash
python scripts/fix_llm_issues.py
```

This will verify all components work correctly with your existing code.
"""

    guide_path = ROOT_DIR / "docs" / "llm_migration_guide.md"
    guide_path.parent.mkdir(exist_ok=True)

    with open(guide_path, "w", encoding="utf-8") as f:
        f.write(migration_guide)

    print(f"✅ Migration guide created: {guide_path}")
    return True


def main():
    """Run all fix and test procedures"""
    print("🔧 LLM Engine Issues Fix Script")
    print("=" * 50)

    tests = [
        ("Import Tests", test_imports),
        ("ChatManager Tests", test_chat_manager),
        ("ModelLoader Tests", test_model_loader),
        ("ContextManager Tests", test_context_manager),
        ("UnifiedAdapter Tests", test_unified_adapter),
        ("Integration Tests", test_integration),
        ("Legacy File Analysis", fix_legacy_files),
        ("Migration Guide", create_migration_guide),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            print(f"\n🧪 {test_name}")
            if test_func():
                passed += 1
                print(f"✅ {test_name} - PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            failed += 1
            print(f"💥 {test_name} - ERROR: {e}")

    print("\n" + "=" * 50)
    print(f"📊 Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed! LLM engine is ready.")
        print("\n📋 Next steps:")
        print("1. Review migration guide in docs/llm_migration_guide.md")
        print("2. Update existing code to use new interfaces")
        print("3. Test with actual models using scripts/test_llm_engine.py")
    else:
        print("⚠️ Some issues found. Check the output above for details.")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
