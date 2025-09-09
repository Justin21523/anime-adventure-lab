#!/usr/bin/env python3
# scripts/test_rag_basic.py

"""
RAG引擎基礎功能煙霧測試
Usage: python scripts/test_rag_basic.py
"""

import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

# Shared cache bootstrap
import torch

AI_CACHE_ROOT = os.getenv("AI_CACHE_ROOT", "/mnt/ai_warehouse/cache")
for k, v in {
    "HF_HOME": f"{AI_CACHE_ROOT}/hf",
    "TRANSFORMERS_CACHE": f"{AI_CACHE_ROOT}/hf/transformers",
    "HF_DATASETS_CACHE": f"{AI_CACHE_ROOT}/hf/datasets",
    "HUGGINGFACE_HUB_CACHE": f"{AI_CACHE_ROOT}/hf/hub",
    "TORCH_HOME": f"{AI_CACHE_ROOT}/torch",
}.items():
    os.environ[k] = v
    Path(v).mkdir(parents=True, exist_ok=True)

print(f"[Cache] {AI_CACHE_ROOT} | GPU: {torch.cuda.is_available()}")


def test_rag_engine():
    """Test RAG engine basic functionality"""
    print("\n=== Testing RAG Engine ===")

    try:
        from core.rag.engine import get_rag_engine

        # Get RAG engine
        rag_engine = get_rag_engine()
        print("✓ RAG engine initialized")

        # Test document addition
        test_docs = [
            {
                "doc_id": "test_1",
                "content": "人工智能是計算機科學的一個分支，致力於創建能夠執行通常需要人類智能的任務的系統。",
                "metadata": {"topic": "AI", "language": "zh"},
            },
            {
                "doc_id": "test_2",
                "content": "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
                "metadata": {"topic": "ML", "language": "en"},
            },
            {
                "doc_id": "test_3",
                "content": "深度學習使用神經網絡來模擬人腦的工作方式，是現代AI發展的重要技術。",
                "metadata": {"topic": "DL", "language": "zh"},
            },
        ]

        for doc in test_docs:
            success = rag_engine.add_document(
                doc_id=doc["doc_id"], content=doc["content"], metadata=doc["metadata"]
            )
            print(f"✓ Added document {doc['doc_id']}: {success}")

        # Test search
        query = "什麼是人工智能？"
        results = rag_engine.search(query, top_k=3)
        print(f"\n✓ Search results for '{query}': {len(results)} found")

        for i, result in enumerate(results):
            print(f"  {i+1}. Score: {result.score:.3f} | Doc: {result.document.doc_id}")
            print(f"     Content: {result.document.content[:50]}...")

        # Test context generation
        context_data = rag_engine.generate_context(query, max_context_length=500)
        print(f"\n✓ Context generated: {len(context_data['context'])} chars")
        print(f"  Sources: {len(context_data['sources'])}")

        # Test statistics
        stats = rag_engine.get_stats()
        print(f"\n✓ Engine stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        return True

    except Exception as e:
        print(f"✗ RAG engine test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_document_processor():
    """Test document processor"""
    print("\n=== Testing Document Processor ===")

    try:
        from core.rag.document_processor import DocumentProcessor

        processor = DocumentProcessor()
        print("✓ Document processor initialized")

        # Test text processing
        test_text = """
        這是一個測試文檔。
        This is a test document.

        包含中英文混合內容，用於測試文檔處理功能。
        It contains mixed Chinese and English content for testing.
        """

        processed_doc = processor.process_text(text=test_text, metadata={"test": True})

        print(f"✓ Text processed:")
        print(f"  Doc ID: {processed_doc.doc_id}")
        print(f"  Content length: {len(processed_doc.content)}")
        print(f"  File type: {processed_doc.file_type}")

        # Test supported formats
        formats = processor.get_supported_formats()
        print(f"\n✓ Supported formats: {list(formats.keys())}")

        return True

    except Exception as e:
        print(f"✗ Document processor test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_vector_store():
    """Test vector store"""
    print("\n=== Testing Vector Store ===")

    try:
        from core.rag.vector_store import VectorStore
        import numpy as np

        # Create vector store
        store = VectorStore(dimension=128)
        print("✓ Vector store created")

        # Test adding vectors
        test_vectors = np.random.randn(3, 128).astype(np.float32)
        doc_ids = ["vec_1", "vec_2", "vec_3"]

        for i, (doc_id, vector) in enumerate(zip(doc_ids, test_vectors)):
            index_id = store.add_vector(
                doc_id=doc_id, vector=vector, metadata={"test": True, "index": i}
            )
            print(f"✓ Added vector {doc_id}: index_id={index_id}")

        # Test search
        query_vector = np.random.randn(128).astype(np.float32)
        results = store.search(query_vector, top_k=2)

        print(f"\n✓ Vector search results: {len(results)} found")
        for index_id, score, metadata in results:
            print(f"  Index: {index_id}, Score: {score:.3f}, Doc: {metadata.doc_id}")

        # Test statistics
        stats = store.get_stats()
        print(f"\n✓ Vector store stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        return True

    except Exception as e:
        print(f"✗ Vector store test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_embedding_models():
    """Test embedding models"""
    print("\n=== Testing Embedding Models ===")

    try:
        from core.rag.embeddings import get_embedding_manager

        manager = get_embedding_manager()
        print("✓ Embedding manager initialized")

        # Test encoding
        test_texts = [
            "這是中文測試文本",
            "This is English test text",
            "混合中英文 mixed content test",
        ]

        embeddings = manager.encode(test_texts)
        print(f"✓ Encoded {len(test_texts)} texts")
        print(f"  Embedding shape: {embeddings.shape}")
        print(f"  Dimension: {manager.get_dimension()}")

        return True

    except Exception as e:
        print(f"✗ Embedding models test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_api_imports():
    """Test API components can be imported"""
    print("\n=== Testing API Imports ===")

    try:
        from api.routers.rag import router as rag_router

        print("✓ RAG router imported")

        from schemas.rag import RAGAddDocumentRequest, RAGSearchRequest

        print("✓ RAG schemas imported")

        # Test schema validation
        request = RAGAddDocumentRequest(
            doc_id="test", content="test content", metadata={"test": True}
        )
        print(f"✓ Schema validation passed: {request.doc_id}")

        return True

    except Exception as e:
        print(f"✗ API imports test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("🚀 Starting RAG Engine Smoke Tests")
    print("=" * 50)

    tests = [
        test_api_imports,
        test_document_processor,
        test_vector_store,
        test_embedding_models,
        test_rag_engine,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test_func.__name__} crashed: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed! RAG engine is ready.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
