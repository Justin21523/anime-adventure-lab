# tests/unit/test_rag.py
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tempfile
import uuid
import fitz  # PyMuPDF for PDF parsing

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append("../..")

from api.dependencies import AI_CACHE_ROOT, APP_DIRS
from core.rag.engine import ChineseRAGEngine, DocumentMetadata

router = APIRouter()
# Initialize RAG engine
rag_engine = ChineseRAGEngine(APP_DIRS["RAG_INDEX"])


def test_retrieve(client):
    r = client.post("/retrieve", params={"query": "q"})
    assert r.status_code == 200


def run_smoke_tests():
    """Run comprehensive smoke tests"""
    print("\n" + "=" * 50)
    print("🧪 STAGE 3 SMOKE TESTS")
    print("=" * 50)

    # Test 1: Document ingestion
    print("\n📄 Test 1: Document Ingestion")
    test_document = """
# 測試世界觀文檔

## 背景設定
這個世界是一個賽博朋克風格的未來都市，名為「新台北」。在這裡，科技與傳統文化交融，形成獨特的社會景觀。

## 主要角色
### 艾莉絲 (Alice Chen)
- 職業：網路安全專家
- 年齡：28歲
- 特徵：擁有改造的機械手臂，專精於駭客技術

### 張博士 (Dr. Zhang)
- 職業：人工智慧研究員
- 年齡：45歲
- 特徵：神秘的過去，與禁忌實驗有關

## 重要地點
### 下城區
充滿霓虹燈的商業區，是駭客和商人的聚集地。這裡的咖啡廳和酒吧往往是情報交易的場所。

### 上城區
企業精英居住的高檔區域，擁有最先進的安全系統和監控網路。
    """

    metadata = DocumentMetadata(
        doc_id="test_world_001",
        title="測試世界觀",
        source="smoke_test.md",
        world_id="neo_taipei",
        upload_time=datetime.now().isoformat(),
        license="test",
    )

    result = rag_engine.add_document(test_document, metadata)
    print(f"✅ Retrieval latency: {latency_ms:.1f}ms (target: <2000ms)")

    if latency_ms > 2000:
        print("⚠️  Warning: Latency exceeds target")

    # Test 5: Index persistence
    print("\n💾 Test 5: Index Persistence")
    original_chunk_count = len(rag_engine.chunks)

    # Save and reload
    rag_engine._save_index()
    new_engine = ChineseRAGEngine(APP_DIRS["RAG_INDEX"])

    if len(new_engine.chunks) == original_chunk_count:
        print(f"✅ Index persistence verified: {len(new_engine.chunks)} chunks")
    else:
        print(
            f"❌ Index persistence failed: {len(new_engine.chunks)} vs {original_chunk_count}"
        )

    print("\n🎉 All smoke tests completed!")
    return True


# Run the tests
smoke_test_result = run_smoke_tests()

# ================================
# Cell 7: API Server Test & Git Steps

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🚀 STARTING FASTAPI SERVER")
    print("=" * 50)
    print("Run the following commands to test the API:")
    print()
    print("# Terminal 1: Start server")
    print("uvicorn stage3_zh_rag:app --host 0.0.0.0 --port 8000 --reload")
    print()
    print("# Terminal 2: Test endpoints")
    print("curl http://localhost:8000/healthz")
    print()
    print("# Upload test document")
    print('curl -X POST "http://localhost:8000/upload" \\')
    print('  -F "file=@test_doc.md" \\')
    print('  -F "world_id=neo_taipei" \\')
    print('  -F "title=測試文檔" \\')
    print('  -F "license=CC-BY-4.0"')
    print()
    print("# Retrieve documents")
    print('curl -X POST "http://localhost:8000/retrieve" \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"query": "艾莉絲的特徵", "world_id": "neo_taipei", "top_k": 3}\'')
    print()

# Git workflow instructions
print("\n" + "=" * 50)
print("📝 GIT WORKFLOW - STAGE 3")
print("=" * 50)
print()
print("# 1. Create feature branch")
print("git checkout -b feature/stage3-zh-rag-engine")
print()
print("# 2. Create directory structure")
print("mkdir -p core/rag api/routers tests/unit")
print()
print("# 3. Split this implementation into proper files:")
print("# - core/rag/engine.py (ChineseRAGEngine)")
print("# - core/rag/chunkers.py (ChineseHierarchicalChunker)")
print("# - api/routers/rag.py (FastAPI routes)")
print("# - tests/unit/test_rag.py (smoke tests)")
print()
print("# 4. Add minimal requirements.txt")
print("echo 'sentence-transformers>=2.2.0' >> requirements.txt")
print("echo 'faiss-cpu>=1.7.0' >> requirements.txt")
print("echo 'rank-bm25>=0.2.2' >> requirements.txt")
print("echo 'opencc-python-reimplemented>=1.1.0' >> requirements.txt")
print("echo 'PyMuPDF>=1.23.0' >> requirements.txt")
print("echo 'fastapi>=0.104.0' >> requirements.txt")
print("echo 'python-multipart>=0.0.6' >> requirements.txt")
print()
print("# 5. Commit implementation")
print("git add .")
print('git commit -m "feat(rag): implement Chinese RAG engine with hybrid retrieval"')
print()
print('git commit -m "feat(api): add /upload and /retrieve endpoints with citations"')
print()
print("# 6. Merge to develop")
print("git checkout develop")
print("git merge --no-ff feature/stage3-zh-rag-engine")
print()

# ================================
# Additional utilities and config examples

print("\n" + "=" * 50)
print("⚙️  CONFIGURATION EXAMPLES")
print("=" * 50)
