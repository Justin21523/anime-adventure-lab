# api/routers/rag.py
"""
RAG (Retrieval Augmented Generation) Router.

提供：
- 單一 / 批次 / 檔案上傳 的文件寫入
- 向量檢索 / 取得 raw context
- 透過 LLM 的 RAG 問答
- RAG engine 的統計、重建、清除、文件管理
"""

import asyncio
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, File, UploadFile

from core.rag.engine import get_rag_engine
from core.rag.document_processor import DocumentProcessor
from core.exceptions import RAGError
from schemas.rag import (
    RAGAddDocumentRequest,
    RAGAddDocumentResponse,
    RAGSearchRequest,
    RAGSearchResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    RAGBatchAddRequest,
    RAGBatchAddResponse,
    RAGStatsResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Ingestion endpoints
# ---------------------------------------------------------------------------


@router.post("/rag/add", response_model=RAGAddDocumentResponse)
async def add_document(request: RAGAddDocumentRequest):
    """
    Add a raw text document into the RAG index.

    - 由 engine 負責切 chunk + 建立向量。
    - 回傳的 chunks_created 為估算值（方便前端/監控用）。
    """
    try:
        rag_engine = get_rag_engine()
        metadata = request.metadata or {}

        added = rag_engine.add_document(
            doc_id=request.doc_id,
            content=request.content,
            metadata=metadata,
        )

        # Safe estimation of created chunks
        default_chunk_size = 512
        if request.parameters and getattr(request.parameters, "chunk_size", None):
            try:
                default_chunk_size = max(1, int(request.parameters.chunk_size))  # type: ignore[arg-type]
            except Exception:
                pass

        estimated_chunks = max(1, len(request.content) // default_chunk_size)

        return RAGAddDocumentResponse(  # type: ignore[call-arg]
            doc_id=request.doc_id,
            added=added,
            chunks_created=estimated_chunks,
            parameters=request.parameters,
        )

    except RAGError as e:
        raise HTTPException(500, f"RAG operation failed: {str(e)}")
    except Exception as e:
        logger.error(f"Add document failed: {e}")
        raise HTTPException(500, f"Internal error: {str(e)}")


@router.post("/rag/upload", response_model=RAGAddDocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    world_id: str = "default",
    tags: Optional[str] = None,
):
    """
    Upload a file, run it through DocumentProcessor, and add it to the RAG index.

    典型用在：上傳 PDF / TXT / DOCX 之類，抽出文字 + metadata 後寫入 RAG。
    """
    try:
        if not file.filename:
            raise HTTPException(400, "No filename provided")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(file.filename).suffix
        ) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            processor = DocumentProcessor()
            processed_doc = processor.process_file(
                file_path=tmp_file_path,
                metadata={
                    "world_id": world_id,
                    "original_filename": file.filename,
                    "content_type": file.content_type,
                    "tags": tags.split(",") if tags else [],
                },
            )

            rag_engine = get_rag_engine()
            added = rag_engine.add_document(
                doc_id=processed_doc.doc_id,
                content=processed_doc.content,
                metadata=processed_doc.metadata,
            )

            return RAGAddDocumentResponse(  # type: ignore[call-arg]
                doc_id=processed_doc.doc_id,
                added=added,
                chunks_created=None,
                parameters=None,
            )

        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_file_path)
            except OSError:
                logger.warning("Failed to delete temp file: %s", tmp_file_path)

    except RAGError as e:
        raise HTTPException(500, f"RAG operation failed: {str(e)}")
    except Exception as e:
        logger.error(f"Upload document failed: {e}")
        raise HTTPException(500, f"Upload error: {str(e)}")


# ---------------------------------------------------------------------------
# Search / context
# ---------------------------------------------------------------------------


@router.post("/rag/search", response_model=RAGSearchResponse)
async def search_documents(request: RAGSearchRequest):
    """Vector search over indexed chunks."""
    try:
        rag_engine = get_rag_engine()

        top_k = (
            request.parameters.top_k  # type: ignore[attr-defined]
            if request.parameters
            else 5
        )
        min_score = (
            request.parameters.min_score  # type: ignore[attr-defined]
            if request.parameters
            else 0.3
        )

        results = rag_engine.search(
            query=request.query,
            top_k=top_k,
            min_score=min_score,
        )

        return RAGSearchResponse(  # type: ignore[call-arg]
            query=request.query,
            results=[
                {
                    "doc_id": r.document.doc_id,
                    "content": r.document.content,
                    "score": r.score,
                    "metadata": r.document.metadata,
                }
                for r in results
            ],
            total_found=len(results),
            parameters=request.parameters,
        )

    except RAGError as e:
        raise HTTPException(500, f"RAG search failed: {str(e)}")
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(500, f"Search error: {str(e)}")


@router.post("/rag/context")
async def build_context(request: RAGSearchRequest):
    """
    Return raw RAG context and sources for downstream LLM/agents.

    這個 endpoint 給其他 router（像 /llm/rag_chat 或 agents）直接拿 context 用，
    不主動產生回答。
    """
    try:
        rag_engine = get_rag_engine()

        max_context_length = (
            request.parameters.max_context_length  # type: ignore[attr-defined]
            if request.parameters
            else 2000
        )
        top_k = (
            request.parameters.top_k  # type: ignore[attr-defined]
            if request.parameters
            else 5
        )

        context_data = rag_engine.generate_context(
            query=request.query,
            max_context_length=max_context_length,
            top_k=top_k,
        )

        return {
            "success": True,
            "context": context_data.get("context", ""),
            "sources": context_data.get("sources", []),
            "total_chars": context_data.get("total_chars", 0),
            "parameters": request.parameters,
        }
    except Exception as e:  # noqa: BLE001
        logger.error(f"Context generation failed: {e}")
        raise HTTPException(500, f"Context generation failed: {str(e)}")


# ---------------------------------------------------------------------------
# RAG QA
# ---------------------------------------------------------------------------


@router.post("/rag/ask", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    """
    RAG-enhanced question answering.

    流程：
    1. 用 RAG engine 取回最相關的 context。
    2. 把 context + 使用者問題包成一個嚴謹的 QA prompt 丟給 LLM。
    3. 回傳回答 + 原始 context + sources。
    """
    try:
        rag_engine = get_rag_engine()

        params = request.parameters
        max_context_length = (
            params.max_context_length  # type: ignore[attr-defined]
            if params
            else 2000
        )
        top_k = params.top_k if params else 5  # type: ignore[attr-defined]
        max_answer_tokens = (
            params.chunk_size if params and getattr(params, "chunk_size", None) else 512  # type: ignore[attr-defined]
        )

        context_data = rag_engine.generate_context(
            query=request.query,
            max_context_length=max_context_length,
            top_k=top_k,
        )

        if not context_data["context"]:
            return RAGQueryResponse(  # type: ignore[call-arg]
                query=request.query,
                answer="抱歉，我在知識庫中沒有找到足夠相關的資訊可以可靠地回答這個問題。",
                context="",
                sources=[],
                parameters=request.parameters,
            )

        # Lazy import LLM adapter to avoid circular deps on startup
        from core.llm.adapter import get_llm_adapter

        llm_adapter = get_llm_adapter()

        # 更嚴謹的 RAG prompt：限制只能根據 context 回答、資料不足要明講
        prompt = f"""你是一個嚴謹的知識助理，只能根據「相關資料」回答問題。

任務說明：
- 優先根據下面提供的「相關資料」回答問題。
- 不要編造資料中沒有明確提到的事實。
- 若資料不足以支撐明確結論，請明確標示「資料不足」，並說明目前能推論到哪裡。
- 回答時請以繁體中文，盡量清楚、有條理。

相關資料：
{context_data["context"]}

使用者問題：{request.query}

請依照以下格式回答：
1. 先給出直接答案（若能給出）。
2. 接著用 1~3 句說明你是根據哪些「相關資料」段落得出這個結論。
3. 若資料不足，另外加一段「資料不足說明」，說明缺少什麼資訊。"""

        try:
            # Prefer generate_text if available
            if hasattr(llm_adapter, "generate_text"):
                gen = llm_adapter.generate_text(  # type: ignore[call-arg]
                    prompt=prompt,
                    max_tokens=max_answer_tokens,
                    temperature=0.3,
                )
                answer = await gen if asyncio.iscoroutine(gen) else gen
            else:
                gen = llm_adapter.generate(  # type: ignore[call-arg]
                    prompt=prompt,
                    max_tokens=max_answer_tokens,
                    temperature=0.3,
                )
                result = await gen if asyncio.iscoroutine(gen) else gen
                answer = getattr(result, "content", None) or str(result)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"LLM generation failed: {exc}")
            answer = "目前無法產生回答，請稍後再試或稍微修改問題內容。"

        return RAGQueryResponse(  # type: ignore[call-arg]
            query=request.query,
            answer=str(answer),
            context=context_data["context"],
            sources=context_data["sources"],
            parameters=request.parameters,
        )

    except RAGError as e:
        raise HTTPException(500, f"RAG query failed: {str(e)}")
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(500, f"Query error: {str(e)}")


# ---------------------------------------------------------------------------
# Stats / maintenance
# ---------------------------------------------------------------------------


@router.get("/rag/stats", response_model=RAGStatsResponse)
async def get_rag_stats():
    """Get RAG engine statistics."""
    try:
        rag_engine = get_rag_engine()
        stats = rag_engine.get_stats()

        return RAGStatsResponse(  # type: ignore[call-arg]
            success=True,
            total_documents=stats.get("total_documents", 0),
            total_chunks=stats.get("total_chunks", 0),
            index_size=stats.get("index_size", 0),
            embedding_model=stats.get("embedding_model", ""),
            embedding_dim=stats.get("embedding_dim", 0),
            model_loaded=stats.get("model_loaded", False),
        )
    except Exception as e:
        logger.error(f"Get stats failed: {e}")
        raise HTTPException(500, f"Stats error: {str(e)}")


@router.post("/rag/rebuild")
async def rebuild_index():
    """Rebuild FAISS / BM25 index from current documents."""
    try:
        rag_engine = get_rag_engine()

        start = time.time()
        success = rag_engine.rebuild_index()
        duration = time.time() - start
        stats = rag_engine.get_stats() if success else {}

        return {
            "success": success,
            "message": "Index rebuild completed" if success else "Index rebuild failed",
            "time_taken_seconds": duration,
            "documents_processed": stats.get("total_documents", 0) if success else 0,
        }
    except Exception as e:
        logger.error(f"Rebuild failed: {e}")
        raise HTTPException(500, f"Rebuild error: {str(e)}")


@router.delete("/rag/clear")
async def clear_index():
    """Clear RAG index and all documents (dev / reset only)."""
    try:
        rag_engine = get_rag_engine()

        rag_engine.documents.clear()
        rag_engine.doc_id_map.clear()
        rag_engine.index = rag_engine._create_index()
        rag_engine.bm25 = None
        rag_engine.bm25_corpus = []

        return {"success": True, "message": "RAG index cleared"}
    except Exception as e:
        logger.error(f"Clear index failed: {e}")
        raise HTTPException(500, f"Clear error: {str(e)}")


# ---------------------------------------------------------------------------
# Document management
# ---------------------------------------------------------------------------


@router.get("/rag/documents")
async def list_documents(limit: int = 50, offset: int = 0):
    """List logical documents (grouping chunks by parent_doc_id)."""
    try:
        rag_engine = get_rag_engine()

        unique_docs = {}
        for doc_id, doc in rag_engine.documents.items():
            parent_id = doc.metadata.get("parent_doc_id", doc_id)
            if parent_id not in unique_docs:
                unique_docs[parent_id] = {
                    "doc_id": parent_id,
                    "title": doc.metadata.get("title", parent_id),
                    "chunks": 0,
                    "total_chars": 0,
                    "created_at": doc.created_at.isoformat()
                    if doc.created_at
                    else None,
                    "metadata": doc.metadata,
                }

            unique_docs[parent_id]["chunks"] += 1
            unique_docs[parent_id]["total_chars"] += len(doc.content)

        docs_list = list(unique_docs.values())
        total = len(docs_list)
        paginated_docs = docs_list[offset : offset + limit]

        return {
            "success": True,
            "documents": paginated_docs,
            "total": total,
            "limit": limit,
            "offset": offset,
        }
    except Exception as e:
        logger.error(f"List documents failed: {e}")
        raise HTTPException(500, f"List error: {str(e)}")


@router.delete("/rag/documents/{doc_id}")
async def delete_document(doc_id: str):
    """
    Delete a logical document from RAG index.

    作法：
    - 先把所有對應 chunks 從 rag_engine.documents 移除。
    - 再呼叫 rebuild_index() 重新建立 FAISS / BM25。
    """
    try:
        rag_engine = get_rag_engine()

        # Identify chunks for this logical document
        before_count = len(rag_engine.documents)
        chunks_to_remove: List[str] = []

        for chunk_id, doc in rag_engine.documents.items():
            parent_id = doc.metadata.get("parent_doc_id", chunk_id)
            if parent_id == doc_id or chunk_id == doc_id:
                chunks_to_remove.append(chunk_id)

        if not chunks_to_remove:
            raise HTTPException(404, f"Document {doc_id} not found")

        for chunk_id in chunks_to_remove:
            rag_engine.documents.pop(chunk_id, None)

        # Rebuild index to keep FAISS/doc_id_map consistent
        rag_engine.rebuild_index()
        after_count = len(rag_engine.documents)
        removed = before_count - after_count

        return {
            "success": True,
            "message": f"Deleted document {doc_id} ({removed} chunks)",
            "chunks_removed": removed,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete document failed: {e}")
        raise HTTPException(500, f"Delete error: {str(e)}")


@router.post("/rag/batch_add", response_model=RAGBatchAddResponse)
async def batch_add_documents(request: RAGBatchAddRequest):
    """
    Batch add documents to RAG index.

    request.documents 預期是 list[dict]，每個 dict 至少包含 doc_id / content。
    """
    rag_engine = get_rag_engine()
    success_count = 0
    failed_docs: List[str] = []

    for doc in request.documents:
        try:
            added = rag_engine.add_document(
                doc_id=doc["doc_id"],
                content=doc["content"],
                metadata=doc.get("metadata", {}),
            )
            if added:
                success_count += 1
            else:
                failed_docs.append(doc["doc_id"])
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Batch add failed for {doc['doc_id']}: {exc}")
            failed_docs.append(doc["doc_id"])

    return RAGBatchAddResponse(
        success=True,
        total_requested=len(request.documents),
        successful=success_count,
        failed=len(failed_docs),
        failed_docs=failed_docs,
        parameters=request.parameters,
    )
