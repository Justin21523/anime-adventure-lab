# api/routers/rag.py
"""
RAG (Retrieval Augmented Generation) Router
"""

import logging
from fastapi import APIRouter, HTTPException, File, UploadFile
from typing import List, Optional
from pathlib import Path
import tempfile
import os

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
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/rag/add", response_model=RAGAddDocumentResponse)
async def add_document(request: RAGAddDocumentRequest):
    """Add document to RAG index"""
    try:
        rag_engine = get_rag_engine()
        success = rag_engine.add_document(
            doc_id=request.doc_id,
            content=request.content,
            metadata=request.metadata or {},
        )

        return RAGAddDocumentResponse(  # type: ignore
            doc_id=request.doc_id, added=success, parameters=request.parameters
        )

    except RAGError as e:
        raise HTTPException(500, f"RAG operation failed: {str(e)}")
    except Exception as e:
        logger.error(f"Add document failed: {e}")
        raise HTTPException(500, f"Internal error: {str(e)}")


@router.post("/rag/upload", response_model=RAGAddDocumentResponse)
async def upload_document(
    file: UploadFile = File(...), world_id: str = "default", tags: Optional[str] = None
):
    """Upload and process document file"""
    try:
        # Validate file
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
            # Process document
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

            # Add to RAG
            rag_engine = get_rag_engine()
            success = rag_engine.add_document(
                doc_id=processed_doc.doc_id,
                content=processed_doc.content,
                metadata=processed_doc.metadata,
            )

            return RAGAddDocumentResponse(  # type: ignore
                doc_id=processed_doc.doc_id, added=success, parameters=None
            )

        finally:
            # Clean up temp file
            os.unlink(tmp_file_path)

    except RAGError as e:
        raise HTTPException(500, f"RAG operation failed: {str(e)}")
    except Exception as e:
        logger.error(f"Upload document failed: {e}")
        raise HTTPException(500, f"Upload error: {str(e)}")


@router.post("/rag/search", response_model=RAGSearchResponse)
async def search_documents(request: RAGSearchRequest):
    """Search documents in RAG index"""
    try:
        rag_engine = get_rag_engine()
        results = rag_engine.search(
            query=request.query,
            top_k=request.parameters.top_k,  # type: ignore
            min_score=request.parameters.min_score,  # type: ignore
        )

        return RAGSearchResponse(  # type: ignore
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


@router.post("/rag/ask", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    """RAG-enhanced question answering"""
    try:
        rag_engine = get_rag_engine()

        # Generate context from RAG
        context_data = rag_engine.generate_context(
            query=request.query,
            max_context_length=request.parameters.max_context_length,  # type: ignore
            top_k=request.parameters.top_k,  # type: ignore
        )

        if not context_data["context"]:
            return RAGQueryResponse(  # type: ignore
                query=request.query,
                answer="抱歉，我在知識庫中沒有找到相關信息。",
                context="",
                sources=[],
                parameters=request.parameters,
            )

        # Get LLM engine for generation
        from core.llm.adapter import get_llm_adapter

        llm_adapter = get_llm_adapter()

        # Build prompt with context
        prompt = f"""基於以下相關資料回答問題：

{context_data["context"]}

問題：{request.query}

請根據上述資料提供準確且有幫助的回答。如果資料不足以回答問題，請說明。"""

        # Generate answer
        response = await llm_adapter.generate(  # type: ignore
            prompt=prompt,
            max_tokens=request.parameters.chunk_size or 512,
            temperature=0.3,  # Lower temperature for factual responses
        )

        return RAGQueryResponse(  # type: ignore
            query=request.query,
            answer=response.content,
            context=context_data["context"],
            sources=context_data["sources"],
            parameters=request.parameters,
        )

    except RAGError as e:
        raise HTTPException(500, f"RAG query failed: {str(e)}")
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(500, f"Query error: {str(e)}")


@router.get("/rag/stats")
async def get_rag_stats():
    """Get RAG engine statistics"""
    try:
        rag_engine = get_rag_engine()
        stats = rag_engine.get_stats()

        return {"success": True, "stats": stats}

    except Exception as e:
        logger.error(f"Get stats failed: {e}")
        raise HTTPException(500, f"Stats error: {str(e)}")


@router.post("/rag/rebuild")
async def rebuild_index():
    """Rebuild RAG index"""
    try:
        rag_engine = get_rag_engine()
        success = rag_engine.rebuild_index()

        return {
            "success": success,
            "message": "Index rebuild completed" if success else "Index rebuild failed",
        }

    except Exception as e:
        logger.error(f"Rebuild failed: {e}")
        raise HTTPException(500, f"Rebuild error: {str(e)}")


@router.delete("/rag/clear")
async def clear_index():
    """Clear RAG index (development only)"""
    try:
        rag_engine = get_rag_engine()

        # Clear all documents
        rag_engine.documents.clear()
        rag_engine.doc_id_map.clear()

        # Recreate index
        rag_engine.index = rag_engine._create_index()
        rag_engine.bm25 = None
        rag_engine.bm25_corpus = []

        return {"success": True, "message": "RAG index cleared"}

    except Exception as e:
        logger.error(f"Clear index failed: {e}")
        raise HTTPException(500, f"Clear error: {str(e)}")


@router.get("/rag/documents")
async def list_documents(limit: int = 50, offset: int = 0):
    """List documents in RAG index"""
    try:
        rag_engine = get_rag_engine()

        # Get unique documents (group chunks by parent_doc_id)
        unique_docs = {}
        for doc_id, doc in rag_engine.documents.items():
            parent_id = doc.metadata.get("parent_doc_id", doc_id)
            if parent_id not in unique_docs:
                unique_docs[parent_id] = {
                    "doc_id": parent_id,
                    "title": doc.metadata.get("title", parent_id),
                    "chunks": 0,
                    "total_chars": 0,
                    "created_at": (
                        doc.created_at.isoformat() if doc.created_at else None
                    ),
                    "metadata": doc.metadata,
                }

            unique_docs[parent_id]["chunks"] += 1
            unique_docs[parent_id]["total_chars"] += len(doc.content)

        # Apply pagination
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
    """Delete document from RAG index"""
    try:
        rag_engine = get_rag_engine()

        # Find and remove all chunks for this document
        chunks_to_remove = []
        for chunk_id, doc in rag_engine.documents.items():
            parent_id = doc.metadata.get("parent_doc_id", chunk_id)
            if parent_id == doc_id or chunk_id == doc_id:
                chunks_to_remove.append(chunk_id)

        if not chunks_to_remove:
            raise HTTPException(404, f"Document {doc_id} not found")

        # Remove chunks
        for chunk_id in chunks_to_remove:
            if chunk_id in rag_engine.documents:
                del rag_engine.documents[chunk_id]

            # Remove from doc_id_map
            index_to_remove = None
            for idx, mapped_id in rag_engine.doc_id_map.items():  # type: ignore
                if mapped_id == chunk_id:
                    index_to_remove = idx
                    break

            if index_to_remove is not None:
                del rag_engine.doc_id_map[index_to_remove]

        # Update BM25
        rag_engine._update_bm25()

        return {
            "success": True,
            "message": f"Deleted document {doc_id} ({len(chunks_to_remove)} chunks)",
            "chunks_removed": len(chunks_to_remove),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete document failed: {e}")
        raise HTTPException(500, f"Delete error: {str(e)}")
