# api/routers/rag.py
"""
RAG (Retrieval Augmented Generation) Router
"""

import logging
from fastapi import APIRouter, HTTPException, File, UploadFile
from typing import List, Optional
from core.rag.engine import get_rag_engine
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
        raise HTTPException(500, f"RAG operation failed: {e.message}")


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
        raise HTTPException(500, f"RAG search failed: {e.message}")


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

        # Use LLM with RAG context
        from core.llm.adapter import get_llm_adapter

        llm_adapter = get_llm_adapter()

        system_prompt = f"""你是一個知識助手。請基於以下提供的上下文信息回答問題。

上下文信息：
{context_data['context']}

請基於上述信息回答問題。如果上下文中沒有相關信息，請明確說明。"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.query},
        ]

        response = llm_adapter.chat(
            messages=messages,  # type: ignore
            max_length=request.parameters.max_length,  # type: ignore
            temperature=request.parameters.temperature,  # type: ignore
        )

        return RAGQueryResponse(  # type: ignore
            query=request.query,
            answer=response.content,
            context=context_data["context"],
            sources=context_data["sources"],
            parameters=request.parameters,
            usage={
                "prompt_tokens": response.usage.get("prompt_tokens", 0),
                "completion_tokens": response.usage.get("completion_tokens", 0),
                "total_tokens": response.usage.get("total_tokens", 0),
            },
        )

    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(500, f"RAG query failed: {str(e)}")
