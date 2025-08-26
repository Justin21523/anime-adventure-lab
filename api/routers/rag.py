# api/routers/rag.py
"""
RAG router (DI-based).
- No import-time side effects
- Engines injected from api/dependencies.py
- Unified request/response models
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any
import time
from datetime import datetime
from fastapi import (
    FastAPI,
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    Form,
    UploadFile,
    Depends,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import tempfile
import uuid
import fitz  # PyMuPDF for PDF parsing


from api.schemas import RAGRequest, RAGResponse, RAGSource
from ..dependencies import get_rag, get_llm  # shared singletons
from api.schemas import RAGRequest, RAGResponse, RAGSource  # keep your existing schemas
from core.rag.engine import DocumentMetadata


router = APIRouter(prefix="/rag", tags=["rag"])


#  ocal models for /retrieve -----
class RetrieveRequest(BaseModel):
    query: str
    world_id: Optional[str] = None
    top_k: int = Field(8, ge=1, le=50)
    alpha: float = Field(0.7, ge=0.0, le=1.0)


class RetrieveItem(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float
    section_title: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrieveResponse(BaseModel):
    query: str
    results: List[RetrieveItem]
    total_found: int
    processing_time_ms: float


# -Helpers
def _parse_pdf_bytes(content: bytes) -> str:
    """Best-effort PDF to text."""
    if fitz is None:
        raise HTTPException(
            status_code=400,
            detail="PDF support is not available (PyMuPDF not installed)",
        )
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp.flush()
            doc = fitz.open(tmp.name)
            text = "".join(page.get_text() for page in doc)
            doc.close()
            return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF parsing failed: {e}")


@router.get("/health")
async def rag_health(rag=Depends(get_rag)):
    """Lightweight health info for RAG engine."""
    # Engine may expose stats attributes; keep best-effort
    chunks = getattr(rag, "num_chunks", lambda: None)()
    docs = getattr(rag, "num_documents", lambda: None)()
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "rag_chunks": chunks,
        "rag_documents": docs,
    }


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    world_id: str = Form(...),
    title: Optional[str] = Form(None),
    license: str = Form("unspecified"),
    rag=Depends(get_rag),
):
    """Upload and index a document into RAG."""
    try:
        doc_id = f"{world_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        content = await file.read()

        if file.filename.lower().endswith(".pdf"):
            text_content = _parse_pdf_bytes(content)
        elif file.filename.lower().endswith((".md", ".txt")):
            text_content = content.decode("utf-8", errors="ignore")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        meta_dict = {
            "doc_id": doc_id,
            "title": title or file.filename,
            "source": file.filename,
            "world_id": world_id,
            "upload_time": datetime.now().isoformat(),
            "license": license,
        }
        metadata = (
            DocumentMetadata(**meta_dict) if DocumentMetadata else meta_dict
        )  # tolerate absent class

        result = rag.add_document(text_content, metadata)
        return {
            "success": True,
            "doc_id": doc_id,
            "chunks_added": (
                result.get("chunks_added", 0) if isinstance(result, dict) else result
            ),
            "message": f"Document '{file.filename}' uploaded and indexed",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_documents(request: RetrieveRequest, rag=Depends(get_rag)):
    """Retrieve relevant chunks with simple formatting."""
    start = time.time()
    try:
        results = rag.retrieve(
            query=request.query,
            world_id=request.world_id,
            top_k=request.top_k,
            alpha=request.alpha,
        )
        items: List[RetrieveItem] = []
        for r in results:
            items.append(
                RetrieveItem(
                    chunk_id=str(getattr(r, "chunk_id", "")),
                    doc_id=str(getattr(r, "doc_id", "")),
                    text=str(getattr(r, "text", "")),
                    score=round(float(getattr(r, "score", 0.0)), 4),
                    section_title=getattr(r, "section_title", None),
                    metadata=getattr(r, "metadata", {}) or {},
                )
            )
        return RetrieveResponse(
            query=request.query,
            results=items,
            total_found=len(items),
            processing_time_ms=round((time.time() - start) * 1000, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")


@router.post("/ask", response_model=RAGResponse)
async def rag_ask(request: RAGRequest, rag=Depends(get_rag), llm=Depends(get_llm)):
    """
    Answer a question using retrieved context.
    Keeps LLM call minimal so it works with your stub or real engine.
    """
    try:
        hits = rag.retrieve(
            query=request.question, world_id=None, top_k=request.top_k, alpha=0.7
        )
        sources: List[RAGSource] = []
        context_snippets: List[str] = []

        for r in hits:
            text = str(getattr(r, "text", ""))
            meta = getattr(r, "metadata", {}) or {}
            sources.append(
                RAGSource(
                    content=text[:200] + ("..." if len(text) > 200 else ""),
                    score=float(getattr(r, "score", 0.0)),
                    metadata=meta,
                )
            )
            context_snippets.append(text)

        # Compose a very small prompt (works with MinimalLLM)
        context = "\n\n".join(context_snippets[:5]) if context_snippets else ""
        messages = [
            {
                "role": "system",
                "content": "Answer concisely using the provided context. If unsure, say you don't know.",
            },
            {
                "role": "user",
                "content": f"Question: {request.question}\n\nContext:\n{context}",
            },
        ]
        answer = llm.chat(messages)

        return RAGResponse(
            answer=answer, sources=sources, model_used=getattr(llm, "model_name", "llm")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG ask failed: {e}")


@router.get("/stats")
async def rag_stats(rag=Depends(get_rag)):
    """Return internal stats if engine exposes them."""
    try:
        stats = getattr(rag, "get_stats", lambda: {})()
        return stats or {"status": "ok", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
