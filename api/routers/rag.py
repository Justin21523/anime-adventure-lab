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
from core.shared_cache import bootstrap_cache
from core.rag.engine import ChineseRAGEngine, DocumentMetadata

# Setup cache on module import
cache = bootstrap_cache()

router = APIRouter(prefix="/rag", tags=["RAG"])

# Initialize RAG engine
rag_engine = ChineseRAGEngine(APP_DIRS["RAG_INDEX"])


@router.post("/upload")
async def upload(file: UploadFile = File(...), world_id: str = Form("default")):
    # Stub: accept file and say indexed
    return {"world_id": world_id, "doc_id": f"{file.filename}", "chunks": 0}


@router.post("/retrieve")
def retrieve(query: str, world_id: str = "default", top_k: int = 8):
    # Stub: return empty hits
    return {"query": query, "world_id": world_id, "hits": []}


app = FastAPI(title="SagaForge RAG API", version="0.1.0")


class RetrieveRequest(BaseModel):
    query: str
    world_id: Optional[str] = None
    top_k: int = 8
    alpha: float = 0.7


class RetrieveResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total_found: int
    processing_time_ms: float


@app.get("/healthz")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "rag_chunks": len(rag_engine.chunks),
        "rag_documents": len(rag_engine.documents),
    }


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    world_id: str = Form(...),
    title: str = Form(None),
    license: str = Form("unspecified"),
):
    """Upload and index a document"""
    try:
        # Generate document ID
        doc_id = f"{world_id}_{uuid.uuid4().hex[:8]}"

        # Read file content
        content = await file.read()

        # Parse based on file type
        if file.filename.endswith(".pdf"):
            text_content = _parse_pdf(content)
        elif file.filename.endswith((".md", ".txt")):
            text_content = content.decode("utf-8")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        # Create metadata
        metadata = DocumentMetadata(
            doc_id=doc_id,
            title=title or file.filename,
            source=file.filename,
            world_id=world_id,
            upload_time=datetime.now().isoformat(),
            license=license,
        )

        # Add to RAG index
        result = rag_engine.add_document(text_content, metadata)

        return JSONResponse(
            {
                "success": True,
                "doc_id": doc_id,
                "chunks_added": result["chunks_added"],
                "message": f"Document '{file.filename}' uploaded and indexed successfully",
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_documents(request: RetrieveRequest):
    """Retrieve relevant documents with citations"""
    import time

    start_time = time.time()

    try:
        # Perform retrieval
        results = rag_engine.retrieve(
            query=request.query,
            world_id=request.world_id,
            top_k=request.top_k,
            alpha=request.alpha,
        )

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append(
                {
                    "chunk_id": result.chunk_id,
                    "doc_id": result.doc_id,
                    "text": result.text,
                    "score": round(result.score, 4),
                    "section_title": result.section_title,
                    "metadata": {
                        "title": result.metadata.get("title", ""),
                        "source": result.metadata.get("source", ""),
                        "world_id": result.metadata.get("world_id", ""),
                    },
                }
            )

        processing_time = (time.time() - start_time) * 1000

        return RetrieveResponse(
            query=request.query,
            results=formatted_results,
            total_found=len(formatted_results),
            processing_time_ms=round(processing_time, 2),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")


def _parse_pdf(content: bytes) -> str:
    """Parse PDF content to text"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(content)
            tmp_file.flush()

            doc = fitz.open(tmp_file.name)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()

            return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF parsing failed: {str(e)}")


print("[api] FastAPI routes registered")
print("Available endpoints:")
print("  GET  /healthz")
print("  POST /upload")
print("  POST /retrieve")
