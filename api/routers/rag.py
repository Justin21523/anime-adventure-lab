from fastapi import APIRouter, UploadFile, File, Form
router = APIRouter()

@router.post("/upload")
async def upload(file: UploadFile = File(...), world_id: str = Form("default")):
    # Stub: accept file and say indexed
    return {"world_id": world_id, "doc_id": f"{file.filename}", "chunks": 0}

@router.post("/retrieve")
def retrieve(query: str, world_id: str = "default", top_k: int = 8):
    # Stub: return empty hits
    return {"query": query, "world_id": world_id, "hits": []}
