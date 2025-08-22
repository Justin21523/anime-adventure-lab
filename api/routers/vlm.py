from fastapi import APIRouter, UploadFile, File
router = APIRouter()

@router.post("/caption")
async def caption(file: UploadFile = File(...)):
    # Stub
    return {"caption": f"Stub caption for {file.filename}"}

@router.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Stub
    return {"labels": ["object:stub", "style:stub"]}
