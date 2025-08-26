# api/routers/vqa.py
"""
Visual Question Answering router (DI-based).
Saves image to a temp file and calls vlm.vqa(path, question) if available.
"""
from __future__ import annotations
from fastapi import APIRouter, HTTPException, File, UploadFile, Depends
from api.schemas import VQARequest, VQAResponse
import base64, io, tempfile, os
from PIL import Image

from ..dependencies import get_vlm

router = APIRouter(tags=["vqa"])


def _load_image_from_request(req: VQARequest) -> Image.Image:
    if req.image_base64:
        data = base64.b64decode(req.image_base64)
        return Image.open(io.BytesIO(data)).convert("RGB")
    if req.image_path:
        return Image.open(req.image_path).convert("RGB")
    raise HTTPException(
        status_code=400, detail="Either image_path or image_base64 required"
    )


@router.post(
    "/vqa", response_model=VQAResponse, summary="Answer a question about an image"
)
async def visual_question_answering(request: VQARequest, vlm=Depends(get_vlm)):
    try:
        pil = _load_image_from_request(request)

        # Many engines take a file path; save temp and cleanup
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pil.save(tmp.name)
            tmp_path = tmp.name
        try:
            # Prefer vqa(image=..., question=...), fallback to vqa(path, question)
            try:
                answer = vlm.vqa(image=pil, question=request.question)
            except Exception:
                answer = vlm.vqa(
                    tmp_path, question=request.question
                )  # keep signature tolerant
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        return VQAResponse(
            question=request.question,
            answer=str(answer),
            confidence=0.85,
            model_used=request.model,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
