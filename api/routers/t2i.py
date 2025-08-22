from fastapi import APIRouter
from pydantic import BaseModel
router = APIRouter()

class GenIn(BaseModel):
    prompt: str
    negative: str | None = None
    width: int = 768
    height: int = 768
    steps: int = 25
    seed: int | None = None
    lora_ids: list[str] = []

@router.post("/gen_image")
def gen_image(body: GenIn):
    # Stub: not generating; just return a fake path
    return {"image_path": "/warehouse/cache/outputs/saga-forge/fake.png", "meta": body.model_dump()}
