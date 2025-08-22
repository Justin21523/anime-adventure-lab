from fastapi import APIRouter
router = APIRouter()

@router.get("/models")
def models(): return {"llm": [], "t2i": [], "vlm": [], "lora": []}

@router.get("/presets")
def presets(): return {"styles": []}
