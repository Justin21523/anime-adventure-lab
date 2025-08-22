from fastapi import APIRouter
from pydantic import BaseModel
router = APIRouter()

class TurnIn(BaseModel):
    user: str
    persona: str | None = None
    state: dict | None = None

@router.post("/turn")
def turn(body: TurnIn):
    # Minimal echo stub (LLMAdapter comes later)
    return {"reply": f"Echo: {body.user}", "persona": body.persona or "default", "citations": []}

@router.get("/persona")
def persona_list(): return {"personas": ["default"]}
