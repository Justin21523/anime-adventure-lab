# ===== api/routers/chat.py =====
from fastapi import APIRouter, HTTPException, Depends
from api.schemas import ChatRequest, ChatResponse
from typing import List, Dict, Any, Optional
from ..dependencies import get_llm

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse, summary="Chat with LLM")
async def chat_completion(request: ChatRequest, llm: Any = Depends(get_llm)):
    """
    Chat completion endpoint.
    Delegates to the shared LLM engine (injected via dependencies).
    """
    try:
        # llm.chat expects a list of messages [{"role": "...", "content": "..."}]
        response_text = llm.chat(request.messages)

        return ChatResponse(
            message=response_text,
            model_used=request.model,
            usage={
                "prompt_tokens": sum(
                    len(msg["content"].split()) for msg in request.messages
                ),
                "completion_tokens": len(response_text.split()),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
