# api/routers/llm.py
"""
LLM Management Router
"""

import logging
from fastapi import APIRouter, HTTPException
from core.llm.adapter import get_llm_adapter
from schemas.chat import ChatModelsResponse, ChatModelInfo

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/llm/models", response_model=ChatModelsResponse)
async def list_llm_models():
    """List available LLM models"""
    try:
        llm_adapter = get_llm_adapter()

        available_models = [
            ChatModelInfo(
                name="Qwen/Qwen-7B-Chat",
                description="Qwen 7B Chat model with Chinese support",
                languages=["en", "zh"],
                parameters="7B",
                context_length=8192,
                recommended=True,
                loaded="Qwen/Qwen-7B-Chat" in llm_adapter.list_loaded_models(),
            ),
            ChatModelInfo(
                name="meta-llama/Llama-2-7b-chat-hf",
                description="Llama 2 7B Chat model",
                languages=["en"],
                parameters="7B",
                context_length=4096,
                recommended=False,
                loaded="meta-llama/Llama-2-7b-chat-hf"
                in llm_adapter.list_loaded_models(),
            ),
        ]

        return ChatModelsResponse(  # type: ignore
            available_models=available_models,
            loaded_models=llm_adapter.list_loaded_models(),
            default_model="Qwen/Qwen-7B-Chat",
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to list LLM models: {str(e)}")


@router.post("/llm/unload/{model_name}")
async def unload_llm_model(model_name: str):
    """Unload specific LLM model"""
    try:
        llm_adapter = get_llm_adapter()

        if model_name == "all":
            llm_adapter.unload_all()
            return {"message": "All LLM models unloaded"}

        # Note: Current adapter doesn't support selective unloading
        return {
            "message": f"Unload request for {model_name} received (not implemented)"
        }

    except Exception as e:
        raise HTTPException(500, f"Failed to unload model: {str(e)}")
