# ===== api/routers/chat.py =====
"""
Text Chat API
LLM-based conversation interface
"""

import logging
from fastapi import APIRouter, HTTPException
from typing import List, Optional

from core.story.engine import get_story_engine
from core.exceptions import GameError, SessionNotFoundError, InvalidChoiceError
from schemas.game import (
    NewGameRequest,
    GameStepRequest,
    GameResponse,
    GameSessionSummary,
    GamePersonaInfo,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    """
    Text chat completion using LLM

    - **messages**: Conversation history with roles (system/user/assistant)
    - **max_length**: Maximum response length (50-1000)
    - **temperature**: Creativity level (0.1-2.0)
    - **model**: Optional model override
    """

    # Validate request
    if not request.messages:
        raise HTTPException(400, "Messages cannot be empty")

    if not (50 <= request.max_length <= 1000):
        raise HTTPException(400, "max_length must be between 50 and 1000")

    if not (0.1 <= request.temperature <= 2.0):
        raise HTTPException(400, "temperature must be between 0.1 and 2.0")

    try:
        # Get LLM adapter
        llm_adapter = get_llm_adapter()

        # Convert request messages to internal format
        formatted_messages = []
        for msg in request.messages:
            if hasattr(msg, "dict"):
                msg_dict = msg.dict()
            else:
                msg_dict = msg
            formatted_messages.append(msg_dict)

        # Generate response
        response = llm_adapter.chat(
            messages=formatted_messages,
            model_name=request.model,
            max_length=request.max_length,
            temperature=request.temperature,
            do_sample=request.temperature > 0.0,
        )

        return ChatResponse(
            message=response.content,
            model_used=response.model_name,
            usage=response.usage,
            metadata={
                "conversation_length": len(request.messages),
                "response_length": len(response.content),
                "parameters": {
                    "max_length": request.max_length,
                    "temperature": request.temperature,
                },
            },
        )

    except ValidationError as e:
        logger.warning(f"Chat validation error: {e}")
        raise HTTPException(400, f"Validation failed: {e.message}")

    except ModelError as e:
        logger.error(f"Chat model error: {e}")
        raise HTTPException(500, f"Model error: {e.message}")

    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(500, "Internal server error occurred")


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat completion (SSE)

    Returns Server-Sent Events for real-time response
    """

    from fastapi.responses import StreamingResponse
    import asyncio
    import json

    async def generate_stream():
        try:
            # For now, simulate streaming by chunking the complete response
            # Real implementation would use model.generate() with callback

            llm_adapter = get_llm_adapter()
            response = llm_adapter.chat(
                messages=[
                    msg.dict() if hasattr(msg, "dict") else msg
                    for msg in request.messages
                ],
                model_name=request.model,
                max_length=request.max_length,
                temperature=request.temperature,
            )

            # Simulate token-by-token streaming
            words = response.content.split()
            for i, word in enumerate(words):
                chunk = {
                    "content": word + " ",
                    "done": False,
                    "model": response.model_name,
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.05)  # Simulate typing delay

            # Send completion signal
            final_chunk = {"content": "", "done": True, "usage": response.usage}
            yield f"data: {json.dumps(final_chunk)}\n\n"

        except Exception as e:
            error_chunk = {"error": str(e), "done": True}
            yield f"data: {json.dumps(error_chunk)}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.get("/chat/models")
async def list_chat_models():
    """List available chat models"""
    return {
        "available_models": [
            {
                "name": "Qwen/Qwen-7B-Chat",
                "description": "Qwen 7B Chat model with Chinese support",
                "languages": ["en", "zh"],
                "parameters": "7B",
                "recommended": True,
            },
            {
                "name": "meta-llama/Llama-2-7b-chat-hf",
                "description": "Llama 2 7B Chat model",
                "languages": ["en"],
                "parameters": "7B",
                "recommended": False,
            },
        ],
        "loaded_models": get_llm_adapter().list_loaded_models(),
    }


@router.delete("/chat/models/{model_name}")
async def unload_chat_model(model_name: str):
    """Unload specific chat model to free memory"""
    try:
        llm_adapter = get_llm_adapter()

        if model_name == "all":
            llm_adapter.unload_all()
            return {"message": "All models unloaded successfully"}

        # Unload specific model (would need to implement in adapter)
        return {"message": f"Model {model_name} unload requested"}

    except Exception as e:
        logger.error(f"Model unload error: {e}")
        raise HTTPException(500, f"Failed to unload model: {str(e)}")


@router.post("/chat/system-prompt")
async def test_system_prompt(
    system_prompt: str,
    user_message: str,
    model: Optional[str] = None,
    temperature: float = 0.7,
):
    """
    Test system prompt with a user message

    Useful for prompt engineering and testing persona behavior
    """

    if not system_prompt.strip() or not user_message.strip():
        raise HTTPException(400, "Both system_prompt and user_message are required")

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        llm_adapter = get_llm_adapter()
        response = llm_adapter.chat(
            messages=messages, model_name=model, max_length=300, temperature=temperature  # type: ignore
        )

        return {
            "system_prompt": system_prompt,
            "user_message": user_message,
            "assistant_response": response.content,
            "model_used": response.model_name,
            "usage": response.usage,
        }

    except Exception as e:
        logger.error(f"System prompt test error: {e}")
        raise HTTPException(500, f"System prompt test failed: {str(e)}")
