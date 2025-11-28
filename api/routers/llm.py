# api/routers/llm.py
"""
LLM Management Router
"""

import asyncio
import logging
from fastapi import APIRouter, HTTPException
from core.llm.adapter import get_llm_adapter, reset_llm_adapter
from schemas.chat import (
    ChatModelsResponse,
    ChatModelInfo,
    ChatRequest,
    ChatResponse,
    ChatParameters,
)
from core.agents.tool_registry import ToolRegistry
from core.exceptions import ValidationError

logger = logging.getLogger(__name__)
router = APIRouter()
tool_registry = ToolRegistry()


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


@router.post("/llm/load")
async def load_llm_model(model_name: str, use_mock: bool = False):
    """Load an LLM model (mock fallback optional)."""
    try:
        adapter = get_llm_adapter(model_name=model_name, use_mock=use_mock)
        # Trigger load if supported
        if hasattr(adapter, "chat"):
            try:
                adapter.chat(messages=[{"role": "user", "content": "ping"}], max_length=8)
            except Exception:
                # Ignore load-time failures, rely on list for visibility
                pass
        return {
            "message": f"Load request for {model_name} accepted",
            "loaded_models": adapter.list_loaded_models(),
            "mock": use_mock,
        }
    except ValidationError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to load model: %s", e)
        raise HTTPException(500, f"Failed to load model: {str(e)}") from e


@router.post("/llm/unload/{model_name}")
async def unload_llm_model(model_name: str):
    """Unload specific LLM model (all if 'all')."""
    try:
        adapter = get_llm_adapter()

        if model_name == "all":
            adapter.unload_all()
            reset_llm_adapter()
            return {"message": "All LLM models unloaded"}

        # Selective unloading is not implemented in adapter
        return {
            "message": f"Unload request for {model_name} received (not implemented)",
            "loaded_models": adapter.list_loaded_models(),
        }

    except Exception as e:  # noqa: BLE001
        logger.error("Failed to unload model: %s", e)
        raise HTTPException(500, f"Failed to unload model: {str(e)}") from e


@router.get("/llm/status")
async def llm_status():
    """Report current LLM adapter status."""
    try:
        adapter = get_llm_adapter()
        loaded = adapter.list_loaded_models()
        return {
            "success": True,
            "loaded_models": loaded,
            "default_model": adapter.model_name if hasattr(adapter, "model_name") else None,
            "mock_mode": getattr(adapter, "use_mock", None),
        }
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to get LLM status: %s", e)
        raise HTTPException(500, f"Failed to get LLM status: {str(e)}") from e


@router.post("/llm/rag_chat", response_model=ChatResponse)
async def rag_augmented_chat(request: ChatRequest, use_mock: bool = False):
    """Chat with optional RAG augmentation before hitting the LLM."""
    try:
        adapter = get_llm_adapter(
            model_name=request.parameters.model if request.parameters else None,
            use_mock=use_mock,
        )

        # Retrieve context via rag_search tool if available
        rag_context = ""
        if tool_registry.is_tool_available("rag_search"):
            try:
                rag_fn = tool_registry.get_function("rag_search")
                if rag_fn:
                    if asyncio.iscoroutinefunction(rag_fn):
                        rag_payload = await rag_fn(
                            query=request.messages[-1].content, top_k=5
                        )
                    else:
                        rag_payload = rag_fn(
                            query=request.messages[-1].content, top_k=5
                        )
                if rag_payload and isinstance(rag_payload, dict):
                    snippets = rag_payload.get("results") or rag_payload.get("chunks") or []
                    if isinstance(snippets, list):
                        rag_context = "\n".join([str(s) for s in snippets][:5])
            except Exception as exc:  # noqa: BLE001
                logger.warning("RAG lookup failed, continuing without context: %s", exc)

        # Build messages with context injection
        messages = [
            {"role": m.role, "content": m.content} for m in request.messages
        ]
        if rag_context:
            messages.insert(
                -1,
                {
                    "role": "system",
                    "content": f"Relevant context:\n{rag_context}",
                },
            )

        params = request.parameters or ChatParameters()
        response = adapter.chat(
            messages=messages,
            max_length=params.max_length,
            temperature=params.temperature,
        )

        usage = getattr(response, "usage", {}) or {}
        return ChatResponse(  # type: ignore
            success=True,
            message=response.content,
            model_used=response.model_name,
            parameters=params,
            usage=usage,
        )
    except Exception as e:  # noqa: BLE001
        logger.error("RAG chat failed: %s", e)
        raise HTTPException(500, f"RAG chat failed: {str(e)}") from e
