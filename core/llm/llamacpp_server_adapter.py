# core/llm/llamacpp_server_adapter.py
"""
llama.cpp Server Adapter — OpenAI-compatible API
Connects to a running llama.cpp server (e.g. llama-server --port 8080)
using its OpenAI-compatible chat completions endpoint.

This replaces the heavy HuggingFace Transformers backend, allowing
anime-adventure-lab to share the same llama.cpp instance as OpenClaw.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Union

import httpx

from .base import BaseLLM, ChatMessage, LLMResponse
from ..exceptions import ModelLoadError, ModelNotFoundError

logger = logging.getLogger(__name__)


class LlamaCppServerLLM(BaseLLM):
    """
    Adapter for llama.cpp server (OpenAI-compatible mode).

    Environment variables (optional):
        LLAMA_SERVER_URL   – base URL, default http://localhost:8080
        LLAMA_SERVER_MODEL – model name/ID for the /v1/chat/completions call
        LLAMA_SERVER_TIMEOUT – request timeout in seconds, default 300
    """

    def __init__(
        self,
        model_name: str,
        server_url: str = "http://localhost:8080",
        timeout: int = 300,
        **kwargs,
    ):
        super().__init__(model_name)
        import os

        self.server_url = os.environ.get("LLAMA_SERVER_URL", server_url).rstrip("/")
        self.timeout = int(os.environ.get("LLAMA_SERVER_TIMEOUT", str(timeout)))
        self._client: Optional[httpx.AsyncClient] = None
        self._loaded = False
        logger.info(
            "LlamaCppServerLLM initialised (lazy-connect) → %s | model=%s",
            self.server_url,
            self.model_name,
        )

    # ------------------------------------------------------------------ #

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.server_url,
                timeout=httpx.Timeout(self.timeout, connect=10),
            )
        return self._client

    # ------------------------------------------------------------------ #
    # BaseLLM interface
    # ------------------------------------------------------------------ #

    def load_model(self) -> None:
        """Non-blocking — we connect lazily on first request."""
        self._loaded = True
        logger.info(
            "LlamaCppServerLLM ready (server=%s, model=%s)",
            self.server_url,
            self.model_name,
        )

    def unload_model(self) -> None:
        self._loaded = False
        if self._client is not None:
            # Best-effort close (sync)
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self._client.aclose())
                loop.close()
            except Exception:
                pass
            self._client = None

    def is_available(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------ #
    # Generation helpers
    # ------------------------------------------------------------------ #

    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Sync generate — delegates to chat."""
        resp = self.chat(
            [ChatMessage(role="user", content=prompt)],
            max_length=max_length,
            temperature=temperature,
            **kwargs,
        )
        return resp.content if isinstance(resp, LLMResponse) else str(resp)

    def chat(
        self,
        messages: List[Union[ChatMessage, Dict[str, str]]],
        max_length: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        """
        Sync chat — safely runs async _chat whether or not an event loop is
        already running (avoids 'Cannot run the event loop while another loop
        is running' in FastAPI / uvicorn contexts).
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        coro = self._chat(
            messages, max_length=max_length, temperature=temperature, **kwargs
        )

        if loop is not None:
            # We're inside an async context — schedule on a thread so we
            # don't block the running loop, then wait for the result.
            import concurrent.futures

            futures = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = futures.submit(asyncio.run, coro)
            return future.result()  # type: ignore[return-value]
        else:
            return asyncio.run(coro)  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    # Async core (used by narrative engine)
    # ------------------------------------------------------------------ #

    async def _chat(
        self,
        messages: List[Union[ChatMessage, Dict[str, str]]],
        max_length: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        """Send chat completion request to llama.cpp server."""
        if not self.is_available():
            raise ModelNotFoundError(self.model_name)

        formatted = self.format_messages(messages)

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": m.role, "content": m.content} for m in formatted
            ],
            "max_tokens": max_length,
            "temperature": temperature,
            "top_p": kwargs.get("top_p", 0.9),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
        }

        # Optional streaming disabled by default for stability
        stream = kwargs.get("stream", False)
        if stream:
            payload["stream"] = True

        client = await self._ensure_client()

        try:
            if stream:
                return await self._chat_stream(client, payload, formatted)
            else:
                return await self._chat_non_stream(client, payload)

        except httpx.ConnectError as e:
            raise ModelNotFoundError(
                f"llama.cpp server unreachable at {self.server_url}: {e}"
            )
        except httpx.TimeoutException as e:
            raise ModelLoadError(
                self.model_name, f"Request timed out after {self.timeout}s: {e}"
            )
        except Exception as e:
            raise ModelLoadError(self.model_name, f"llama.cpp API error: {e}")

    async def _chat_non_stream(
        self, client: httpx.AsyncClient, payload: Dict[str, Any]
    ) -> LLMResponse:
        t0 = time.monotonic()
        resp = await client.post("/v1/chat/completions", json=payload)

        if resp.status_code != 200:
            body = resp.text[:500]
            raise ModelLoadError(self.model_name, f"HTTP {resp.status_code}: {body}")

        data = resp.json()
        content = (
            data.get("choices", [{}])[0].get("message", {}).get("content", "")
        )

        usage = data.get("usage", {})
        elapsed = time.monotonic() - t0

        return LLMResponse(
            content=content,
            model_name=self.model_name,
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            metadata={
                "elapsed_s": round(elapsed, 3),
                "server": self.server_url,
                "provider": "llama.cpp",
            },
        )

    async def _chat_stream(
        self,
        client: httpx.AsyncClient,
        payload: Dict[str, Any],
        formatted: List[ChatMessage],
    ) -> LLMResponse:
        """Streaming chat (returns full content after stream completes)."""
        t0 = time.monotonic()
        parts: List[str] = []
        usage: Dict[str, int] = {}

        async with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise ModelLoadError(
                    self.model_name, f"HTTP {resp.status_code}: {body[:500]}"
                )

            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = (
                        chunk.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                    if delta:
                        parts.append(delta)
                    usage_data = chunk.get("usage")
                    if usage_data:
                        usage = usage_data
                except json.JSONDecodeError:
                    continue

        elapsed = time.monotonic() - t0
        content = "".join(parts)

        return LLMResponse(
            content=content,
            model_name=self.model_name,
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            metadata={
                "elapsed_s": round(elapsed, 3),
                "server": self.server_url,
                "provider": "llama.cpp",
                "streamed": True,
            },
        )
