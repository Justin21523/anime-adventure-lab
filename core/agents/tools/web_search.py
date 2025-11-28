# core/agent/tools/web_search.py
"""
Web Search Tool
- 預設使用 mock 模式，避免外部 API 需求
- 若提供 Brave API Key（BRAVE_API_KEY/BRAVE_SEARCH_API_KEY），可切換為真實查詢
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import asyncio

import requests

logger = logging.getLogger(__name__)

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
_OVERRIDE_API_KEY: Optional[str] = None
_MOCK_ENABLED: bool = True
_ENGINE_SINGLETON: Optional["WebSearchEngine"] = None


def _get_api_key(api_key: Optional[str] = None) -> Optional[str]:
    return (
        api_key
        or _OVERRIDE_API_KEY
        or os.getenv("BRAVE_SEARCH_API_KEY")
        or os.getenv("BRAVE_API_KEY")
    )


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    score: float = 0.8
    language: str = "en"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WebSearchEngine:
    def __init__(self, mock_enabled: bool = True, api_key: Optional[str] = None, search_engine_type: str = "mock"):
        self.mock_enabled = mock_enabled
        self.api_key = api_key or _get_api_key(api_key)
        self.search_engine_type = search_engine_type
        self.history: List[Dict[str, Any]] = []

    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        if self.mock_enabled or not self.api_key:
            return self._mock_search(query, max_results)
        return await self._brave_search(query, max_results)

    def _mock_search(self, query: str, max_results: int) -> List[SearchResult]:
        base_results = [
            SearchResult(title=f"{query} overview", url="https://example.com/overview", snippet=f"Summary about {query}", score=0.9),
            SearchResult(title=f"{query} tutorial", url="https://example.com/tutorial", snippet=f"Tutorial on {query}", score=0.85),
            SearchResult(title=f"{query} reference", url="https://example.com/reference", snippet=f"Reference for {query}", score=0.8),
        ]
        results = base_results[:max_results]
        self.history.append({"query": query, "results": len(results)})
        return results

    async def _brave_search(self, query: str, max_results: int) -> List[SearchResult]:
        resp = requests.get(
            BRAVE_ENDPOINT,
            headers={"X-Subscription-Token": self.api_key},
            params={"q": query, "count": max_results, "country": "us"},
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning("Brave search failed %s: %s", resp.status_code, resp.text)
            return self._mock_search(query, max_results)

        data = resp.json()
        web_results = data.get("web", {}).get("results", [])
        formatted = [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
                score=0.8,
                language=item.get("language", "en"),
            )
            for item in web_results
        ]
        results = formatted[:max_results]
        self.history.append({"query": query, "results": len(results)})
        return results

    async def search_and_summarize(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        results = await self.search(query, max_results)
        summary = "；".join([r.snippet for r in results])[:400] if results else ""
        return {
            "success": True,
            "query": query,
            "results": [r.to_dict() for r in results],
            "results_count": len(results),
            "summary": summary,
        }

    def get_search_history(self, max_items: int = 10) -> List[Dict[str, Any]]:
        return self.history[-max_items:]

    def get_search_stats(self) -> Dict[str, Any]:
        return {"total_searches": len(self.history), "last_query": self.history[-1]["query"] if self.history else None}


def get_search_engine() -> WebSearchEngine:
    global _ENGINE_SINGLETON
    if _ENGINE_SINGLETON is None:
        _ENGINE_SINGLETON = WebSearchEngine(mock_enabled=_MOCK_ENABLED, api_key=_get_api_key())
    return _ENGINE_SINGLETON


def configure_search_engine(mock_enabled: Optional[bool] = None, api_key: Optional[str] = None, search_engine_type: str = "mock") -> Dict[str, Any]:
    global _MOCK_ENABLED, _OVERRIDE_API_KEY, _ENGINE_SINGLETON
    if mock_enabled is not None:
        _MOCK_ENABLED = mock_enabled
    if api_key is not None:
        _OVERRIDE_API_KEY = api_key
    _ENGINE_SINGLETON = WebSearchEngine(mock_enabled=_MOCK_ENABLED, api_key=_get_api_key(api_key), search_engine_type=search_engine_type)
    return {"success": True, "mock_enabled": _MOCK_ENABLED, "search_engine_type": search_engine_type}


async def search(query: str, max_results: int = 5, api_key: Optional[str] = None) -> Dict[str, Any]:
    engine = get_search_engine()
    results = await engine.search(query, max_results)
    return {
        "success": True,
        "query": query,
        "results": [r.to_dict() for r in results],
        "results_count": len(results),
    }


async def search_and_summarize(query: str, max_results: int = 5) -> Dict[str, Any]:
    engine = get_search_engine()
    return await engine.search_and_summarize(query, max_results)


# Backward-compatible Brave API wrappers
async def brave_search(query: str, max_results: int = 5, api_key: Optional[str] = None) -> Dict[str, Any]:
    return await search(query, max_results=max_results, api_key=api_key)


async def brave_search_summary(
    query: str, max_results: int = 5, api_key: Optional[str] = None, llm_adapter=None
) -> Dict[str, Any]:
    result = await search(query, max_results=max_results, api_key=api_key)
    if not result.get("success"):
        return result
    summary = ""
    if llm_adapter and result.get("results"):
        try:
            snippets = "\n".join(
                [f"{i+1}. {r.get('title')}: {r.get('snippet')}" for i, r in enumerate(result["results"])]
            )
            prompt = (
                "You are a concise researcher. Summarize the following web results as bullet points:\n"
                f"Query: {query}\nResults:\n{snippets}\nSummary:"
            )
            llm_summary = await llm_adapter.generate_text(prompt, max_tokens=180, temperature=0.3)
            summary = llm_summary.strip()
        except Exception as e:
            logger.warning(f"LLM summary failed: {e}")
    return {**result, "summary": summary}
