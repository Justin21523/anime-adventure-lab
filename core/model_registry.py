"""Local model registry and path resolver.

All runtime model loading should resolve through AI_MODELS_ROOT first.  This
keeps model ownership in one shared warehouse and prevents accidental downloads
from remote model hubs during local startup or story generation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional


_LOCAL_MODEL_ONLY_ENV = "LOCAL_MODEL_ONLY"


_ALIASES = {
    "Qwen/Qwen2.5-7B-Instruct": "language/llm/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct": "language/llm/Qwen2.5-14B-Instruct",
    "BAAI/bge-m3": "language/sentence_transformers/bge-m3",
    "BAAI/bge-base-en-v1.5": "language/sentence_transformers/bge-m3",
    "BAAI/bge-reranker-v2-m3": "language/reranker/bge-reranker-large",
    "google/gemma-4-31B-it-GGUF": "language/vlm/gemma-4-31B-it-GGUF",
    "gemma-4-31B-it-GGUF": "language/vlm/gemma-4-31B-it-GGUF",
    "llava-hf/llava-1.5-7b-hf": "language/vlm/gemma-4-31B-it-GGUF",
    "Qwen/Qwen-VL-Chat": "language/vlm/gemma-4-31B-it-GGUF",
    "Qwen/Qwen2-VL-7B-Instruct": "language/vlm/gemma-4-31B-it-GGUF",
    "Salesforce/blip2-opt-2.7b": "language/vlm/gemma-4-31B-it-GGUF",
    "stabilityai/stable-diffusion-xl-base-1.0": (
        "diffusion/stable-diffusion/stable-diffusion-xl-base-1.0"
    ),
    "stabilityai/stable-diffusion-xl-base-1.0-inpainting-0.1": (
        "diffusion/stable-diffusion/stable-diffusion-xl-1.0-inpainting-0.1"
    ),
}


_KIND_DIRS = {
    "llm": ["language/llm", "llm", "video"],
    "embedding": ["language/sentence_transformers", "embeddings", "language/llm"],
    "reranker": ["language/reranker", "reranker"],
    "vlm": ["language/vlm", "vlm", "video"],
    "t2i": [
        "diffusion/stable-diffusion",
        "stable-diffusion",
        "diffusion/diffusers",
        "diffusion/CogView",
    ],
    "controlnet": ["diffusion/controlnet", "controlnet"],
    "safety": ["safety", "clip", "diffusion/clip"],
}


def models_root() -> Path:
    return Path(os.getenv("AI_MODELS_ROOT", "/mnt/c/ai_models")).expanduser()


def local_model_only() -> bool:
    return os.getenv(_LOCAL_MODEL_ONLY_ENV, "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _candidate_names(model_id: str) -> list[str]:
    raw = str(model_id or "").strip()
    slash_tail = raw.rsplit("/", 1)[-1] if "/" in raw else raw
    return list(
        dict.fromkeys(
            [
                raw,
                raw.replace("/", "__"),
                raw.replace("/", "--"),
                raw.replace("/", "_"),
                slash_tail,
                slash_tail.replace("_", "-"),
                slash_tail.replace("-", "_"),
            ]
        )
    )


def _existing_dir(path: Path) -> Optional[Path]:
    try:
        path = path.expanduser()
        if path.exists():
            return path.resolve()
    except Exception:
        return None
    return None


def _search_under(base_dirs: Iterable[Path], names: Iterable[str]) -> Optional[Path]:
    for base in base_dirs:
        for name in names:
            found = _existing_dir(base / name)
            if found is not None:
                return found
    return None


def resolve_model_path(model_id: str, kind: str = "generic", *, required: bool = True) -> str:
    """Resolve a model id or path to a local path under AI_MODELS_ROOT.

    If `LOCAL_MODEL_ONLY=1` (default), unresolved remote-looking ids raise
    FileNotFoundError.  Set `LOCAL_MODEL_ONLY=0` only when intentionally allowing
    remote downloads.
    """

    raw = str(model_id or "").strip()
    if not raw:
        raise ValueError("model_id is required")

    explicit = _existing_dir(Path(raw))
    if explicit is not None:
        return str(explicit)

    root = models_root()
    alias = _ALIASES.get(raw)
    if alias:
        found = _existing_dir(root / alias)
        if found is not None:
            return str(found)

    names = _candidate_names(raw)
    kind_dirs = [root / rel for rel in _KIND_DIRS.get(kind, [])]
    found = _search_under(kind_dirs + [root], names)
    if found is not None:
        return str(found)

    if not required or not local_model_only():
        return raw

    searched = ", ".join(str(p) for p in kind_dirs[:4] or [root])
    raise FileNotFoundError(
        f"Local model not found for {kind}: {raw}. "
        f"Put it under {root} or add an alias in core/model_registry.py. "
        f"Searched: {searched}"
    )


def resolve_model_path_optional(model_id: str, kind: str = "generic") -> str:
    return resolve_model_path(model_id, kind=kind, required=False)
