#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, Dict

import torch

from core.config import get_config
from core.shared_cache import bootstrap_cache
from core.llm.adapter import EnhancedLLMAdapter, ModelLoadConfig
from core.t2i.engine import get_t2i_engine


def _print_gpu_info() -> None:
    print("== GPU ==")
    print(f"cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        return
    count = torch.cuda.device_count()
    print(f"device_count={count}")
    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / (1024**3)
        print(f"- cuda:{i} name={props.name} vram_gb={total_gb:.1f}")


def _smoke_llm() -> None:
    cfg = get_config()
    model_name = (
        os.getenv("MODEL_CHAT_MODEL")
        or str(getattr(cfg.model, "chat_model", "") or "").strip()
        or "Qwen/Qwen2.5-7B-Instruct"
    )

    print("\n== LLM ==")
    print(f"model={model_name}")

    adapter = EnhancedLLMAdapter()
    load_cfg = ModelLoadConfig(
        model_name=model_name,
        device_map=str(getattr(cfg.model, "device_map", "auto") or "auto"),
        torch_dtype=str(getattr(cfg.model, "torch_dtype", "float16") or "float16"),
        use_quantization=bool(getattr(cfg.model, "use_4bit_loading", True)),
        quantization_bits=4,
        trust_remote_code=True,
    )
    llm = adapter.get_llm(model_name=model_name, model_type="auto", load_config=load_cfg)
    llm.load_model()

    resp = llm.chat(
        messages=[
            {"role": "system", "content": "你是一個簡潔、可靠的遊戲敘事者。"},
            {"role": "user", "content": "用一句話說明：今天要去探索哪裡？"},
        ],
        max_length=64,
        temperature=0.2,
    )
    text = (resp.content or "").strip()
    print("ok")
    print(text[:240])


async def _smoke_t2i_async() -> None:
    cfg = get_config()
    model_id = (
        os.getenv("MODEL_DEFAULT_SD_MODEL")
        or str(getattr(cfg.model, "default_sd_model", "") or "").strip()
        or "stabilityai/stable-diffusion-xl-base-1.0"
    )

    print("\n== T2I ==")
    print(f"model={model_id}")

    engine = get_t2i_engine()
    engine.mock_generation = False

    request: Dict[str, Any] = {
        "prompt": "anime style, cinematic lighting, a fantasy town street at sunset, high detail",
        "negative_prompt": "lowres, blurry, bad anatomy, text, watermark",
        "width": 768,
        "height": 768,
        "num_inference_steps": 10,
        "guidance_scale": 6.0,
        "seed": 123456,
        "batch_size": 1,
        "session_id": "gpu_smoke",
        "model_id": model_id,
    }
    result = await engine.txt2img(request)
    meta = result.get("metadata", {}) or {}
    paths = meta.get("output_paths") or []
    print("ok")
    if paths:
        print(f"output={paths[0]}")
    else:
        print("output=<none>")


def main() -> int:
    bootstrap_cache()
    _print_gpu_info()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. (請確認驅動/torch CUDA 版本)")
        return 2

    try:
        _smoke_llm()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: LLM smoke failed: {exc}")
        return 3

    try:
        asyncio.run(_smoke_t2i_async())
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: T2I smoke failed: {exc}")
        return 4

    print("\nALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

