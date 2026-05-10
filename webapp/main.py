from __future__ import annotations

import hashlib
import io
import time
from typing import Optional

from fastapi import FastAPI, Form
from fastapi.responses import Response
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont


app = FastAPI(
    title="Anime Adventure Lab (Demo API)",
    version="0.1.0",
    docs_url="/api/v1/docs",
    openapi_url="/api/v1/openapi.json",
)


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "mode": "demo", "note": "No LLM/RAG/T2I models are loaded on the server demo."}


class TurnRequest(BaseModel):
    system: str = Field(default="You are a story engine.")
    user: str = Field(min_length=1, max_length=4000)
    seed: Optional[int] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


@app.post("/api/v1/turn")
def turn(req: TurnRequest):
    """
    Lightweight story turn demo:
    returns deterministic structured output (no LLM).
    """
    t0 = time.perf_counter()
    seed_str = str(req.seed) if req.seed is not None else "auto"
    blob = f"{req.system}\n{req.user}\n{seed_str}\n{req.temperature}".encode("utf-8")
    h = hashlib.sha256(blob).hexdigest()

    # deterministic "choices"
    choices = [
        f"Investigate the neon alley (id={h[:6]})",
        f"Talk to the masked merchant (id={h[6:12]})",
        f"Return to the safehouse (id={h[12:18]})",
    ]
    narrative = (
        "Demo narrative (no LLM running on server).\n"
        f"Seed={seed_str}\n\n"
        f"You said: {req.user.strip()[:600]}\n\n"
        "The story engine would normally call LLM/RAG here; this demo returns deterministic choices."
    )

    return {
        "narrative": narrative,
        "choices": choices,
        "meta": {"hash": h, "elapsed_ms": int((time.perf_counter() - t0) * 1000)},
    }


def _color_from_hash(h: bytes, offset: int) -> tuple[int, int, int]:
    return (h[offset] % 256, h[offset + 1] % 256, h[offset + 2] % 256)


@app.post("/api/v1/txt2img")
def txt2img(
    prompt: str = Form(...),
    seed: Optional[int] = Form(default=None),
    width: int = Form(default=640),
    height: int = Form(default=640),
):
    """
    Placeholder txt2img demo (no diffusion):
    returns a deterministic PNG derived from prompt+seed.
    """
    t0 = time.perf_counter()
    prompt = (prompt or "").strip()
    if not prompt:
        return Response(content=b"prompt required", status_code=400)

    width = max(128, min(int(width or 640), 1024))
    height = max(128, min(int(height or 640), 1024))
    seed_str = str(seed) if seed is not None else "auto"

    blob = f"{prompt}\n{seed_str}".encode("utf-8")
    dig = hashlib.sha256(blob).digest()
    c1 = _color_from_hash(dig, 0)
    c2 = _color_from_hash(dig, 3)
    c3 = _color_from_hash(dig, 6)

    img = Image.new("RGB", (width, height), c1)
    draw = ImageDraw.Draw(img)
    step = max(1, height // 28)
    for y in range(0, height, step):
        t = y / max(1, height - 1)
        r = int(c1[0] * (1 - t) + c2[0] * t)
        g = int(c1[1] * (1 - t) + c2[1] * t)
        b = int(c1[2] * (1 - t) + c2[2] * t)
        draw.rectangle([0, y, width, min(height, y + step)], fill=(r, g, b))
    draw.rectangle([18, 18, width - 18, height - 18], outline=c3, width=6)

    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    text = f"Anime Adventure Demo (txt2img)\nseed={seed_str}\n{ts}\n\n{prompt[:220]}"
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.multiline_text((28, 28), text, fill=(255, 255, 255), font=font, spacing=6)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    payload = buf.getvalue()
    return Response(
        content=payload,
        media_type="image/png",
        headers={"X-Elapsed-Ms": str(int((time.perf_counter() - t0) * 1000))},
    )

