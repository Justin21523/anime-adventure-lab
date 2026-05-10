# Anime-Adventure-lab

**LLM + RAG + T2I + VLM + LoRA** service stack for story-driven experiences.
FastAPI backend, Celery workers, and a React (Vite) WebUI, with AI_WAREHOUSE 3.0 storage roots.

---

## Key Principles

- **AI_WAREHOUSE 3.0**: No large models/datasets in the repo. Use split roots:
  - `AI_CACHE_ROOT` (HF/torch/XDG cache) → `/mnt/c/ai_cache`
  - `AI_MODELS_ROOT` (managed weights / LoRAs / checkpoints) → `/mnt/c/ai_models`
  - `AI_OUTPUT_ROOT` (runs / generated media / exports) → `/mnt/c/ai_output/anime-adventure-lab`
- **Low-VRAM Defaults**: Prefer device_map="auto", fp16/bf16, gradient checkpointing, and LoRA over full fine-tuning.
- **Security & Governance**: Never commit secrets; use `.env`. Optional NSFW/face-blur for public demos.

Architecture and folder layout follow the project plan and RAG/T2I modules described in our internal docs.
(health, LLM, RAG, T2I, VLM, LoRA, batch, and admin routers are scaffolded)
<!-- mirrors the structure in our architecture spec -->

---

## Quickstart

### 1) Environment (conda)

```bash
conda create -n ai_env python=3.10 -y
conda activate ai_env
pip install -r requirements.txt -r requirements-test.txt
cp .env.example .env  # if present
```

Set these in `.env` (see `.env.example`):

```
AI_CACHE_ROOT=/mnt/c/ai_cache
AI_MODELS_ROOT=/mnt/c/ai_models
AI_OUTPUT_ROOT=/mnt/c/ai_output/anime-adventure-lab
AI_DATASETS_ROOT=/mnt/c/ai_datasets/anime-adventure-lab
```

### 2) Run API

```bash
uvicorn api.main:app --reload
# -> http://localhost:8000/healthz
```

### 3) Run Worker (optional)

```bash
REDIS_URL=redis://localhost:6379/0 celery -A workers.celery_app:celery_app worker -l INFO
```

### 4) WebUI (React / Vite)

```bash
cd frontend/react
npm install
npm run dev
# -> http://localhost:3000
```

---

## Endpoints (MVP)

* `GET /healthz` – health check
* `POST /turn` – LLM turn (stub)
* `POST /upload`, `POST /retrieve` – RAG ingest/retrieve (stubs)
* `POST /gen_image` – T2I generate (stub path)
* `POST /caption`, `POST /analyze` – VLM (stubs)
* `POST /finetune/lora` – submit LoRA job (stub)
* `GET /batch/status/{job_id}` – job status (stub)
* `GET /models`, `GET /presets` – admin listings

> The API is intentionally minimal to pass smoke tests and wire the whole stack; each module has clear files to extend.

---

## Project Layout

```
api/          # FastAPI app, routers, middleware, dependencies
core/         # Business logic (LLM, RAG, Story, T2I, VLM, Train)
workers/      # Celery tasks and utilities
frontend/     # React (Vite) WebUI
configs/      # app/models/rag/train/presets
worldpacks/   # tiny samples (no large assets)
tests/        # unit + integration
docs/         # developer & deployment docs, notebooks
.github/      # CI/CD workflows and issue templates
```

---

## Tests & CI

* Run tests: `pytest -q`
* GitHub Actions CI: installs dependencies and runs tests on PRs.

---

## Roadmap (stages)

1. **Bootstrap** – `/healthz`, shared cache, basic WebUI
2. **LLM Core & Persona** – `/turn` MVP
3. **ZH RAG** – upload → chunk (hierarchical) → embed (bge-m3) → hybrid retrieve → rerank + citations
4. **Story Engine** – GameState + choice resolution
5. **T2I** – SD/SDXL pipelines, ControlNet, LoRA hot-swap
6. **VLM** – captions/tags and consistency write-back
7. **LoRA Fine-tuning** – /finetune/lora API + worker and evaluation
8. **Safety & License** – content filter, watermark, license records
9. **Perf & Export** – 4/8-bit, KV-cache, batch generation, export
10. **Release** – Docker Compose stack, optional Electron

---

License: Apache-2.0 (TBD).
