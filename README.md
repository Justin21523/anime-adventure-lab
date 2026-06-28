# Anime Adventure Lab

Portfolio demo for an AI-driven anime adventure platform.

Anime Adventure Lab combines a story engine, world/lore management, RAG retrieval, agent tools, and text-to-image hooks into one playable visual-novel style workbench. The public demo is designed to be stable and interview-friendly: it uses deterministic mock/sample data, so reviewers can understand the architecture without downloading models or using a GPU.

> 中文摘要：這是一個 AI 故事冒險平台作品集專案，展示 Story Engine、RAG、Agent、T2I、WorldPack 與背景任務流程。公開展示版以穩定 mock demo 為主；真模型/GPU 推論保留為進階本機模式。

## Demo Strategy

- **Static portfolio demo:** `portfolio-web/`
  - Works on GitHub Pages, Vercel, Netlify, or any static host.
  - Includes scenario switching, visual novel screen, RAG/Agent/T2I trace panels, and a 90-second recording flow.
- **React workbench:** `frontend/react/`
  - Main product UI for story sessions, Visual Novel mode, World Studio, RAG ingestion, and job progress.
- **FastAPI backend:** `api/`
  - Mock-safe API for smoke testing and local demos; full AI model loading is optional.

## What It Demonstrates

- **Story Engine:** session state, choices, turn history, memory summaries, relationship/state deltas.
- **WorldPack system:** world metadata, characters, visual style defaults, player templates, world-scoped RAG.
- **RAG pipeline:** upload, chunking, metadata, world filters, retrieval/rerank flags, citation traces.
- **Agent layer:** tool catalog, assisted decisions, world state checks, review queue/writeback suggestions.
- **T2I integration:** scene prompt generation, LoRA/world style hooks, async image jobs, mock-safe fallbacks.
- **Ops architecture:** FastAPI routers, Celery-compatible jobs, Docker deployment, AI_WAREHOUSE storage roots.

## Architecture

```text
portfolio-web/        Static interview demo and scenario showcase
frontend/react/       React + Vite visual novel/workbench UI
api/                  FastAPI app, routers, dependencies, middleware
core/                 Story, RAG, Agent, T2I, VLM, training, monitoring logic
schemas/              Shared Pydantic request/response models
workers/              Celery tasks and job execution wrappers
configs/              Runtime, model, RAG, training, and style presets
tests/                Pytest smoke/unit/integration coverage
docker/               Demo/frontend/backend Docker assets
```

Data and generated artifacts are intentionally outside the repo:

```bash
AI_CACHE_ROOT=/mnt/c/ai_cache
AI_MODELS_ROOT=/mnt/c/ai_models
AI_OUTPUT_ROOT=/mnt/c/ai_output/anime-adventure-lab
AI_DATASETS_ROOT=/mnt/c/ai_datasets/anime-adventure-lab
```

## Quick Start: Stable Mock Demo

Backend smoke mode:

```bash
conda create -n ai_env python=3.10 -y
conda activate ai_env
pip install -r requirements.txt -r requirements-test.txt

export T2I_MOCK=1 VLM_MOCK=1 LLM_MOCK=1
export MODEL_DEVICE=cpu CUDA_VISIBLE_DEVICES=
export JOBS_SYNC_FALLBACK=1

uvicorn api.main:app --reload
# http://localhost:8000/healthz
# http://localhost:8000/docs
```

React workbench:

```bash
cd frontend/react
npm ci
npm run dev
# http://localhost:3000
```

Static portfolio demo:

```bash
cd portfolio-web
python -m http.server 4173
# http://localhost:4173
```

## Verification

Recommended portfolio gate:

```bash
make test-smoke
cd frontend/react && npm ci && npm run build
```

Direct endpoint checks in mock mode:

```bash
curl http://localhost:8000/healthz
curl http://localhost:8000/api/v1/ready
curl http://localhost:8000/api/v1/worlds
curl http://localhost:8000/api/v1/runtime/presets
curl http://localhost:8000/api/v1/t2i/status
```

Notes:

- `make test-smoke` is the current reliable demo gate.
- The broader historical test/lint suite still contains legacy quality debt and is not used as the portfolio readiness gate yet.

## Deployment

Recommended public deployment:

- **GitHub Pages / Vercel / Netlify:** deploy `portfolio-web/` as a static site.
- **Optional full stack:** deploy FastAPI + React + Redis/Celery with Docker Compose or a platform such as Render/Railway/Fly.
- **GPU inference:** keep as local/advanced mode unless the host has the required model warehouse and worker setup.

Docker demo assets:

```bash
docker build -f docker/demo.Dockerfile -t anime-adventure-lab-demo .
docker build -f docker/demo.backend.Dockerfile -t anime-adventure-lab-demo-api .
```

## Recording Flow

1. Open `portfolio-web/` and switch through the three showcase scenarios.
2. Explain the trace tabs: Story, RAG, Agent, T2I.
3. Run `make test-smoke`.
4. Run `cd frontend/react && npm run build`.
5. Close by explaining the split between stable public mock demo and optional real GPU/model mode.

## Current Status

This repo is now organized around a portfolio-first demo path:

- Mock-safe backend startup and smoke tests.
- Static demo page for screenshots and recordings.
- React production build path.
- README and CI aligned with the current project structure.

Remaining non-blocking work:

- Reduce broader lint debt.
- Expand real model/GPU documentation with exact hardware presets.
- Add more polished sample stories and generated image assets.

License: Apache-2.0 (TBD).
