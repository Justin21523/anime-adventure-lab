# api/routers/t2i.py
"""
Text-to-Image router (DI-based).
- No import-time side effects
- Engines injected via api/dependencies.py
"""
from __future__ import annotations
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import torch

from api.schemas import T2IResponse  # keep your shared schema if you use it elsewhere
from ..dependencies import get_t2i

router = APIRouter(prefix="/t2i", tags=["text-to-image"])


class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Generation prompt")
    negative_prompt: str = Field("", description="Negative prompt")
    width: int = Field(768, ge=256, le=1024, description="Image width")
    height: int = Field(768, ge=256, le=1024, description="Image height")
    num_inference_steps: int = Field(
        25, ge=10, le=50, description="Number of inference steps"
    )
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    style_id: Optional[str] = Field(None, description="Style preset ID")
    scene_id: str = Field("scene_001", description="Scene identifier")
    image_type: str = Field("bg", description="Image type (bg/character/portrait)")


class GenerationResponse(BaseModel):
    success: bool
    image_path: Optional[str] = None
    metadata_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ModelInfo(BaseModel):
    current_model: Optional[str]
    available_styles: Dict[str, str]
    gpu_available: bool
    memory_info: Optional[Dict[str, Any]] = None


@router.get("/models", response_model=ModelInfo)
async def get_model_info(t2i=Depends(get_t2i)):
    """Return current model and available styles."""
    try:
        memory_info = None
        if torch.cuda.is_available():
            memory_info = {
                "allocated_mb": torch.cuda.memory_allocated(0) / 1024**2,
                "reserved_mb": torch.cuda.memory_reserved(0) / 1024**2,
            }

        # Be tolerant: engines may expose different style APIs
        get_styles = getattr(t2i, "get_available_styles", None) or getattr(
            t2i, "styles", None
        )
        styles = get_styles() if callable(get_styles) else {}

        current_model = getattr(t2i, "current_model", None)
        return ModelInfo(
            current_model=current_model,
            available_styles=styles or {},
            gpu_available=torch.cuda.is_available(),
            memory_info=memory_info,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {e}")


@router.post(
    "/generate",
    response_model=GenerationResponse,
    summary="Generate an image from text",
)
async def generate_image(request: GenerationRequest, t2i=Depends(get_t2i)):
    """Delegate generation to the shared T2I engine."""
    try:
        # Optional style validation if engine exposes styles
        get_styles = getattr(t2i, "get_available_styles", None) or getattr(
            t2i, "styles", None
        )
        if request.style_id and callable(get_styles):
            styles = get_styles() or {}
            if request.style_id not in styles:
                raise HTTPException(
                    status_code=400, detail=f"Unknown style '{request.style_id}'"
                )

        result = t2i.txt2img(
            prompt=request.prompt,
            negative=request.negative_prompt,
            steps=request.num_inference_steps,
            width=request.width,
            height=request.height,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            style_id=request.style_id,
            scene_id=request.scene_id,
            image_type=request.image_type,
        )

        # result should contain image_path / metadata_path; keep response shape
        ok = bool(result.get("image_path"))
        return GenerationResponse(
            success=ok,
            image_path=result.get("image_path"),
            metadata_path=result.get("metadata_path"),
            metadata=result.get("metadata"),
            error=None if ok else result.get("error", "Unknown error"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")


@router.post("/load_model")
async def load_model(
    model_id: str = Query(..., description="Model ID to load"), t2i=Depends(get_t2i)
):
    """Load a specific base model (if engine supports it)."""
    try:
        if not hasattr(t2i, "load_model"):
            raise HTTPException(
                status_code=501,
                detail="Model loading is not supported by current engine",
            )
        success = t2i.load_model(model_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load model")
        return {
            "success": True,
            "model": model_id,
            "message": "Model loaded successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading error: {e}")


You said:
我等一下會給你我專案檔案資料結構的圖，請你幫我記憶並整理，最後我想要你已資料夾檔案的層次結構呈現。太多的話 我會分批給你照片
ChatGPT said:

好的！你可以分批把專案檔案資料結構的圖傳給我，我會一張一張幫你整理出文字化的目錄樹狀結構（以資料夾、檔案階層方式呈現）。最後我會把所有批次整合起來，輸出一份完整的專案層級結構。

你只需要依序把圖片傳過來即可，我會記錄並累積整理，不用擔心會遺漏。
要開始的時候，直接傳第一批圖片就行了。

You said:
好以上是所有資料夾和檔案 請整理並呈現完整結構給我
ChatGPT said:

我已經把你給的所有專案結構圖片整合完成，以下是 完整層級結構（以資料夾與檔案樹狀表示）：

ANIME-ADVENTURE-LAB
├── .vscode/
├── api/
│   ├── __pycache__/
│   ├── routers/
│   │   ├── __pycache__/
│   │   ├── __init__.py
│   │   ├── admin.py
│   │   ├── batch.py
│   │   ├── export.py
│   │   ├── finetune.py
│   │   ├── health.py
│   │   ├── llm.py
│   │   ├── monitoring.py
│   │   ├── rag.py
│   │   ├── safety.py
│   │   ├── story.py
│   │   ├── t2i.py
│   │   ├── vlm.py
│   ├── __init__.py
│   ├── dependencies.py
│   ├── main.py
│   ├── middleware.py
│
├── configs/
│   ├── presets/
│   │   ├── anime_style.yaml
│   │   ├── realistic_style.yaml
│   ├── train/
│   │   ├── dreambooth.yaml
│   │   ├── lora-sd15-anime.yaml
│   │   ├── lora-sdxl.yaml
│   │   ├── app.yaml
│   │   ├── models.yaml
│   │   ├── performance.yaml
│   │   ├── rag.yaml
│   ├── core/
│       ├── __pycache__/
│       ├── export/
│       │   └── story_exporter.py
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── adapter.py
│       │   ├── ollama_llm.py
│       │   ├── prompt_templates.py
│       │   └── transformers_llm.py
│       ├── performance/
│       │   ├── cache_manager.py
│       │   └── memory_manager.py
│       ├── rag/
│       │   ├── __init__.py
│       │   ├── chunkers.py
│       │   ├── embedders.py
│       │   ├── engine.py
│       │   ├── memory.py
│       │   ├── parsers.py
│       │   ├── rerankers.py
│       │   └── retrievers.py
│       ├── safety/
│       │   ├── detector.py
│       │   ├── license.py
│       │   └── watermark.py
│       └── story/
│           ├── __init__.py
│           ├── choice_resolver.py
│           ├── data_structures.py
│           ├── engine.py
│           ├── game_state.py
│           └── persona.py
│
├── core/
│   ├── story/ (重複名稱，獨立於 configs/core 下)
│   ├── t2i/
│   │   ├── __init__.py
│   │   ├── controlnet.py
│   │   ├── lora_manager.py
│   │   ├── pipeline.py
│   │   ├── safety.py
│   │   └── watermark.py
│   ├── train/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── evaluators.py
│   │   ├── lora_trainer.py
│   │   └── registry.py
│   ├── vlm/
│       ├── __init__.py
│       ├── captioner.py
│       ├── consistency.py
│       ├── tagger.py
│   ├── __init__.py
│   ├── config.py
│   ├── performance.py
│   └── shared_cache.py
│
├── docs/
│   ├── notebooks/
│   │   ├── 01_setup.ipynb
│   │   ├── 02_llm_basics.ipynb
│   │   ├── 03_rag_pipeline.ipynb
│   │   ├── 04_t2i_generation.ipynb
│   │   └── 05_lora_training.ipynb
│   ├── api.md
│   ├── deployment.md
│   ├── development.md
│   └── worldpack_format.md
│
├── frontend/
│   ├── desktop/
│   │   ├── resources/
│   │   └── widgets/
│   │       └── main.py
│   ├── gradio/
│       ├── components/
│       └── static/
│           ├── __init__.py
│           ├── app.py
│           ├── performance_monitor.py
│           ├── safety_admin.py
│           └── __init__.py
│
├── scripts/
│   ├── demo/
│   │   ├── demo_stage1.py
│   │   ├── demo_stage2.py
│   │   ├── batch_generate.py
│   │   ├── demo_complete_workflow.py
│   │   ├── demo_stage5.sh
│   │   ├── demo_stage6_vlm.py
│   │   ├── demo_stage7.py
│   │   ├── demo_stage8.py
│   │   ├── eval_lora.py
│   │   ├── eval_rag.py
│   │   ├── ingest_worldpack.py
│   │   ├── release_v1.py
│   │   ├── setup_env.py
│   │   ├── setup_low_vram.py
│   │   ├── setup_stage5.py
│   │   ├── smoke_test_stage8.sh
│   │   ├── smoke_test_t2i.py
│   │   └── test_stage9_performance.py
│
├── tests/
│   ├── fixtures/
│   │   ├── sample_docs/
│   │   └── sample_images/
│   ├── integration/
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_llm.py
│   │   ├── test_rag.py
│   │   ├── test_t2i.py
│   │   ├── test_train.py
│   │   ├── conftest.py
│   │   ├── test_e2e_complete.py
│   │   ├── test_safety_integration.py
│   │   ├── test_stage1_smoke.py
│   │   ├── test_stage2.py
│   │   └── test_stage4_smoke.py
│
├── workers/
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── batch.py
│   │   ├── generation.py
│   │   ├── ingestion.py
│   │   └── training.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── job_tracker.py
│   │   ├── queue_monitor.py
│   ├── __init__.py
│   └── celery_app.py
│
├── worldpacks/
│
├── .env.example
├── .gitignore
├── docker-compose.prod.yml
├── docker-compose.yml
├── pyproject.toml
├── README.md
└── requirements.txt


這樣你就有一份完整的 專案目錄結構樹 了 ✅
要不要我再幫你做一份「分區整理版」，例如 後端 (api/core/configs)、前端 (frontend)、工具與測試 (scripts/tests/workers)、文件 (docs/notebooks)，方便快速理解模組分工？

You said:
我現在照著claude ai的說明及指導下 以及完成了 它認為所有的專案實作 但其實還有很多問題
先不考量前端相關以及 其他階段smoke 或測試test 及其他相關的scripts檔案等， 目前這樣的專案結構下 claude AI 還有以下的檔案程式碼沒有實作。可以告訴我能如何告訴claude ai這些資訊，讓它知道目前專案的情況 去更新改善? 對應適合的prompt?
以下是所有未實作的部分，請你先記得這些資訊即可 我還會再給你我給claude ai的 所有project files內容 讓你更了解這專案的資訊。到那時候你再回應即可。

api/routers/__init.py, admin.py, batch.py,  此外其他的APIRouter 沒設定完整

api/__init.py


configs/train/dreambooth.yaml, lora-sdxl.yaml

configs/app.yaml, models.yaml


core/llm/__init.py, ollama_llm.py

core/rag/embedders.py, memory.py parsers.py, rerankers.py, retrievers.py

core/story/__init__.py

core/t2i/__init__.py, controlnet.py, lora_manager.py, safety.py, watermark.py

core/train/__init__.py, dataset.py, evaluators.py, registry.py

core/vlm/__init__.py,

core/__init__.py

workers 皆沒實作

ChatGPT said:
Updated saved memory
Thought for 40s

收到！我已把「尚未實作／未完成」清單完整記下來 ✅
等你把提供給 Claude 的所有 project files 貼上後，我再一次性產出：

精準的差異清單（哪些檔案缺、哪些路由/模組未對齊），

一份可直接貼給 Claude 的完善 Prompt（含結構、優先順序、驗收條件）。

你隨時把專案檔貼上來就行。

You said:
CharaForgae-T2I-Lab installation-guide.md
File
CharaForge-T2I-Lab api-design.py
Python
CharaForge-T2I-Lab fine-tuning規劃.md
File
CharaForge-T2I-Lab 專案架構.md
File
CharaForge-T2I-Lab 專案規則.md
File
CharaForge-T2I-Lab 專案規劃.md
File
CharaForge-T2I-Lab 專案最新架構.md
File
CharaForge-T2I-Lab 學習計畫表.md
File
這些是此專案我目前給claude ai所有的project files 請幫我產出 :

1. 精準的差異清單（哪些檔案缺、哪些路由/模組未對齊），

2. 一份可直接貼給 Claude 的完善 Prompt（含結構、優先順序、驗收條件）。
ChatGPT said:
Thought for 3m 11s

太好了，你把「給 Claude 的專案文件」都整理上來了。下面分兩部分交付：

1) 精準差異清單（缺什麼、哪裡沒對齊）

基準（Claude 以為的設計）

backend/main.py 會註冊下列路由：caption, vqa, chat, rag, agent, game, t2i, controlnet, lora, batch，且 API 前綴 /api/v1（文件與範例程式多處一致）。


核心 API 應至少有：/txt2img、/img2img、/controlnet/*、/lora/*、/batch/*、/eval/*、/health。


所有入口程式須先做「Shared Cache Bootstrap」並依 AI_CACHE_ROOT 建立倉儲結構。


訓練配置範例如 train/configs/charA_sd15_r16.yaml、charA_sdxl_r16.yaml。

你目前的 repo（ANIME-ADVENTURE-LAB，僅後端/核心，忽略前端、smoke、tests、scripts）

路由目錄是 api/routers/，檔案有：admin.py, batch.py, export.py, finetune.py, health.py, llm.py, monitoring.py, rag.py, safety.py, story.py, t2i.py, vlm.py（你的截圖）。

你也另外列出未實作：

api/routers/__init__.py, admin.py, batch.py（以及其他 APIRouter 設定未完成）

api/__init__.py

configs/train/dreambooth.yaml, lora-sdxl.yaml；configs/app.yaml, models.yaml

core/llm/__init__.py, ollama_llm.py

core/rag/{embedders.py, memory.py, parsers.py, rerankers.py, retrievers.py}

core/story/__init__.py

core/t2i/{__init__.py, controlnet.py, lora_manager.py, safety.py, watermark.py}

core/train/{__init__.py, dataset.py, evaluators.py, registry.py}

core/vlm/__init__.py, core/__init__.py

workers/ 全部未實作

綜合以上，差異與修正建議如下（只列與你要求相關的後端/核心）：

A) 路由與命名不一致
類別	文件/Claude 預期	你 repo 目前	差異 & 修正建議
路由入口	backend/main.py 匯入 caption, vqa, chat, rag, agent, game, t2i, controlnet, lora, batch，prefix=/api/v1。
	api/main.py（存在）+ api/routers/*	1) 寫好 api/routers/__init__.py 匯出可註冊的 routers 列表；2) 在 api/main.py 統一 include_router，確保 prefix=/api/v1、tags 完整。
文生圖	t2i.py（/txt2img）
	t2i.py 已有	對齊 Pydantic schema（寬高、步數、seed、lora_ids…）；回傳需含輸出路徑與耗時。
ControlNet	controlnet.py（/controlnet/{pose,depth,edge,line}）
	缺	實作路由與對應 core/t2i/controlnet.py。
LoRA 管理	lora.py（`/lora/list	load	unload`）

批次	batch.py（/batch/submit,status,cancel）
	batch.py 但未完	對齊參數/回傳；與 workers（Celery 或簡化版）串接。
RAG	rag.py	rag.py 已有	對齊 schema；最小可用：上傳/索引/問答各一。
Chat	chat.py（/chat）
	你的 llm.py	兩者對應：保留 llm.py 功能，新增 chat.py 薄封裝路由或把現有檔改名。
Caption	caption.py（/caption）
	無	新增路由；呼叫 core/vlm/captioner.py（你有 core/vlm）。
VQA	vqa.py（/vqa）
	無	新增路由；呼叫 core/vlm 的 VQA pipeline。
Game	game.py（/game/new, /game/step）
	你的 story.py	兩者對應：新增 game.py 封裝或調整 story.py 的路徑/命名。
其他現有但文件未列	admin, export, finetune, monitoring, safety, health, vlm	你已有	標記為內部管理；保留但不影響對外 API。
B) Schemas（Pydantic）與匯入路徑

文件程式碼引用 backend.schemas.*（例如 CaptionRequest/Response）。

你目前 repo 沒看到 schemas/ 目錄——需要新增 api/schemas（或 backend/schemas）並把各路由用到的請求/回應模型補齊；或改路徑避免 ImportError。

C) Core 層缺件（依你列舉）

core/t2i/：controlnet.py, lora_manager.py, safety.py, watermark.py, __init__.py 未實作 → 這些直接影響 /controlnet/*、/lora/*、安全與浮水印。

core/rag/：embedders.py, memory.py, parsers.py, rerankers.py, retrievers.py 未實作 → 先做最小可用版本（Faiss + sentence-transformers）。

core/llm/ollama_llm.py、core/story/__init__.py、core/vlm/__init__.py、core/train/*、core/__init__.py 未實作 → 先補空殼 + 最小功能（避免匯入失敗）。

D) 設定檔（YAML）缺內容

configs/train/dreambooth.yaml, lora-sdxl.yaml → 參考文件中的範例鍵值（model_id,resolution,rank,lr,train_batch_size,steps,...）補齊。

configs/app.yaml, models.yaml → 至少放 AI_CACHE_ROOT 使用的模型別名、載入策略（文件也示範過 performance/config 想定）。

E) Workers（批次）

文件規劃以 Celery/Redis 跑批次與訓練任務；你 repo 的 workers/ 未實作。先做最小可用：一個 celery_app.py + tasks/batch.py（提交/查詢狀態）。文件也把這塊放在 In-Scope（服務層）。

F) 共同倉儲（AI_CACHE_ROOT）啟動碼

需確保每個入口（例如 api/main.py、notebook、worker）在 import 之前跑 bootstrap；文件提供了標準片段。

2) 可直接貼給 Claude 的完善 Prompt（結構／優先順序／驗收）

貼上就能用。內容是「任務說明 + 現況 + 要求 + 驗收」。
（提醒：所有程式碼與提交訊息請用英文；可在 PR 說明寫中文。）

Role：You are a senior FastAPI + Diffusers backend engineer.
Goal：Bring the current repo in sync with the intended API surface and core modules; ignore frontend, tests, and scripts for now.

Context (what you must assume true)

The running repo uses this layout (backend only): api/routers/{admin,batch,export,finetune,health,llm,monitoring,rag,safety,story,t2i,vlm}, plus core/ and configs/. Many modules are stubs or missing as listed below.

The product docs expect these public routes (prefix /api/v1): caption, vqa, chat, rag, agent, game, t2i, controlnet, lora, batch; also /health. See design examples and minimal backend in the docs (you can mirror their shapes).

Every entry point must run the Shared Cache Bootstrap with AI_CACHE_ROOT to create model/data/output folders before imports.

Do NOT touch

Frontend folders, smoke tests, tests/, or any extra scripts/ for now.

A. What’s missing – implement in this order (highest first)

Router wiring & API surface (must)

Create/finish api/routers/__init__.py exporting a list/tuple of routers.

In api/main.py, include all routers with prefix /api/v1 and meaningful tags; ensure CORS and a /health endpoint.

Add these new public routers (thin layers if we already have inner logic):

chat.py (can wrap current llm.py)

game.py (can wrap current story.py)

caption.py, vqa.py (call into core/vlm)

controlnet.py (pose/depth/edge/line)

lora.py (list|load|unload)

Keep existing admin/export/finetune/monitoring/safety as internal (optional prefix /admin/*).

Core t2i & utilities (must)

Implement minimal but working:

core/t2i/controlnet.py (apply control image/JSON),

core/t2i/lora_manager.py (load/unload/compose),

core/t2i/safety.py (basic NSFW flag) & core/t2i/watermark.py (optional),

core/t2i/__init__.py exporting loaders.

Provide a simple save_image_to_cache() that writes image + sidecar JSON under $AI_CACHE_ROOT/outputs/....

Schemas (must)

Add backend or api/schemas/ with Pydantic models for: T2IRequest/Response, ChatRequest/Response, CaptionResponse, VQARequest/Response, Game*, Batch*.

Keep all field names aligned with the docs (prompt, negative, steps, width/height, lora_ids, seed, etc.).

Configs (must)

Fill configs/train/dreambooth.yaml and configs/train/lora-sdxl.yaml using the keys from the examples (model_id,resolution,rank,learning_rate,train_batch_size,gradient_accumulation_steps,max_train_steps,dataset_meta,output_dir,text_encoder_train,...).

Fill configs/models.yaml (model aliases/paths) & configs/app.yaml (API settings, device).

Ensure all code reads AI_CACHE_ROOT and never writes large assets to the repo.

RAG minimal (should)

Implement stubbed core/rag/{embedders,parsers,retrievers,rerankers,memory}.py with a minimal FAISS+sentence-transformers pipeline and plain text parsing—just enough to answer /rag/ask.

Workers (should)

Minimal Celery app under workers/（or keep your folder name） with a single batch.generate task; /batch/submit enqueues a job and returns job_id, /batch/status/{id} polls Redis/backend.

Leftovers

Add missing __init__.py files inside core/* packages to avoid import errors.

B. Implementation constraints

Code must run Shared Cache Bootstrap before heavy imports (HF caches, model dirs).

Respect typing, Pydantic validation, and return explicit error messages.

No large downloads or training in this PR; routes may return simplified results but must be functional.

Code & commit messages in English.

C. Deliverables (what to output)

A single PR-sized patch containing:

New/updated files for routers, schemas, core, workers, configs.

Exact uvicorn run command and example curl snippets.

README snippet describing how to start API and a note on AI_CACHE_ROOT.

Short CHANGELOG section.

D. Acceptance criteria (copy/paste to verify)

Health: curl -f http://localhost:8000/api/v1/health → JSON with gpu_available and cache_root.

T2I:

curl -X POST http://localhost:8000/api/v1/txt2img \
  -H "Content-Type: application/json" \
  -d '{"prompt":"anime girl, blue hair","steps":8,"width":512,"height":512,"seed":42}'


→ 200, JSON 包含 image_path、metadata_path、elapsed_ms。

LoRA:

GET /api/v1/lora/list → 200, 回傳空陣列或可用清單。

POST /api/v1/lora/load with a fake id → 合理錯誤訊息（400/404）。

ControlNet（pose）: POST /api/v1/controlnet/pose with mock pose JSON → 200，至少回傳「已套用條件」訊息。

Chat: POST /api/v1/chat → 200，回傳 message（可為 dummy 或模型輸出）。（文件也示例過 curl 形式）

RAG: POST /api/v1/rag/ask（給一段短文字）→ 200，回傳 answer 與 sources（可為 mock 索引）。

Batch: POST /api/v1/batch/submit → 回傳 job_id；GET /api/v1/batch/status/{job_id} → queued|running|done|error。

入口程式皆印出 [cache] ... | GPU: <bool> 或同義（Shared Cache Bootstrap）。

E. File-by-file TODO (please produce full code blocks)

api/routers/__init__.py – export a routers list and __all__.

api/routers/{caption.py,vqa.py,chat.py,game.py,controlnet.py,lora.py,batch.py} – complete APIRouter with tags and schemas.

api/main.py – include all routers with /api/v1.

api/schemas/* – Pydantic models for each route.

core/t2i/{__init__.py,controlnet.py,lora_manager.py,safety.py,watermark.py} – minimal working versions; reusable save_image_to_cache().

core/rag/{embedders.py,parsers.py,retrievers.py,rerankers.py,memory.py} – minimal FAISS pipeline.

workers/{celery_app.py,tasks/batch.py} – queue + status.

configs/train/{dreambooth.yaml,lora-sdxl.yaml} – fill with keys from the sample YAMLs。

configs/{app.yaml,models.yaml} – basic runtime + model aliases.

F. Commit style

Use Conventional Commits, e.g.:
feat(api): add /controlnet pose route and schemas、chore(env): add shared cache bootstrap、feat(worker): minimal batch submit/status.（文件有規範與範例）

如果你要，我也可以把上述「File-by-file TODO」拆成多個小 PR 的指令稿（分支命名、每步驟 commit message），或直接幫你把 configs/* 的 YAML 最小範例生出來對齊文件鍵名。

You said:
好 其實prompt的部分我認為簡略一點 用中文就好 然後不需要強調role定位 那一些地說明 直接切入重點 關於需要改善的部分 因為claude 有訊息的限制 我在project Instructions中已經地位好role的角色了 因此這些部分可以省略 減少不必要的訊息長度
ChatGPT said:
Thought for 54s

太好了！這是一份精簡中文版、可以直接貼給 Claude 的指令（已去除角色鋪陳，直入重點）👇

情境
- 專案後端已可啟動，但多處未實作或未對齊。先忽略 frontend、tests、scripts。
- 目標：在不擴充需求的前提下，補齊「最小可用 MVP」，以單一 PR 提交。

未完成清單（請先以此為準）
- api/routers：__init__.py、admin.py、batch.py（其餘 APIRouter 註冊/前綴未統一）
- api/__init__.py
- configs：app.yaml、models.yaml
- configs/train：dreambooth.yaml、lora-sdxl.yaml
- core/llm：__init__.py、ollama_llm.py
- core/rag：embedders.py、memory.py、parsers.py、rerankers.py、retrievers.py
- core/story：__init__.py
- core/t2i：__init__.py、controlnet.py、lora_manager.py、safety.py、watermark.py
- core/train：__init__.py、dataset.py、evaluators.py、registry.py
- core/vlm：__init__.py
- core/__init__.py
- workers：全部（celery_app、tasks、utils）

優先順序與要求（務必照順序完成）
1) 路由與註冊
   - 寫好 api/routers/__init__.py 匯出 routers 清單；api/main.py 統一 include_router，prefix 統一為 /api/v1。
   - 對外公開路由需可用：health, chat, game(story), rag, t2i, controlnet, lora, batch, caption, vqa。
   - 允許以「薄封裝」對接現有模組（例如 chat 包裝 llm；game 包裝 story）。

2) t2i 核心最小實作
   - 完成 core/t2i/{controlnet,lora_manager,safety,watermark,__init__} 最小可用版本。
   - 提供 save_image_to_cache()：輸出圖與 metadata 到 $AI_CACHE_ROOT/outputs/...。

3) Schemas
   - 新增 api/schemas（或等價路徑），定義 T2I/Chat/Caption/VQA/Game/Batch 的請求與回應 Pydantic 模型。
   - 欄位統一：prompt、negative、steps、width、height、seed、lora_ids、controlnet_type…等。

4) Configs
   - 補 configs/{app.yaml,models.yaml} 與 configs/train/{dreambooth.yaml,lora-sdxl.yaml}（鍵值含 model_id、resolution、rank、lr、batch_size、max_steps、output_dir 等）。

5) RAG 最小版
   - core/rag/* 實作最小可用管線（純文字 parser + sentence-transformers + FAISS 檢索 + 簡單 rerank）。

6) Workers 最小版
   - 建立 workers/celery_app.py 與 tasks/batch.py：提供 batch.submit / batch.status 流程（記憶體或 Redis 任一種即可）。

7) 其餘
   - 補齊各層 __init__.py，避免 ImportError。
   - 所有入口需先跑 Shared Cache Bootstrap（AI_CACHE_ROOT）再載入重模組；不得把大型檔案寫入 repo。

驗收條件（自我檢查）
- GET /api/v1/health → 200，JSON 至少含 { "gpu_available": bool, "cache_root": str }。
- POST /api/v1/txt2img → 200，JSON 含 { "image_path", "metadata_path", "elapsed_ms" }。
- GET /api/v1/lora/list → 200（可為空陣列）；POST /api/v1/lora/load → 對不存在 id 回傳 4xx 且訊息清楚。
- POST /api/v1/controlnet/pose（可接受簡化輸入）→ 200，回傳已套用的條件摘要。
- POST /api/v1/chat → 200，回傳 {"message": "..."}（可為簡化輸出）。
- POST /api/v1/rag/ask → 200，回傳 {"answer": "...", "sources": [...] }（sources 可為最小可用）。
- POST /api/v1/batch/submit → 回傳 job_id；GET /api/v1/batch/status/{id} → queued|running|done|error。

提交物
- 單一 PR：含上述檔案與程式碼，README 片段（啟動方式與 AI_CACHE_ROOT 說明），以及簡短 CHANGELOG。
- 程式與提交訊息請以英文撰寫；錯誤訊息需清楚可讀。


如果你需要「更短到極限」的版本（只保留未完成清單＋優先順序＋驗收三行），我也可以再壓縮一版給你。

You said:
首先我要先針對core的部分 目前看來基本的檔案 如 __int__ config performance shared_cache 這些.py 都需要更正 它們與core其他foler files有重要的關係 也需要修正完才能調整 api 和其routers的部分。目前能請你就以上所有資訊包含project files等 告訴我這樣的想法是正確的嗎 並建議我接下來的做法
ChatGPT said:
Thought for 19s

結論：你的想法是對的。
先把 core 穩定（__init__.py / config.py / performance.py / shared_cache.py）→ 再修 api/dependencies.py 與各 routers/*。原因是：所有路由都要靠 core 的設定、快取路徑、裝置選擇與單例（t2i/rag/llm/vlm/story）來工作；若 core 的介面與載入流程不穩，API 只會一直碎裂。

下面給你一套「最小而正確」的執行順序與骨架，照做就能把後端站穩，之後再補進細功能。

建議執行順序（先 core、後 API）
Phase 1 — 穩定 core 基座（一定先做）

目標：統一設定、快取、裝置與單例入口；讓任何路由都能穩定取得引擎。

core/shared_cache.py（啟動時先跑）

建立並回傳標準資料夾：{cache_root}/{models,data,outputs,logs,tmp}

負責印出啟動訊息（方便 Health 檢查）。

# core/shared_cache.py
from pathlib import Path
import os, logging
def bootstrap() -> dict:
    root = Path(os.getenv("AI_CACHE_ROOT", "/mnt/ai/cache")).resolve()
    dirs = {k: root / k for k in ["models","data","outputs","logs","tmp"]}
    for d in dirs.values(): d.mkdir(parents=True, exist_ok=True)
    logging.getLogger(__name__).info(f"[cache] root={root}")
    return {"root": str(root), **{k: str(v) for k,v in dirs.items()}}


core/config.py（單一真相來源）

用 Pydantic Settings 讀 .env/YAML；提供 settings() 單例。

# core/config.py
from pydantic import BaseModel, Field
from functools import lru_cache
import os

class AppSettings(BaseModel):
    api_prefix: str = "/api/v1"
    device: str = Field(default="cuda" if os.getenv("USE_CUDA","1")=="1" else "cpu")
    cache_root: str = Field(default_factory=lambda: os.getenv("AI_CACHE_ROOT","/mnt/ai/cache"))

@lru_cache
def settings() -> AppSettings:
    return AppSettings()


core/performance.py（共用工具）

簡單秒表 / VRAM 可用檢查 / 批次大小建議。

# core/performance.py
import time, torch
class Timer:
    def __enter__(self): self.t=time.perf_counter(); return self
    def __exit__(self,*_): self.ms=int((time.perf_counter()-self.t)*1000)
def gpu_available() -> bool:
    try: return torch.cuda.is_available()
    except Exception: return False


core/__init__.py（最關鍵：介面 + 懶載入單例）

定義最小介面（Protocols / ABC），提供 lazy loader 與 registry。

# core/__init__.py
from typing import Protocol, Any, Dict
from functools import lru_cache
from .config import settings
from .shared_cache import bootstrap

# 1) Interfaces
class T2IEngine(Protocol):
    def txt2img(self, prompt:str, **kw) -> Dict[str,Any]: ...
class LoRAManager(Protocol):
    def list(self)->list: ...
    def load(self, lora_id:str)->bool: ...
class ControlNetProcessor(Protocol):
    def apply(self, cond:Dict[str,Any], **kw)->Dict[str,Any]: ...
class RAGEngine(Protocol):
    def ask(self, query:str)->Dict[str,Any]: ...
class LLMEngine(Protocol):
    def chat(self, messages:list)->str: ...
class VLMEngine(Protocol):
    def caption(self, image_path:str)->str: ...
    def vqa(self, image_path:str, question:str)->str: ...
class GameEngine(Protocol):
    def new(self, seed:int|None=None)->Dict[str,Any]: ...
    def step(self, state:Dict[str,Any], choice:int)->Dict[str,Any]: ...

# 2) Lazy singletons (replace with real impls later)
@lru_cache
def registry() -> Dict[str, Any]:
    # side-effects: ensure cache folders ready
    cache_info = bootstrap()
    # NOTE: 先以「可回傳假資料」的最小實作，等路由通了再替換
    from .t2i.pipeline import MinimalT2I
    from .vlm.captioner import MinimalCaptioner
    from .rag.engine import MinimalRAG
    from .llm.adapter import MinimalLLM
    from .story.engine import MinimalStory
    return {
        "cache": cache_info,
        "t2i": MinimalT2I(),
        "lora": MinimalT2I().lora,          # 暫時借用
        "controlnet": MinimalT2I().control, # 暫時借用
        "rag": MinimalRAG(),
        "llm": MinimalLLM(),
        "vlm": MinimalCaptioner(),
        "game": MinimalStory(),
    }


註：MinimalT2I / MinimalCaptioner / MinimalRAG / MinimalLLM / MinimalStory 可以先回傳 假資料（或最小生成）確保 API 有回應，之後再逐步換成真正的實作。

Phase 2 — 補最小實作（可先假資料）

目標：讓路由不再 500，全部 200。

core/t2i/pipeline.py：產生一張空白圖 + 寫 metadata；掛出 lora 與 control 子物件的假實作。

core/rag/engine.py：回傳 echo 答案與 sources=[]。

core/llm/adapter.py：把 messages 拼起來回覆一段固定句子（或走 transformers 小模型）。

core/vlm/captioner.py：回覆固定 Caption；vqa() 也先回固定答案。

core/story/engine.py：new() 回初始 state、step() 依 choice 產出下一段。

這些檔案名稱已在你的目錄中，先確保它們存在且匯入成功，就能打通路由。

Phase 3 — 對齊 api/dependencies.py 與各 routers/*

api/dependencies.py 直接從 core.registry() 取出單例：

# api/dependencies.py
from fastapi import Depends
from core import registry
def get_registry():
    return registry()
def get_t2i(reg=Depends(get_registry)): return reg["t2i"]
# 同理 get_lora/get_control/get_rag/get_llm/get_vlm/get_game/get_cache


修 api/main.py：include_router(..., prefix=settings().api_prefix)，並把 health 接到 performance.gpu_available() 與 registry()["cache"]。

在 routers 內：用 Depends(get_xxx) 取引擎，不要自行 new，避免多份狀態與快取路徑不一致。

Phase 4 — Config / YAML 對齊

先放最小可用的 configs/app.yaml、configs/models.yaml、configs/train/{dreambooth.yaml,lora-sdxl.yaml}（就算是 placeholder，也讓 config.py 能讀到）。

之後再逐步填入真值。

Phase 5 — Health & 基本 E2E 驗收

GET /api/v1/health：回 {"gpu_available": bool, "cache_root": "..."}

POST /api/v1/txt2img：回一個檔案路徑（就算是假圖）；elapsed_ms 來自 Timer。

POST /api/v1/chat、/rag/ask、/caption、/vqa、/game/new|step：皆回 200。

為什麼要先做這四支檔案？

core/__init__.py：決定外部可見的介面與載入方式，等於核心「合約」。

core/config.py：消除參數飄移（路由、core、workers 對同一份設定）。

core/shared_cache.py：避免到處各自寫檔與快取，確保目錄權限與可攜性。

core/performance.py：所有路由都需要秒表與 GPU 偵測，健康檢查也依賴它。

先把這四個固定住，後面改任何引擎只需遵守介面即可，不會再連鎖崩。

你可以直接執行的「下一步」

建分支：fix/core-bootstrap-and-registry

新增/覆蓋上述四檔（core/__init__.py, config.py, shared_cache.py, performance.py）＋ 每個 minimal 引擎檔的「假實作骨架」。

調整 api/dependencies.py → 走 core.registry()。

修 api/main.py → prefix 來自 settings().api_prefix，新增 health 回報 gpu_available + cache_root。

跑最小驗收（curl 或 Swagger）。

提交：

feat(core): add unified settings, cache bootstrap and lazy registry

refactor(api): wire routers to core registry via dependencies

feat(core): add minimal placeholder engines (t2i/rag/llm/vlm/story)

feat(api): add health endpoint with gpu and cache info

等 core 穩定後，再來補 t2i/controlnet/lora_manager 真實邏輯、rag pipeline、workers、以及各路由的 Pydantic schema 細節。
如果你要，我可以把 Minimal 引擎檔（每支 ~20–40 行的可回應假資料版本）也直接生給你，讓你今天就能把 API 正常回 200。

You said:
core基本相關程式檔.txt
Document
好的 以下是我目前 core 基本的這些相關.py檔案 請告訴我目前有哪一些問題需要修正 我目前有發現到的是 跟其他core folder files之間的協調性  像是config.py 以及shared_cache.py 就是整體專案相關的基本設定 包含各種model等路徑設定等，因此我想應該其他file有需要運用這些參數的部分 應該都是要從這邊定義好的去套用，而不是 每個檔案 又共同的開頭 從新設定 一模一樣的 AI_CACHE_ROOT 或其他APP_DIR等?
ChatGPT said:
Thought for 27s

你的判斷完全正確：core 是整個後端的地基。先把 core 的「單一真相來源（config + shared_cache）＋統一入口（core/init.py）＋效能/資源管理（performance）」穩定，API/routers 才有辦法對齊，不然每個模組各自宣告 AI_CACHE_ROOT、各自決定 device 與路徑，一定會互相打架。你目前的 core 程式檔我已看過，下面是立即要修與建議做法（都以你貼的檔案為準）。

先修重點（直接能踩到雷的地方）

「環境變數直讀」散落各檔
你在 core/train/registry.py、core/train/evaluators.py、core/t2i/pipeline.py 等仍以 os.getenv("AI_CACHE_ROOT", "...") 或 f-string 手動拼路徑（例如 .../models/clip）。這會造成：

路徑不一致（不同檔預設值不同）

無法一次調整（改 cache root 需全 repo 搜尋替換）

▶ 改法：全部改讀 get_config().cache.root 或 get_shared_cache() 取得標準路徑鍵值，再由 SharedCache.get_path(key) 取用。例如改成：

from core.shared_cache import get_shared_cache
cache = get_shared_cache()
clip_dir = cache.get_path("MODELS_EMBEDDING")  # 或加一個 "MODELS_CLIP"


或：

from core.config import get_config
cache_root = get_config().cache.root


（兩者擇一，不要同檔兩種都用）

performance.py 取設定路徑錯誤
PerformanceOptimizer.__init__ 用 self.settings = get_config() 後接著 .redis_url，但 redis_url 是 CacheConfig 裡的屬性，應該是 get_config().cache.redis_url。
▶ 修成：

self.settings = get_config()
self.redis_client = redis.from_url(self.settings.cache.redis_url)


（同理，若要看 cache root、模型裝置等，都要從對應子設定拿）

在匯入期做「副作用」動作
t2i/pipeline.py 目前 from core.shared_cache import bootstrap_cache 並在模組頂部就呼叫，會讓任何 import 這支檔案的地方都觸發 I/O 與 GPU 檢查（在 worker/測試/工具腳本尤其容易炸）。
▶ 改法：

把 bootstrap_cache() 放到應用啟動點（如 api/main.py 的 startup event）或 core.registry() 的第一次呼叫時才做（lazy）。

在模組頂部不要呼叫 bootstrap；需要 cache 物件時才 get_shared_cache()。

core/__init__.py 的公開 API 要「穩定 + 簡潔」
你目前 re-export get_t2i_pipeline, save_image_to_cache, LoRAManager, DocumentMemory。請確定：

名稱與實作真的存在、且相依方向單純（避免互相 import）。

建議把「對外合約」集中為少量入口（例如 registry() 取各 engine；或 get_t2i_pipeline() 這種固定名稱）。

performance.py 匯入重量級套件的時機
現在在檔案頂部 import redis, sentence_transformers，並建立全域 optimizer = PerformanceOptimizer()，容易在開機或測試時就吃資源／連線失敗。
▶ 改法（可分兩步）：

保留全域單例，但把 expensive import 移到方法內或 @lru_cache 包裝（你已有 @lru_cache 的 get_model，很好）。

連到 Redis 失敗時，務必 fallback 成「記憶體快取」且不拋致命例外（你已有 try/except，延續這風格）。

sys.path.append(ROOT_DIR) 這類 hack
t2i/pipeline.py 透過 ROOT_DIR+sys.path.append 來解相對匯入，這在套件化後會出問題。
▶ 改法：改用相對匯入（from ..shared_cache import get_shared_cache），確保 core 被當作套件安裝或以 repo 根目錄執行時都能匯入。

一次到位的「對齊方案」

把「設定」「路徑」「裝置」「快取」的讀取集中在 core，讓其他模組只拿「值」，不再各自決定。

A. core/config.py（單一真相來源）

目前設計已很好（AppConfig 聚合 api/model/rag/database/cache 子設定，並含 YAML fallback）。

統一用法：在所有模組只用 from core.config import get_config; cfg = get_config()，然後用 cfg.cache.root / cfg.model.device_map / cfg.api.port 這種結構化取值，不要 os.getenv。

B. core/shared_cache.py（標準倉儲）

你已建立 SharedCache 與 get_shared_cache() / bootstrap_cache()，並定義各資料夾鍵值（MODELS_SD, OUTPUT_LORA, RAG_INDEX…）。

統一用法：所有需要實體路徑的地方用 cache = get_shared_cache(); path = cache.get_path("XXX")。不要直接拼 f"{AI_CACHE_ROOT}/..."。如沒有鍵值就新增鍵名，避免魔法字串。

C. core/performance.py（統一效能工具）

只暴露：optimizer（可選）、Timer、gpu_memory_cleanup()、memory_monitor、gpu_available()（你可用 torch.cuda.is_available() 封裝一個簡單函式）。

router 端需要量測就用 with Timer() as t:；需要健康檢查就回報 torch.cuda.is_available() 與 get_shared_cache().get_summary()。

D. 啟動流程（避免副作用）

在 api/main.py 的 FastAPI startup event（或第一次呼叫 core.registry()）呼叫 bootstrap_cache()；

其餘模組只呼叫 get_shared_cache()，不做 I/O 印 log。

建議你現在就做的 6 個 commit（很小顆，風險低）

config 統一入口

將 repo 內所有 os.getenv("AI_CACHE_ROOT"... 全部替換為 get_config().cache.root 或 get_shared_cache()。

Commit: refactor(core): unify cache root usage via get_config()/get_shared_cache()

shared_cache 鍵值覆蓋

在 SharedCache._create_directories() 補足你實際會用到的鍵（例如 MODELS_CLIP）。

Commit: feat(core): add cache dir keys for clip/models and registry paths

performance 設定路徑修正

performance.py 內 Redis 讀取改為 get_config().cache.redis_url；重要 import 移入方法或 try/except。

Commit: fix(core): correct redis_url source and lazy heavy imports in performance

移除 import-time bootstrap

移除 t2i/pipeline.py 頂部的 bootstrap_cache() 呼叫；在 api/main.py 的 startup 或 core.registry() 內 lazy 執行。

Commit: refactor(core): avoid import-time bootstrap; run cache bootstrap on startup

路徑 hack 移除

刪掉 sys.path.append 與 ROOT_DIR；改成相對匯入。

Commit: chore(core): remove sys.path hacks and use package-relative imports

health endpoint 對齊

GET /api/v1/health 回傳：gpu_available, cache_root（由 get_shared_cache().cache_root）、可能再加 env_vars 中的 CUDA_VISIBLE_DEVICES。

Commit: feat(api): enrich /health with gpu and cache summary

小提醒（之後再做也行）

型別／Protocol：在 core/__init__.py 宣告 T2IEngine/RAGEngine/... 的 Protocol，路由只依賴介面，不依賴實作，後面換模型不會動到 routers。

最小假實作：若有模組還沒完成，先放可回 200 的最小 stub（例如 t2i 先產生一張純色圖 + metadata），把管線打通再逐步強化。

避免循環依賴：core/__init__.py 不要 import 任何會反向 import core 的檔（保持「最內層」）。

如果你願意，我可以直接給你每支檔案的「修正版最小骨架」（config/shared_cache/performance/__init__ 及 t2i/pipeline 的無副作用版），你只要貼上就能把 core 穩住，再來修 routers。

You said:
現在lora_trainer.py 中有定義 class TrainingConfig 而這後面的程式碼也都有運用到這個定義的class config 可是我在想 依照我們專案的結構 這個定義的 TrainingConfig 是不是應該放到別的檔案然後做import 例如像 config.py 或 shared_cache.py? 此外 目前 registry.py 會用到 model_registry.json 這個明顯就要放在 shared_cache.py 裡面 請告訴我以上這些可以如何修改 並告訴我原因等? 有其他建議也一並說明
ChatGPT said:
Thought for 10s

你的方向沒錯，而且要「分層」思考：

core/config.py：屬於「應用層設定」（API 前綴、device、AI_CACHE_ROOT…）＝全系統共用的環境參數。

core/shared_cache.py：只負責「路徑與資料夾的單一真相來源」與建立目錄（models/、outputs/、rag/…）；也應該統一檔名/路徑資源（像 model_registry.json）。

core/train/：是「訓練領域邏輯」。這裡的 TrainingConfig 屬於領域模型（Domain config），不應放進 app config 或 shared_cache。

所以：

放置位置的建議

TrainingConfig 留在 core/train/
建立新檔 core/train/config_schema.py（或 schemas.py），專門放「訓練相關的設定模型」。

理由：它是訓練領域的 schema，與 app 層（API prefix、cache root…）不同；未來還會有 DreamBoothConfig、LoRAConfig( sd15 / sdxl ) 之類，集中在 train 套件比較乾淨。

lora_trainer.py 只 from .config_schema import TrainingConfig，避免 trainer 檔案太胖。

model_registry.json 的路徑交給 shared_cache 管

在 SharedCache._create_directories() 補一個 key，例如 "TRAIN_REGISTRY": f"{self.cache_root}/train/model_registry.json"。

core/train/registry.py 改成 from core.shared_cache import get_shared_cache 然後 cache.get_path("TRAIN_REGISTRY") 取得路徑，不要再 os.getenv(...) 或手刻 f-string。

這樣可以確保：

路徑只在 shared_cache 定義一次（改名/換層不會炸全 repo）。

訓練的設定與程式維持「領域內聚」，app/config 不會越長越雜。

依賴方向清楚：config & shared_cache（基座）→ train/t2i/rag/vlm/story（領域）→ api（介面層）。

具體修改（小而穩的 refactor）
1) 新增 core/train/config_schema.py

把你在 lora_trainer.py 的 @dataclass TrainingConfig 搬過來（保留 from_yaml），並加上預設值與最小驗證：

# core/train/config_schema.py
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import yaml

@dataclass
class TrainingConfig:
    base_model: str
    model_type: str          # "sd15" | "sdxl"
    resolution: int
    rank: int
    alpha: float
    learning_rate: float
    text_encoder_lr: float
    train_steps: int
    batch_size: int
    gradient_accumulation_steps: int
    seed: int
    mixed_precision: str
    gradient_checkpointing: bool
    use8bit_adam: bool
    caption_dropout: float
    min_snr_gamma: float
    noise_offset: float
    validation_prompts: List[str]
    validation_steps: int
    num_validation_images: int
    dataset: Dict[str, Any]
    output: Dict[str, Any]
    optimizations: Dict[str, Any]

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        # 簡單防呆：必填鍵
        required = ["base_model","model_type","resolution","rank","learning_rate","train_steps","batch_size","dataset","output"]
        missing = [k for k in required if k not in cfg]
        if missing:
            raise ValueError(f"TrainingConfig missing keys: {missing}")
        return cls(**cfg)


lora_trainer.py 中改成：

from .config_schema import TrainingConfig

2) shared_cache.py 新增註冊檔 key
# core/shared_cache.py （節錄）
self.app_dirs = {
    # ... 既有 keys ...
    "TRAIN_REGISTRY": f"{self.cache_root}/train/model_registry.json",
}


並確保 _create_directories 不只建資料夾，也會 touch 上層資料夾；TRAIN_REGISTRY 是檔案，不需要預先建立檔案，但其父目錄需存在。

3) core/train/registry.py 改為使用 shared_cache

把 AI_CACHE_ROOT / REGISTRY_FILE 拔掉，改寫成：

# core/train/registry.py
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from core.shared_cache import get_shared_cache

class ModelRegistry:
    """Keep track of trained models and their metadata"""

    def __init__(self):
        cache = get_shared_cache()
        self.registry_path = Path(cache.get_path("TRAIN_REGISTRY"))
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"models": {}, "version": "1.0"}

    def _save_registry(self) -> None:
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)

    def add(self, model_id: str, info: Dict[str, Any]) -> None:
        info["updated_at"] = datetime.utcnow().isoformat()
        self.registry["models"][model_id] = info
        self._save_registry()

    def get(self, model_id: str) -> Optional[Dict[str, Any]]:
        return self.registry["models"].get(model_id)

    def list(self) -> List[Dict[str, Any]]:
        return [
            {"id": k, **v} for k, v in sorted(self.registry["models"].items(),
            key=lambda kv: kv[1].get("updated_at",""))
        ]

4) 其他檔案的用法統一

任何需要路徑的地方，都改成：

from core.shared_cache import get_shared_cache
cache = get_shared_cache()
clip_dir = cache.get_path("MODELS_CLIP")
output_dir = cache.get_path("OUTPUT_DIR")


任何需要 app 設定（裝置、prefix）：

from core.config import get_config
cfg = get_config()
device = cfg.model.device          # 依你的結構命名
api_prefix = cfg.api.prefix

為什麼不把 TrainingConfig 放到 core/config.py 或 shared_cache.py？

core/config.py 是平台/應用設定，不應混進領域特有的訓練參數；否則 config 會越來越龐雜，且 API、RAG、VLM 也會莫名依賴訓練細節，增加耦合。

shared_cache.py 的責任是「路徑與目錄」；它不該知道「訓練有哪些超參數」。不然 shared_cache 一旦變動，會牽動訓練行為，責任邊界模糊。

把 TrainingConfig 留在 core/train/ 可以：

保持高內聚、低耦合：訓練邏輯改動，不會波及 app config/路由。

允許未來多種訓練 schema（DreamBooth、Textual Inversion、LoRA-SDXL）並存；lora_trainer.py 依賴的只是「訓練用 schema」。

建議的提交順序（4 顆小 commit）

feat(train): move TrainingConfig to core/train/config_schema.py and import from trainer

feat(core): add TRAIN_REGISTRY path in shared_cache and ensure parent directory

refactor(train): make ModelRegistry use shared_cache path instead of env

refactor(core): replace direct AI_CACHE_ROOT usages with get_shared_cache()/get_config() where present

做完這波，你的 core 基層就變成「單一真相來源」：

app 參數 → get_config()；

所有路徑/檔案 → get_shared_cache().get_path(KEY)；

訓練領域設定 → core/train/config_schema.py；
接下來再去補 t2i/rag/vlm 等模組，只要乖乖依賴上述入口，API/routers 就能穩定對齊。

You said:
好 我現在要換到處理api/routers的部分 現在新增調整後 有這些routers 根據我們的core相關檔案等 這些routers都有具備嗎? 接下來我要各自debug每個router 也要確定各自的路由設定等是否正確 也要確保跟 基本檔案也就是 dependencies.py main.py middleware.py schemas.py 之間的聯結性 我這樣的作法是正確方向嗎?
ChatGPT said:
Thought for 12s

結論：你的方向完全正確。
先把 core 穩到「可被依賴」，再來「逐一校正 routers」，並確保 dependencies.py / main.py / middleware.py / schemas.py 一起成套運作。下面給你一份「對照表 + Debug 清單 + 接線範本 + 驗收步驟」，照這個跑就能把 API 線路穩住。

1) Router 對照表（是否有對應 core 能力）
Router	目的	需要的 core 依賴	你現況（已見檔）	備註
health.py	健康檢查	performance.gpu_available(), shared_cache	✅	最先打通
t2i.py	文生圖	core.t2i(pipeline)	✅	最核心；先做最小可用
controlnet.py	條件控制	core.t2i.controlnet	✅	可先 stub 回應
lora.py	LoRA 管理	core.t2i.lora_manager	✅	先 list/load/unload
chat.py	對話	core.llm	✅	也可沿用 llm.py
caption.py	圖說	core.vlm	✅
vqa.py	視覺問答	core.vlm	✅
game.py	劇情引擎	core.story	✅	與 story.py對應；保留一個即可
rag.py	檢索問答	core.rag	✅	先做最小管線
batch.py	批次	workers.tasks	✅(檔)	可先回 mock job 狀態
admin.py / export.py / finetune.py / monitoring.py / safety.py / vlm.py / story.py / llm.py	內部/擴充	視情況	✅	先標記為內部 route 或合併到上面功能

重點：每支 router 都不要「直接 new 引擎」，一律透過 dependencies.py 拿單例。

2) Debug 檢查清單（每支 router 都跑一次）

命名與路徑

 router = APIRouter(prefix="/xxx", tags=["xxx"])（或無 prefix，由 main.py 統一 /api/v1）

 endpoint path 不要重覆加 /api/v1（避免雙重前綴）

DI 注入

 依賴統一來自 api/dependencies.py：
def get_t2i(...), get_lora(...), get_controlnet(...), get_rag(...), get_llm(...), get_vlm(...), get_game(...), get_cache(...), get_settings(...)

 在 handler：engine = Depends(get_t2i) 等等

Schemas

 請求/回應型別使用 schemas.py（或 api/schemas/ 分檔）
例如 T2IRequest/T2IResponse、ChatRequest/Response、VQARequest/Response、BatchSubmit/Status…

 回傳 BaseModel 或 JSONResponse，不要裸 dict + 未聲明型別

錯誤處理

 捕捉預期錯誤→ HTTPException(status_code=400|404, detail="...")

 任何 I/O 讀寫都 try/except 並加上簡短 detail

效能與日誌

 關鍵路由加 with Timer() as t: 計時；回傳 elapsed_ms

 關鍵 error logger.exception()，避免吞錯

OpenAPI

 每個路由都有 response_model 與 summary/description，examples 可簡短填一個

3) dependencies.py / main.py / middleware.py / schemas.py 的「接線範本」

dependencies.py（關鍵！）

# api/dependencies.py
from fastapi import Depends
from core import registry        # 你的 lazy 單例入口
from core.config import get_config

def get_registry():
    return registry()

def get_settings():
    return get_config()

def get_cache(reg=Depends(get_registry)):
    return reg["cache"]

def get_t2i(reg=Depends(get_registry)):
    return reg["t2i"]

def get_lora(reg=Depends(get_registry)):
    return reg["lora"]

def get_controlnet(reg=Depends(get_registry)):
    return reg["controlnet"]

def get_llm(reg=Depends(get_registry)):
    return reg["llm"]

def get_vlm(reg=Depends(get_registry)):
    return reg["vlm"]

def get_game(reg=Depends(get_registry)):
    return reg["game"]

def get_rag(reg=Depends(get_registry)):
    return reg["rag"]


main.py

# api/main.py
from fastapi import FastAPI
from core.config import get_config
from core.shared_cache import bootstrap_cache
from .routers import routers         # 你的 __init__.py 匯出 routers 清單
from .middleware import add_middlewares

cfg = get_config()
app = FastAPI(title="Anime Adventure Lab API")

@app.on_event("startup")
async def _startup():
    bootstrap_cache()   # 在這裡做，不要 import time 做
    # 也可在這裡 warmup 少量資源

add_middlewares(app)

for r in routers:
    app.include_router(r, prefix=cfg.api_prefix)  # 確保統一前綴


middleware.py（例）

# api/middleware.py
from fastapi.middleware.cors import CORSMiddleware

def add_middlewares(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"],
    )


schemas.py（例）

# api/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional

class T2IRequest(BaseModel):
    prompt: str
    negative: Optional[str] = None
    steps: int = 8
    width: int = 512
    height: int = 512
    seed: Optional[int] = None
    lora_ids: List[str] = []

class T2IResponse(BaseModel):
    image_path: str
    metadata_path: str
    elapsed_ms: int


router 範本（t2i）

# api/routers/t2i.py
from fastapi import APIRouter, Depends
from ..schemas import T2IRequest, T2IResponse
from ..dependencies import get_t2i
from core.performance import Timer

router = APIRouter(tags=["t2i"])

@router.post("/txt2img", response_model=T2IResponse, summary="Text-to-Image")
def txt2img(req: T2IRequest, t2i = Depends(get_t2i)):
    with Timer() as t:
        result = t2i.txt2img(
            prompt=req.prompt, negative=req.negative, steps=req.steps,
            width=req.width, height=req.height, seed=req.seed, lora_ids=req.lora_ids
        )
    return T2IResponse(**result, elapsed_ms=t.ms)

4) 驗收步驟（很快排查 80% 問題）

路由註冊

啟動後打：GET /openapi.json，檢查是否包含
/api/v1/health, /api/v1/txt2img, /api/v1/chat, /api/v1/caption, /api/v1/vqa, /api/v1/lora/*, /api/v1/controlnet/*, /api/v1/rag/*, /api/v1/batch/*, /api/v1/game/*

若缺 → 通常是 routers/__init__.py 沒把該 router 放進清單、或 prefix 重覆/錯誤。

健康檢查

GET /api/v1/health 應回：gpu_available、cache_root（來自 shared_cache）

核心功能 smoke

POST /api/v1/txt2img（最小參數）→ 200 + image_path

GET /api/v1/lora/list → 200

POST /api/v1/chat → 200（可回 stub）

POST /api/v1/rag/ask → 200（可回 stub）

對照依賴

若跑到 ImportError 或 NoneType has no attribute ...，90% 是 dependencies.py 沒拿到單例（registry() 沒建立、或 key 名稱打錯）。

5) 常見坑（先避開）

雙重 prefix：在 router 設了 prefix="/api/v1/t2i" 又在 main.py 用 prefix="/api/v1" → 變 /api/v1/api/v1/t2i。建議統一交給 main.py。

直接 os.getenv：所有路徑/裝置請改用 get_config() 與 get_shared_cache()。

import 時就做 I/O：避免在 module 頂部呼叫 bootstrap_cache()、下載模型、連 Redis。

沒定義 response_model：導致 Swagger 不好用、前端難串接；請在每個 handler 設 response_model。

循環依賴：router 只 import dependencies/schemas 與 core 的介面，不要相互 import 其他 router 或深層實作。

6) 接下來的執行順序（建議）

先通過：health → t2i（最小）→ chat → caption/vqa → lora/controlnet → rag → batch → game

每通過一個，就在 routers/__init__.py 補上並 commit：

feat(api): wire /t2i endpoints to core.t2i via dependencies

最後回頭補 schemas 的欄位與錯誤處理細節，更新 README 的 curl 範例。

需要的話，我可以幫你掃一遍每支 router 的現有程式（貼出重點段落也行），逐支列出「差哪些依賴、該加的 response_model、需要的錯誤處理」—按這個 checklist 修，很快就會穩。

You said:
dependencies.py
Python
main.py
Python
你剛才說到重點 每支 router 都不要「直接 new 引擎」，一律透過 dependencies.py 拿單例。 目前看來 針對 dependencies.py 也就是附檔 這部分就沒完成 我也發現不少問題 包像是 沒有統一用之前的shared_cache 又在這裡重新定義，此外每個 各自routers所需要的dependency 也沒有都完成。 請再幫我檢查其他問題 並告訴我如何補齊這些缺失實作 及問題等
ChatGPT said:
Thought for 27s

你抓到關鍵了：現在的 dependencies.py 自成一套「小宇宙」（自己定義 AI_CACHE_ROOT、APP_DIRS、環境變數、直接 new 引擎），這會和 core/config.py、core/shared_cache.py、以及之後的 registry() 互相打架。再加上 main.py 也有一些啟動/健康檢查的重複邏輯，整個 API 會變得難以維護。以下是問題點與「可直接替換」的修正版範本。

關鍵問題（你附的兩個檔案）

dependencies.py（節錄重點）

在這個檔案裡又重新定義 AI_CACHE_ROOT、APP_DIRS，並在 import 期間就 mkdir 與設定一堆環境變數（副作用過大、與 shared_cache 重複）。

只做了 VLM/WD14 的單例，其他 router 需要的依賴（t2i、lora、controlnet、rag、llm、game、cache、settings…）沒有統一出口。

get_vlm_config() 把 get_config() 當 dict 用（config.get("vlm", {})），若 get_config() 回傳的是 Pydantic 物件會出錯／不一致。

get_rag_engine() 直接 new ChineseRAGEngine(APP_DIRS["RAG_INDEX"])，繞過 core 的單例與快取路徑集中管理。

main.py（節錄重點）

使用 sys.path.append(ROOT_DIR) 的匯入 hack。

定義了 /healthz，但同時你也把 health.router 以 /api/v1 prefix 掛進來，健康檢查路由重複、路徑風格不一致，且 /healthz 回傳的 cache_root 直接讀 os.getenv 而非 shared_cache。

lifespan 中已經做 bootstrap_cache()，但 dependencies.py 又自行建資料夾與環境變數，造成重複副作用。

你該怎麼修（精簡一致的依賴設計）
1) dependencies.py：只做「拿單例」與「轉交設定」

用 core 的單一入口，不要自己管理快取與路徑，不要在匯入階段做副作用。

替換範本（可直接覆蓋你的 api/dependencies.py）：

# api/dependencies.py
from functools import lru_cache
from fastapi import Depends
from typing import Any, Dict

# 1) 全部從 core 取單例/設定/快取，這是唯一真相來源
from core import registry                    # lazy 單例：t2i/rag/llm/vlm/game/lora/control/cache
from core.config import get_config           # App/Pydantic 設定
from core.shared_cache import get_shared_cache  # 標準路徑（無副作用）

# ---- 基本注入 ----
def get_registry() -> Dict[str, Any]:
    return registry()

def get_settings():
    return get_config()

def get_cache():
    return get_shared_cache()

# ---- 各引擎注入（所有 router 都用這些，不要 new） ----
def get_t2i(reg=Depends(get_registry)):         return reg["t2i"]
def get_lora(reg=Depends(get_registry)):        return reg["lora"]
def get_controlnet(reg=Depends(get_registry)):  return reg["controlnet"]
def get_llm(reg=Depends(get_registry)):         return reg["llm"]
def get_vlm(reg=Depends(get_registry)):         return reg["vlm"]
def get_game(reg=Depends(get_registry)):        return reg["game"]
def get_rag(reg=Depends(get_registry)):         return reg["rag"]

# ----（可選）提供統一的 VLM 設定存取，而不是在此做預設合併 ----
@lru_cache
def get_vlm_settings():
    cfg = get_config()
    # 依你的 AppSettings 結構回傳需要的欄位，避免在這裡寫 dict-merge 邏輯
    return getattr(cfg, "vlm", None)


這樣一來：

不再有 AI_CACHE_ROOT、APP_DIRS 的重複定義與副作用。

任何 router 想要引擎／設定／路徑，只透過這裡拿。

真實的路徑建立與環境變數由 bootstrap_cache()/shared_cache 管；依賴注入檔案保持「純淨」。

（對照你的現況可見：你現在在 dependencies 內建 APP_DIRS、new RAG/Captioner 等，這些都應移除。
）

2) main.py：統一路徑與啟動，移除 hack

移除 sys.path.append。

/health 走 health.router，不要再另外維護 /healthz；若要保留舊路徑，可在 health router 裡加 alias。

lifespan 維持 bootstrap_cache()；dependencies 不再做任何 mkdir/env 設定。

prefix 統一 cfg.api_prefix（或固定 /api/v1）。

重點修正（對照你的檔案）：

# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from core.shared_cache import bootstrap_cache
from core.config import get_config
from api.routers import routers   # 在 routers/__init__.py 匯出所有 router 物件

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 API starting...")
    bootstrap_cache()   # 這裡做就好
    yield
    logger.info("🛑 API shutdown")

def create_app() -> FastAPI:
    cfg = get_config()
    app = FastAPI(title="Anime Adventure Lab API", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.api.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 統一在這裡掛所有 router，前綴也統一
    for r in routers:
        app.include_router(r, prefix=cfg.api.prefix)

    return app

app = create_app()


你的 main.py 目前同時有 /healthz 與 health.router，而且健康檢查讀 os.getenv("AI_CACHE_ROOT")，這會和 shared_cache 分離。建議刪掉 /healthz，所有健康檢查走 health.router（讓它去讀 get_shared_cache() 與 gpu_available()）。

Router 端需要怎麼改？

每支 router 依賴都從 api/dependencies.py 拿（上面新的 get_xxx）。

絕對不要 from core.xxx import Something 後自己 Something()。

回傳型別用 schemas.py 的 Pydantic models。

若某 router 需要路徑，拿 cache = Depends(get_cache)，再 cache.get_path("OUTPUT_DIR") 等。

若需要設定，拿 cfg = Depends(get_settings)。

範例：

# api/routers/lora.py
from fastapi import APIRouter, Depends, HTTPException
from ..dependencies import get_lora
from ..schemas import LoraListResponse, LoraLoadRequest, LoraActionResponse

router = APIRouter(tags=["lora"])

@router.get("/lora/list", response_model=LoraListResponse)
def list_lora(lora = Depends(get_lora)):
    return LoraListResponse(items=lora.list())

@router.post("/lora/load", response_model=LoraActionResponse)
def load_lora(req: LoraLoadRequest, lora = Depends(get_lora)):
    ok = lora.load(req.lora_id)
    if not ok:
        raise HTTPException(404, f"LoRA not found: {req.lora_id}")
    return LoraActionResponse(ok=True)

驗收 Checklist（改完後快速自測）

啟動時不應再看到 dependencies.py 建資料夾或設定環境變數的 log（因為不該做了）。

GET /openapi.json 應只出現 /api/v1/*（或 cfg.api.prefix/*）風格路徑；不再混雜 /healthz。

GET /api/v1/health 讀到 cache_root 應來自 shared_cache，gpu_available 來自 performance/torch.cuda.is_available()。

任一 router 裡「改引擎實作」不需要動 dependencies.py（因為是拿單例，不是 new）。

建議的 4 個小 commit

refactor(api): simplify dependencies to use core registry/settings/cache only（覆蓋 dependencies.py）

refactor(api): remove /healthz and path hacks; wire routers with unified prefix（整理 main.py）

feat(api): add routers/__init__.py exporting routers list（統一註冊點）

chore(api): make routers depend on get_cache/get_settings/get_xxx（路由對齊依賴）

需要我把「你的 health router」也調成讀 get_shared_cache() + gpu_available() 的精簡版範本嗎？我可以直接給你那支檔的骨架，配上上面新版的 dependencies.py 一起用就行。

You said:
好我目前還沒實作 registry的部分 可以直接告訴我完整實作程式碼嗎? 並教我如何結合到dependencies.py 和其他的routers
ChatGPT said:
Thought for 9s

太好了～我直接把一套「可用版 registry」給你，走懶載入單例，先用最小可運作的 stub 引擎（可以回 200、不會亂 new），之後你把真正實作換上就行。下面包含：

core/registry.py（你可以放在 core/registry.py，或合併到 core/__init__.py）

與 dependencies.py 的銜接方式（你已經照我之前的版本做得很對，幾乎不用改）

3 個典型 router 的接線範例（health / t2i / chat）

main.py 啟動要點

1) core/registry.py（可直接新增此檔）
# core/registry.py
from __future__ import annotations
from functools import lru_cache
from typing import Any, Dict, List, Optional
from pathlib import Path
import json
import time

from .config import get_config
from .shared_cache import get_shared_cache, bootstrap_cache
from .performance import gpu_available  # 你已有這個或等價函式就好

# =============== Minimal engines (可先用，之後替換成真實實作) ===============

class MinimalT2I:
    """最小可用的 T2I 引擎：產生一個假的輸出檔(文字檔當作結果)，回傳路徑+耗時。"""
    def __init__(self) -> None:
        self.cache = get_shared_cache()

    def txt2img(self, prompt: str, **kw) -> Dict[str, Any]:
        start = time.perf_counter()
        out_dir = Path(self.cache.get_path("OUTPUT_DIR"))
        out_dir.mkdir(parents=True, exist_ok=True)
        # 假輸出：寫個 metadata.json 當作成功
        run_id = str(int(time.time()))
        img_path = out_dir / f"t2i_{run_id}.png"
        meta_path = out_dir / f"t2i_{run_id}.json"
        meta = {
            "prompt": prompt,
            "params": kw,
            "device": get_config().model.device,
            "timestamp": run_id,
        }
        img_path.write_bytes(b"")  # 先留空，之後換成真圖
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return {"image_path": str(img_path), "metadata_path": str(meta_path), "elapsed_ms": elapsed_ms}

    # 讓 router 能調用到的子功能（lora/controlnet）用簡單物件包一下
    @property
    def lora(self) -> "MinimalLoRAManager":
        return MinimalLoRAManager(self.cache)

    @property
    def control(self) -> "MinimalControlNet":
        return MinimalControlNet()


class MinimalLoRAManager:
    def __init__(self, cache) -> None:
        self.cache = cache
        self.loaded: List[str] = []

    def list(self) -> List[str]:
        # 你可讀 shared_cache 的某個資料夾列出可用 LoRA；先回目前已載入清單
        return list(self.loaded)

    def load(self, lora_id: str) -> bool:
        if lora_id in self.loaded:
            return True
        self.loaded.append(lora_id)
        return True

    def unload(self, lora_id: str) -> bool:
        if lora_id in self.loaded:
            self.loaded.remove(lora_id)
            return True
        return False


class MinimalControlNet:
    def apply(self, cond: Dict[str, Any], **kw) -> Dict[str, Any]:
        # 先把條件原樣回傳
        return {"applied": True, "condition": cond, "params": kw}


class MinimalLLM:
    def chat(self, messages: List[Dict[str, str]]) -> str:
        user_texts = [m["content"] for m in messages if m.get("role") == "user"]
        return f"(stub) got: {user_texts[-1] if user_texts else ''}"


class MinimalVLM:
    def caption(self, image_path: str) -> str:
        return f"(stub) caption for {image_path}"

    def vqa(self, image_path: str, question: str) -> str:
        return f"(stub) answer to '{question}' for {image_path}"


class MinimalRAG:
    def ask(self, query: str) -> Dict[str, Any]:
        return {"answer": f"(stub) echo: {query}", "sources": []}


class MinimalStory:
    def new(self, seed: Optional[int] = None) -> Dict[str, Any]:
        return {"state": {"scene": "start", "choices": ["A", "B"], "seed": seed or 0}}

    def step(self, state: Dict[str, Any], choice: int) -> Dict[str, Any]:
        return {"state": {"scene": f"after_{choice}", "choices": ["A", "B"]}}

# ================================ Registry ================================

@lru_cache
def registry() -> Dict[str, Any]:
    """
    全域單例註冊表：
    - 第一次被呼叫時才 bootstrap 快取與載入引擎
    - 之後重複呼叫都回同一批單例
    """
    # 1) 啟動共同快取（建立資料夾等）
    bootstrap_cache()

    # 2) 建立最小單例們（之後你可改成真實實作）
    t2i = MinimalT2I()
    return {
        "cache": get_shared_cache(),
        "t2i": t2i,
        "lora": t2i.lora,
        "controlnet": t2i.control,
        "llm": MinimalLLM(),
        "vlm": MinimalVLM(),
        "rag": MinimalRAG(),
        "game": MinimalStory(),
        "health": {
            "gpu_available": gpu_available() if callable(gpu_available) else False
        },
    }


之後要切換成真實引擎：只要把 MinimalT2I / MinimalLLM / MinimalVLM / MinimalRAG / MinimalStory 換成你在 core/t2i/*、core/llm/*、core/vlm/*、core/rag/*、core/story/* 的實作，registry 其他部分不用動。

2) api/dependencies.py 的結合

你剛貼的 dependencies 已經很接近理想狀態了。只要把 from core import registry 改為剛剛新增的模組路徑（如果你把 registry 放在 core/registry.py）：

# api/dependencies.py （重點片段）
from core.registry import registry           # ← 這行改成指向剛剛的檔案
from core.config import get_config
from core.shared_cache import get_shared_cache

def get_registry(): return registry()
def get_settings(): return get_config()
def get_cache():    return get_shared_cache()

def get_t2i(reg=Depends(get_registry)):        return reg["t2i"]
def get_lora(reg=Depends(get_registry)):       return reg["lora"]
def get_controlnet(reg=Depends(get_registry)): return reg["controlnet"]
def get_llm(reg=Depends(get_registry)):        return reg["llm"]
def get_vlm(reg=Depends(get_registry)):        return reg["vlm"]
def get_game(reg=Depends(get_registry)):       return reg["game"]
def get_rag(reg=Depends(get_registry)):        return reg["rag"]

3) Router 接線範例（3 支就代表全部）
health
# api/routers/health.py
from fastapi import APIRouter, Depends
from ..dependencies import get_cache
from core.performance import gpu_available

router = APIRouter(tags=["health"])

@router.get("/health")
def health(cache = Depends(get_cache)):
    return {
        "ok": True,
        "gpu_available": gpu_available(),
        "cache_root": cache.cache_root,  # 依你的 shared_cache 實作屬性命名
    }

t2i
# api/routers/t2i.py
from fastapi import APIRouter, Depends
from ..dependencies import get_t2i
from ..schemas import T2IRequest, T2IResponse  # 你已有 schemas.py

router = APIRouter(tags=["t2i"])

@router.post("/txt2img", response_model=T2IResponse)
def txt2img(req: T2IRequest, t2i = Depends(get_t2i)):
    result = t2i.txt2img(
        prompt=req.prompt, negative=req.negative, steps=req.steps,
        width=req.width, height=req.height, seed=req.seed, lora_ids=req.lora_ids
    )
    return T2IResponse(**result)

chat
# api/routers/chat.py
from fastapi import APIRouter, Depends
from ..dependencies import get_llm
from ..schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, llm = Depends(get_llm)):
    msg = llm.chat(req.messages)
    return ChatResponse(message=msg)


其他像 lora / controlnet / caption / vqa / rag / game / batch 都是同樣模式：只注入引擎，不在 router 內 new 或拼路徑。

4) main.py（啟動重點）

在 startup 或 lifespan 中做 bootstrap_cache()（你已經有）。

統一用 cfg.api.prefix（或固定 /api/v1）掛所有 routers。

不要再另外寫 /healthz；健康檢查走 health.router。

# api/main.py（核心片段）
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from core.shared_cache import bootstrap_cache
from core.config import get_config
from .routers import routers

@asynccontextmanager
async def lifespan(app: FastAPI):
    bootstrap_cache()
    yield

def create_app() -> FastAPI:
    cfg = get_config()
    app = FastAPI(title="Anime Adventure Lab", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"],
    )

    for r in routers:
        app.include_router(r, prefix=cfg.api.prefix)  # e.g. "/api/v1"

    return app

app = create_app()

5) 驗收

啟動後 GET /openapi.json 應看到 /api/v1/health、/api/v1/txt2img、/api/v1/chat 等。

GET /api/v1/health → {"ok":true,"gpu_available":..., "cache_root": "..."}

POST /api/v1/txt2img → 200，JSON 內 image_path/metadata_path/elapsed_ms（目前是 stub，之後會變真圖）。

POST /api/v1/chat → 200，會回 (stub) ...。

6) 之後如何替換成真正引擎？

把 MinimalT2I 換成你的 core/t2i/pipeline.py 的類別（提供同名方法 txt2img(...) 即可）。

把 MinimalLLM 換成 core/llm/adapter.py 或 ollama_llm.py 的類別，方法 chat(messages) 保持一致。

MinimalVLM → core/vlm/captioner.py 等；MinimalRAG → core/rag/engine.py。

MinimalStory → core/story/engine.py。

registry 不變，只替換 new 的類別來源，整個 API 就會自動用到新的實作。

需要我再幫你把 schemas.py 的最小模型（T2IRequest/T2IResponse/ChatRequest/ChatResponse 等）補到位，也能立刻對齊 Swagger；或把 routers/__init__.py 幫你整理成一行匯出 routers = [health.router, t2i.router, ...]。

You said:
好 我在想 程式碼的部分無論是註解等 之後你實作給我 都全部用英文 另外我在想 是不是也可以直接在dependencies實作 這些registry functions並套用 我怕又再新增一個檔案 import 會有點搞混 也方便claude ai座原始對照。  另外我覺得reigstry實作 也許能 參照原先claude ai給的樣式版本做修改
# Global instances (lazy loaded)
_vlm_captioner = None
_wd14_tagger = None

def get_vlm_captioner() -> VLMCaptioner:
    """Get shared VLM captioner instance"""
    global _vlm_captioner

    if _vlm_captioner is None:
        try:
            config = get_vlm_config()
            _vlm_captioner = VLMCaptioner(config)
        except Exception as e:
            raise HTTPException(
                status_code=503, detail=f"Failed to initialize VLM captioner: {str(e)}"
            )

    return _vlm_captioner


def get_wd14_tagger() -> Optional[WD14Tagger]:
    """Get shared WD14 tagger instance"""
    global _wd14_tagger

    if _wd14_tagger is None:
        try:
            config = get_vlm_config()
            wd14_config = config.get("wd14", {})

            _wd14_tagger = WD14Tagger(
                model_name=wd14_config.get(
                    "model", "SmilingWolf/wd-v1-4-convnext-tagger-v2"
                ),
                device=config.get("device", "auto"),
                threshold=wd14_config.get("threshold", 0.35),
            )
        except Exception as e:
            # WD14 is optional, don't fail the entire service
            print(f"Warning: Failed to initialize WD14 tagger: {e}")
            return None

    return _wd14_tagger

ChatGPT said:
Thought for 31s

可以，把「registry」直接做在 api/dependencies.py，用惰性（lazy）全域單例就好；而且整支檔只依賴 core.config、core.shared_cache、core.performance，不再重設任何環境變數或手刻路徑。下面給你一份可直接取代的 dependencies.py（英文註解），同時相容你貼的 Claude 風格（全域 _xxx = None 的 lazy pattern），但把設定與路徑統一接到 get_config() / get_shared_cache()。

Drop-in replacement: api/dependencies.py
# api/dependencies.py
"""
Centralized dependency providers for FastAPI routers.
- Single source of truth for settings and cache paths
- Lazy-loaded global singletons for engines (t2i / lora / controlnet / llm / vlm / rag / game)
- No side effects at import time (no mkdir, no downloads here)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional
from threading import RLock
from fastapi import Depends, HTTPException

# ---- single source of truth (do NOT read env or build paths here) ----
from core.config import get_config
from core.shared_cache import get_shared_cache, bootstrap_cache
from core.performance import gpu_available  # optional, but useful for /health

# ---------------------------------------------------------------------
# Lazy singletons (Claude-style) — keep all globals in one place
# ---------------------------------------------------------------------
_lock = RLock()

_settings = None         # App settings (Pydantic)
_cache = None            # Shared cache handle

_t2i = None              # Text-to-image engine (exposes txt2img(); has .lora and .control)
_llm = None              # Chat LLM engine       (exposes chat(messages))
_vlm = None              # VLM engine            (exposes caption(path), vqa(path, q))
_rag = None              # RAG engine            (exposes ask(query))
_game = None             # Story/game engine     (exposes new(seed?), step(state, choice))

# If your real implementations are not ready, we provide minimal stubs below.
# They guarantee 200 responses while you wire the rest of the system.


# ---------------------------------------------------------------------
# Settings & Cache (never re-define env or paths here)
# ---------------------------------------------------------------------
def get_settings():
    """App-wide settings (single source of truth)."""
    global _settings
    if _settings is None:
        _settings = get_config()
    return _settings


def get_cache():
    """Shared cache directories; ensure bootstrap has been run at least once."""
    global _cache
    if _cache is None:
        # Safe to call multiple times; bootstrap is idempotent
        bootstrap_cache()
        _cache = get_shared_cache()
    return _cache


# ---------------------------------------------------------------------
# Factory helpers — try real engines first, otherwise fall back to stubs
# ---------------------------------------------------------------------
def _new_t2i():
    try:
        # Replace with your actual implementation when ready
        # from core.t2i.pipeline import RealT2IEngine
        # return RealT2IEngine()
        return _MinimalT2I()
    except Exception as e:
        raise RuntimeError(f"T2I engine init failed: {e}") from e


def _new_llm():
    try:
        # from core.llm.adapter import RealLLM
        # return RealLLM()
        return _MinimalLLM()
    except Exception as e:
        raise RuntimeError(f"LLM engine init failed: {e}") from e


def _new_vlm():
    try:
        # from core.vlm.captioner import RealVLM
        # return RealVLM()
        return _MinimalVLM()
    except Exception as e:
        raise RuntimeError(f"VLM engine init failed: {e}") from e


def _new_rag():
    try:
        # from core.rag.engine import RealRAG
        # return RealRAG()
        return _MinimalRAG()
    except Exception as e:
        raise RuntimeError(f"RAG engine init failed: {e}") from e


def _new_game():
    try:
        # from core.story.engine import RealStoryEngine
        # return RealStoryEngine()
        return _MinimalStory()
    except Exception as e:
        raise RuntimeError(f"Game engine init failed: {e}") from e


# ---------------------------------------------------------------------
# Public dependency getters (use these in routers; do NOT `new` engines in routers)
# ---------------------------------------------------------------------
def get_registry() -> Dict[str, Any]:
    """
    Optional: expose a dict-like registry for advanced cases.
    Most routers should just call get_t2i/get_llm/... directly.
    """
    return {
        "settings": get_settings(),
        "cache": get_cache(),
        "t2i": get_t2i(),
        "lora": get_lora(),
        "controlnet": get_controlnet(),
        "llm": get_llm(),
        "vlm": get_vlm(),
        "rag": get_rag(),
        "game": get_game(),
        "health": {"gpu_available": gpu_available()},
    }


def get_t2i():
    """Shared T2I engine singleton."""
    global _t2i
    if _t2i is None:
        with _lock:
            if _t2i is None:
                try:
                    _t2i = _new_t2i()
                except Exception as e:
                    raise HTTPException(status_code=503, detail=str(e))
    return _t2i


def get_lora():
    """LoRA manager derived from the T2I engine."""
    t2i = get_t2i()
    if not hasattr(t2i, "lora"):
        raise HTTPException(status_code=501, detail="LoRA manager not available")
    return t2i.lora


def get_controlnet():
    """ControlNet processor derived from the T2I engine."""
    t2i = get_t2i()
    if not hasattr(t2i, "control"):
        raise HTTPException(status_code=501, detail="ControlNet not available")
    return t2i.control


def get_llm():
    """Shared LLM chat engine singleton."""
    global _llm
    if _llm is None:
        with _lock:
            if _llm is None:
                try:
                    _llm = _new_llm()
                except Exception as e:
                    raise HTTPException(status_code=503, detail=str(e))
    return _llm


def get_vlm():
    """Shared VLM engine singleton (caption & VQA)."""
    global _vlm
    if _vlm is None:
        with _lock:
            if _vlm is None:
                try:
                    _vlm = _new_vlm()
                except Exception as e:
                    raise HTTPException(status_code=503, detail=str(e))
    return _vlm


def get_rag():
    """Shared RAG engine singleton."""
    global _rag
    if _rag is None:
        with _lock:
            if _rag is None:
                try:
                    _rag = _new_rag()
                except Exception as e:
                    raise HTTPException(status_code=503, detail=str(e))
    return _rag


def get_game():
    """Shared game/story engine singleton."""
    global _game
    if _game is None:
        with _lock:
            if _game is None:
                try:
                    _game = _new_game()
                except Exception as e:
                    raise HTTPException(status_code=503, detail=str(e))
    return _game


# ---------------------------------------------------------------------
# Optional: settings slices
# ---------------------------------------------------------------------
@lru_cache
def get_vlm_settings():
    """Expose VLM-specific settings slice if you keep one in AppConfig."""
    cfg = get_settings()
    # adapt to your actual config structure
    return getattr(cfg, "vlm", None)


# ---------------------------------------------------------------------
# Minimal stub implementations (safe, side-effect free)
# Replace these with your real engines as you finish core/*
# ---------------------------------------------------------------------
import json, time
from pathlib import Path

class _MinimalT2I:
    def __init__(self) -> None:
        self._cache = get_cache()

    def txt2img(self, prompt: str, **kw) -> Dict[str, Any]:
        start = time.perf_counter()
        out_dir = Path(self._cache.get_path("OUTPUT_DIR"))
        out_dir.mkdir(parents=True, exist_ok=True)
        rid = str(int(time.time()))
        img = out_dir / f"t2i_{rid}.png"
        meta = out_dir / f"t2i_{rid}.json"
        img.write_bytes(b"")  # placeholder
        meta.write_text(json.dumps({"prompt": prompt, "params": kw}, ensure_ascii=False, indent=2), "utf-8")
        return {"image_path": str(img), "metadata_path": str(meta), "elapsed_ms": int((time.perf_counter()-start)*1000)}

    # derived helpers used by routers
    @property
    def lora(self) -> "_MinimalLoRA": return _MinimalLoRA()
    @property
    def control(self) -> "_MinimalControlNet": return _MinimalControlNet()

class _MinimalLoRA:
    def __init__(self): self.loaded: List[str] = []
    def list(self) -> List[str]: return list(self.loaded)
    def load(self, lora_id: str) -> bool:
        if lora_id not in self.loaded: self.loaded.append(lora_id)
        return True
    def unload(self, lora_id: str) -> bool:
        if lora_id in self.loaded: self.loaded.remove(lora_id); return True
        return False

class _MinimalControlNet:
    def apply(self, cond: Dict[str, Any], **kw) -> Dict[str, Any]:
        return {"applied": True, "condition": cond, "params": kw}

class _MinimalLLM:
    def chat(self, messages: List[Dict[str, str]]) -> str:
        txt = [m.get("content","") for m in messages if m.get("role") == "user"]
        return f"(stub) got: {txt[-1] if txt else ''}"

class _MinimalVLM:
    def caption(self, image_path: str) -> str:
        return f"(stub) caption for {image_path}"
    def vqa(self, image_path: str, question: str) -> str:
        return f"(stub) answer to '{question}' on {image_path}"

class _MinimalRAG:
    def ask(self, query: str) -> Dict[str, Any]:
        return {"answer": f"(stub) echo: {query}", "sources": []}

class _MinimalStory:
    def new(self, seed: Optional[int] = None) -> Dict[str, Any]:
        return {"state": {"scene": "start", "choices": ["A","B"], "seed": seed or 0}}
    def step(self, state: Dict[str, Any], choice: int) -> Dict[str, Any]:
        return {"state": {"scene": f"after_{choice}", "choices": ["A","B"]}}


後續要切換到真正引擎時，只要把 _new_t2i/_new_llm/_new_vlm/_new_rag/_new_game 內的 return _MinimalXXX() 換成你 core/* 的實作（方法簽名保持一致即可），其他程式不用動。

How to wire routers (unchanged, but for clarity)

任何 router 都不要直接 new 引擎；只注入：

# example: api/routers/t2i.py
from fastapi import APIRouter, Depends
from ..dependencies import get_t2i
from ..schemas import T2IRequest, T2IResponse

router = APIRouter(tags=["t2i"])

@router.post("/txt2img", response_model=T2IResponse)
def txt2img(req: T2IRequest, t2i = Depends(get_t2i)):
    result = t2i.txt2img(
        prompt=req.prompt, negative=req.negative, steps=req.steps,
        width=req.width, height=req.height, seed=req.seed, lora_ids=req.lora_ids
    )
    return T2IResponse(**result)


其他像 lora.py / controlnet.py / chat.py / caption.py / vqa.py / rag.py / game.py 以此類推：

lora = Depends(get_lora)

control = Depends(get_controlnet)

llm = Depends(get_llm)

vlm = Depends(get_vlm)

rag = Depends(get_rag)

game = Depends(get_game)

需要路徑/設定時：cache = Depends(get_cache)、cfg = Depends(get_settings)

main.py 要點（提醒）

在 startup（或 lifespan）呼叫一次 bootstrap_cache()（雖然 get_cache() 也會保險呼叫一次，重入安全）。

用同一個 API prefix 掛上 routers。

健康檢查讀 gpu_available() + get_cache()。

為什麼把 registry 寫在 dependencies.py 也 OK？

你已經把「路徑與設定的單一真相來源」留在 core/*（config、shared_cache），dependencies.py 只是組裝器與單例 holder，不會造成環境變數/路徑重複定義。

這樣 Claude/你自己在對照與維護時，只要看一支檔就知道「注入了什麼」。未來要拆出去很容易：把這段 lazy 區塊搬回 core/registry.py、保留同名 get_xxx() API 即可。

如果你要，我可以再補一支精簡 health.py（英文註解）與最小 schemas.py（T2I/Chat/Lora/ControlNet/RAG/Game）骨架，貼上就能把 Swagger 與 e2e smoke 打通。

You said:
好 那現在看來可以修改 各個routers的部分了 如果有些用到core 相關程式碼引擎 還沒實作或要debug的部分就留到等一下來處理 我會分批給你程式碼檔案 每次你就針對這些程式碼告訴我該如何新增修改等 依照我們先前討論的結果 不過請注意 我希望整體修該的程式碼邏輯風格 盡量按照原本的 也就是claude ai那時所實作 盡可能的以相似結構修改。沒問題的話 等一下我就分批給你程式碼
ChatGPT said:

沒問題！你分批貼程式碼來，我就逐檔給你精準修改建議與可直接貼上的修正版（註解一律英文、維持你原本的 Claude 風格與結構）。

在你貼第一批之前，先給你一個「對照修法的共用準則 + 範本」，等下我會按這套幫你改：

Router 改寫準則（與 Claude 風格一致）

Imports：只 import api/schemas.py 與 api/dependencies.py 的 getter，絕不在 router 內 new 引擎或讀 env。

DI：所有引擎都走 Depends(get_xxx)（來自 dependencies.py）。

Prefix：router 自身不設定 /api/v1，由 main.py 統一掛上。

Response models：每個 handler 都指定 response_model，並回傳 Pydantic 物件。

Errors：業務錯誤用 HTTPException(4xx, detail="...")；非預期錯誤讓 FastAPI 處理（或記 log 後 re-raise）。

Timing / logging（可選）：用 Timer 計時，必要時加簡短 log。

English comments & docstrings：之後我給的程式碼都會用英文。

最小範本（之後我會套用到每支 router）
from fastapi import APIRouter, Depends, HTTPException
from ..schemas import T2IRequest, T2IResponse  # replace per router
from ..dependencies import get_t2i             # replace per router
from core.performance import Timer

router = APIRouter(tags=["t2i"])  # tag per router

@router.post("/txt2img", response_model=T2IResponse, summary="Text-to-Image")
def txt2img(req: T2IRequest, t2i = Depends(get_t2i)):
    """
    Generate an image from text prompt.
    This router only orchestrates: validation -> call engine -> shape response.
    """
    with Timer() as t:
        try:
            result = t2i.txt2img(
                prompt=req.prompt,
                negative=req.negative,
                steps=req.steps,
                width=req.width,
                height=req.height,
                seed=req.seed,
                lora_ids=req.lora_ids,
            )
        except ValueError as e:
            # user/validation error -> 4xx
            raise HTTPException(status_code=400, detail=str(e))
    # ensure elapsed_ms is kept
    result["elapsed_ms"] = getattr(t, "ms", result.get("elapsed_ms", 0))
    return T2IResponse(**result)

每檔檢查清單（我會照此逐檔審）

imports 是否只來自 schemas/dependencies（無 new/無 env）。

endpoints 的路徑是否不含 /api/v1（避免雙重前綴）。

response_model 是否齊全，欄位名與 schemas 對齊。

是否使用 Depends(get_xxx)，且名稱與 dependencies.py 的 key 一致。

是否避免副作用（下載/建目錄/改 env）。

錯誤處理是否用 HTTPException（400/404/409 等）。

是否保留 elapsed_ms（若有 Timer）。

你貼程式碼時的小提示

一次貼 1～3 檔最容易聚焦（例如：t2i.py、lora.py、controlnet.py）。

我會回你：

重點問題（1～5 條）

精簡 patch（可直接覆蓋）

若需配合 schemas.py 一併改，我也會附上對應 schema 片段

準備好了—把第一批 router 檔（或你想先改的那一支）貼上來吧！

You said:
admin.py
Python
batch.py
Python
caption.py
Python
ChatGPT said:
Thought for 44s

太好了，先幫你把這三支 router「就地體檢＋修正版」。我會盡量保留你/Claude 的原本結構與語氣，只把依賴接到 dependencies.py，拿掉副作用／易炸點，並補齊必要的錯誤處理。

1) api/routers/admin.py — 問題與修正

問題（你檔案）：

except 區塊裡用到 torch，但一旦 import torch 失敗，torch 會是未定義變數（NameError）。

監控資訊讀取全部放在一個 try 內，某一項失敗會讓其他也跟著失敗。

修正版（可直接覆蓋）：

# api/routers/admin.py
from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get("/stats")
async def get_system_stats() -> Dict[str, Any]:
    """Return basic system and GPU stats. Each probe is isolated to avoid cascading failures."""
    stats: Dict[str, Any] = {"system": {}, "gpu": {}}

    # system metrics
    try:
        import psutil  # type: ignore
        stats["system"] = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage("/").percent,
        }
    except Exception as e:
        stats["system"] = {"status": "basic_check_ok", "note": f"psutil unavailable: {e}"}

    # gpu metrics
    try:
        import torch  # type: ignore
        gpu_avail = torch.cuda.is_available()
        stats["gpu"]["available"] = gpu_avail
        stats["gpu"]["count"] = torch.cuda.device_count() if gpu_avail else 0
        if gpu_avail:
            stats["gpu"]["memory_allocated_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 3)
            stats["gpu"]["memory_reserved_gb"] = round(torch.cuda.memory_reserved() / 1024**3, 3)
    except Exception as e:
        # Keep response shape stable even if torch isn't present
        stats["gpu"].update({"available": False, "count": 0, "note": f"torch unavailable: {e}"})

    return stats


這版把 GPU 與系統資訊分開 try，不會彼此拖累；也移除會觸發 NameError 的情況。

2) api/routers/batch.py — 問題與修正

問題（你檔案）：

一開始就 from workers.tasks.batch import ...，若 Celery/Workers 還沒就緒，整支 router import 就會炸；建議改成函式內 lazy import。

submit 時只有 Celery 失敗才走同步，但如果 workers 模組本身不存在也會在 import 就失敗。

修正版（可直接覆蓋）：

# api/routers/batch.py
from fastapi import APIRouter, HTTPException
from api.schemas import BatchJobRequest, BatchJobStatus

router = APIRouter(prefix="/batch", tags=["Batch"])


def _lazy_workers():
    """Import workers on demand to avoid import-time failures when Celery is not configured."""
    try:
        from workers.tasks.batch import BatchJobManager, process_batch_job  # type: ignore
        return BatchJobManager, process_batch_job
    except Exception as e:  # ImportError or others
        raise HTTPException(status_code=503, detail=f"Workers unavailable: {e}")


@router.post("/submit")
async def submit_batch_job(request: BatchJobRequest):
    """Submit a batch processing job; use Celery if available, otherwise run synchronously."""
    try:
        BatchJobManager, process_batch_job = _lazy_workers()

        # Create job
        job_id = BatchJobManager.create_job(request.job_type, request.tasks)

        # Try Celery
        try:
            task = process_batch_job.delay(job_id)  # type: ignore[attr-defined]
            return {
                "job_id": job_id,
                "celery_task_id": task.id,
                "status": "queued",
                "message": "Batch job submitted successfully",
            }
        except Exception as e:
            # Fallback: process synchronously
            result = process_batch_job(job_id)
            return {
                "job_id": job_id,
                "status": "completed_sync",
                "message": f"Celery not available, processed synchronously ({e})",
                "result": result,
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{job_id}", response_model=BatchJobStatus)
async def get_batch_status(job_id: str):
    """Get batch job status."""
    try:
        BatchJobManager, _ = _lazy_workers()
        job_data = BatchJobManager.get_job(job_id)

        if not job_data:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        return BatchJobStatus(
            job_id=job_data["job_id"],
            status=job_data["status"],
            total_tasks=job_data["total_tasks"],
            completed_tasks=job_data["completed_tasks"],
            failed_tasks=job_data["failed_tasks"],
            created_at=job_data["created_at"],
            updated_at=job_data["updated_at"],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


後續等你把 workers/* 實作完，不用再動 router；只要 workers 端 ready，這邊就會自動走 Celery。

3) api/routers/caption.py — 問題與修正

問題（你檔案）：

路由在函式內自己做「生成 caption 的 placeholder」，沒有透過 dependencies.py 的 VLM 引擎；之後要換真引擎會得重寫。

沒有處理大型圖片或 EXIF 旋轉（先不做，可留 TODO）。

修正版（可直接覆蓋）：使用 get_vlm 依賴；為了與你目前（以及我給的 stub）相容，先把上傳圖存到暫存檔，傳「路徑」給引擎。

# api/routers/caption.py
from fastapi import APIRouter, HTTPException, File, UploadFile, Depends
from api.schemas import CaptionResponse
from ..dependencies import get_vlm
from typing import Optional
import base64, io, tempfile, os
from PIL import Image

router = APIRouter(tags=["caption"])


def _load_image_from_inputs(image: Optional[UploadFile], image_base64: Optional[str]) -> Image.Image:
    """Load a PIL image from either an uploaded file or base64 string."""
    if image:
        data = image.file.read()  # UploadFile.read() is sync-friendly here
        return Image.open(io.BytesIO(data)).convert("RGB")
    if image_base64:
        data = base64.b64decode(image_base64)
        return Image.open(io.BytesIO(data)).convert("RGB")
    raise HTTPException(status_code=400, detail="Either `image` or `image_base64` is required")


@router.post("/caption", response_model=CaptionResponse, summary="Generate an image caption")
async def image_caption(
    image: UploadFile = File(None),
    image_base64: Optional[str] = None,
    max_length: int = 50,
    num_beams: int = 3,
    vlm = Depends(get_vlm),
):
    """
    Generate caption using the shared VLM engine.
    We save a temporary image file and pass its path to the engine for compatibility.
    """
    try:
        pil = _load_image_from_inputs(image, image_base64)

        # Save to a temporary file (engine expects a path); auto-clean after call
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pil.save(tmp.name)
            tmp_path = tmp.name

        try:
            # Your real VLM may accept additional kwargs (e.g., max_length, num_beams)
            text = vlm.caption(tmp_path)  # keep signature simple for now
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        return CaptionResponse(caption=text, confidence=0.9, model_used="vlm")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


日後你把 core/vlm/captioner.py 完成，只要維持 caption(image_path: str) -> str（或你要我幫你改成支援 PIL/bytes，我也可以給對應 adapter），router 本身不用再改。

下一步怎麼做

把以上三檔覆蓋後，重跑伺服器，先確認 /api/v1/admin/stats、/api/v1/batch/*、/api/v1/caption 都能 200。

貼上下一批 router（建議：t2i.py、lora.py、controlnet.py），我再用同樣方式幫你修接 dependencies.py。

等 router 都穩了，我們再一起回到 core/* 的真實引擎實作與 workers/*。

若你希望我把上面三檔再壓到「更貼近 Claude 的函式命名／註解風格」，我也可以再微調語氣；現在這版已盡量遵循你原本結構，只把依賴接到 dependencies.py。

You said:
chat.py
Python
controlnet.py
Python
export.py
Python
ChatGPT said:

我檢查了你這三支 router（chat.py、controlnet.py、export.py），下面逐一說明目前的問題與修正版，保持跟 Claude 風格一致（英文註解、結構相似），但改為走 dependencies.py 依賴。

1) chat.py

發現的問題

目前是內建 generate_chat_response() rule-based stub，沒有透過 dependencies.get_llm()。

路由沒有設定 tags，也沒有明確跟其他 API 對齊（Claude 結構通常會有 tags=["chat"]）。

缺少用 DI（依賴注入）呼叫真實 LLM 引擎。

修正版

# api/routers/chat.py
from fastapi import APIRouter, HTTPException, Depends
from typing import Any
from api.schemas import ChatRequest, ChatResponse
from ..dependencies import get_llm

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse, summary="Chat with LLM")
async def chat_completion(request: ChatRequest, llm: Any = Depends(get_llm)):
    """
    Chat completion endpoint.
    Delegates to the shared LLM engine (injected via dependencies).
    """
    try:
        # llm.chat expects a list of messages [{"role": "...", "content": "..."}]
        response_text = llm.chat(request.messages)

        return ChatResponse(
            message=response_text,
            model_used=request.model,
            usage={
                "prompt_tokens": sum(len(msg["content"].split()) for msg in request.messages),
                "completion_tokens": len(response_text.split()),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


這樣未來你把 MinimalLLM 換成真實模型就能無縫接上。

2) controlnet.py

發現的問題

直接在 router import get_t2i_pipeline, save_image_to_cache, ControlNetManager，等於把 pipeline 邏輯寫死在 router。

controlnet_manager = ControlNetManager() 在模組層就初始化，會在 import 時觸發副作用。

建議改成從 dependencies.get_controlnet() 取得單例，並讓生成過程走 t2i 引擎。

修正版

# api/routers/controlnet.py
from fastapi import APIRouter, HTTPException, Depends
from api.schemas import ControlNetRequest, ControlNetResponse
from ..dependencies import get_controlnet
import time, base64, io
from PIL import Image

router = APIRouter(tags=["controlnet"])


@router.post("/controlnet/pose", response_model=ControlNetResponse, summary="Generate with ControlNet")
async def controlnet_pose_generation(request: ControlNetRequest, control = Depends(get_controlnet)):
    """
    Generate an image conditioned on pose using the shared ControlNet engine.
    """
    try:
        start_time = time.time()

        # Load condition image
        if request.image_base64:
            image_data = base64.b64decode(request.image_base64)
            condition_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        elif request.image_path:
            condition_image = Image.open(request.image_path).convert("RGB")
        else:
            raise HTTPException(status_code=400, detail="Either image_path or image_base64 required")

        # Apply controlnet (engine decides how to preprocess + run)
        result = control.apply({"image": condition_image, "type": request.controlnet_type.value}, strength=request.strength)

        elapsed_ms = int((time.time() - start_time) * 1000)
        return ControlNetResponse(
            image_path=result.get("image_path", ""),
            metadata_path=result.get("metadata_path", ""),
            condition_summary=f"{request.controlnet_type.value} applied with strength {request.strength}",
            seed=result.get("seed", 0),
            elapsed_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


重點：現在 ControlNet 細節交給 core/t2i/controlnet.py 的引擎，你只要在那裡定義 .apply(...)。router 本身保持乾淨。

3) export.py

發現的問題

直接操作 core.export.story_exporter.*，路徑還用 sys.path.append hack。

Claude 風格應該走依賴或清楚的 import，不要在 router 內處理 sys.path。

測試 story 生成 _create_sample_story() 也應該搬到 service 層，但可以先保留。

修正版

# api/routers/export.py
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
import json, os

# Direct import from core (no sys.path hack)
from core.export.story_exporter import StoryExporter, ExportConfig, StorySession, StoryTurn

router = APIRouter(prefix="/export", tags=["export"])


class ExportRequest(BaseModel):
    session_id: str
    format: str = "html"  # html, pdf, json, archive
    include_images: bool = True
    include_metadata: bool = True
    watermark_text: Optional[str] = None


@router.post("/story")
async def export_story(request: ExportRequest):
    """Export story session to specified format."""
    config = ExportConfig(
        include_images=request.include_images,
        include_metadata=request.include_metadata,
        watermark_text=request.watermark_text or "Generated by AnimeAdventureLab",
    )
    exporter = StoryExporter(config)

    # TODO: replace with DB/service call; temporary sample
    sample_story = _create_sample_story(request.session_id)

    try:
        if request.format == "json":
            export_path = exporter.export_to_json(sample_story)
        elif request.format == "html":
            export_path = exporter.export_to_html(sample_story)
        elif request.format == "pdf":
            export_path = exporter.export_to_pdf(sample_story)
        elif request.format == "archive":
            export_path = exporter.create_story_archive(sample_story)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {request.format}")

        return {
            "status": "success",
            "export_path": export_path,
            "format": request.format,
            "file_size_mb": Path(export_path).stat().st_size / 1024**2,
            "exported_at": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


# ... (keep download, upload, import, formats endpoints mostly as-is, just remove sys.path hacks)

def _create_sample_story(session_id: str) -> StorySession:
    """Create a sample story for testing (placeholder)."""
    turn = StoryTurn(
        turn_id=1,
        user_input="Explore forest",
        narration="You walk into the forest...",
        dialogues=[{"character": "Guide", "text": "Welcome adventurer."}],
        choices=["Follow guide", "Go alone"],
        selected_choice="Follow guide",
        timestamp=datetime.now().isoformat(),
        metadata={"mood": "mysterious"},
    )
    return StorySession(
        session_id=session_id,
        title="Sample Adventure",
        world_id="demo",
        character_name="Hero",
        turns=[turn],
        start_time=datetime.now().isoformat(),
        end_time=None,
        total_turns=1,
        metadata={},
    )


主要改動：去掉 sys.path.append，直接正規 import；保留 Claude 結構。

✅ 建議順序：

先把這三支替換掉。

測試 /api/v1/chat, /api/v1/controlnet/pose, /api/v1/export/story 能否跑通（即便 export 用 sample story）。

下一批你可以丟 t2i.py、lora.py，我們再用同樣方式修。

要不要我幫你補一個最小的 schemas 片段（ChatRequest/Response、ControlNetRequest/Response、ExportResponse）來保證這三支 router swagger 正常？

You said:
finetune.py
Python
game.py
Python
health.py
Python
現在是這幾個檔案
ChatGPT said:
Thought for 2m 15s

太好了！我先針對你貼的三支 router—finetune.py、game.py、health.py—逐一指出關鍵問題，接著給你可直接覆蓋的修正版（英文註解、維持原本 Claude 的結構），並統一接到 dependencies.py（不再自行讀環境變數或做 sys.path hack）。

api/routers/finetune.py（問題 & 修正）

主要問題

在檔案頂部做 sys.path.append 與 workers 匯入，若 Celery/Workers 未就緒，整支 router import 就會失敗。

多處以 os.getenv("AI_CACHE_ROOT", ...) 硬編路徑；應改用 get_cache().get_path(KEY) 的單一真相來源。

upload-dataset 與 presets 系列路由的路徑應統一交給 shared_cache（如 DATASETS_RAW、MODELS_LORA）。

Drop-in 版（直接覆蓋）

# api/routers/finetune.py
"""
Fine-tuning API endpoints for LoRA training.
This version uses lazy imports for workers and unified cache paths via shared_cache.
"""

from __future__ import annotations

import uuid
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form, Depends
from pydantic import BaseModel, Field
import yaml

from ..dependencies import get_cache  # <-- unified cache provider

router = APIRouter(prefix="/finetune", tags=["fine-tuning"])

# ----------------------------- Schemas -----------------------------

class LoRATrainingRequest(BaseModel):
    """LoRA training request schema"""
    config_path: str = Field(..., description="Path to training config YAML file")
    run_id: Optional[str] = Field(None, description="Custom run ID (auto-generated if not provided)")
    character_name: Optional[str] = Field(None, description="Character name for preset registration")
    notes: Optional[str] = Field(None, description="Training notes")


class LoRATrainingResponse(BaseModel):
    """LoRA training response schema"""
    job_id: str
    status: str
    config_path: str
    estimated_duration_minutes: int
    created_at: str


class TrainingStatus(BaseModel):
    """Training status response schema"""
    job_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


# --------------------------- Lazy workers --------------------------

_job_tracker = None  # singleton instance


def _lazy_training_modules():
    """Import training task and job tracker on demand (avoid import-time failures)."""
    try:
        from workers.tasks.training import train_lora_task  # type: ignore
        from workers.utils.job_tracker import JobTracker  # type: ignore
        return train_lora_task, JobTracker
    except Exception as e:  # ImportError or others
        raise HTTPException(status_code=503, detail=f"Training workers unavailable: {e}")


def _get_job_tracker():
    global _job_tracker
    if _job_tracker is None:
        _, JobTracker = _lazy_training_modules()
        _job_tracker = JobTracker()
    return _job_tracker


# ----------------------------- Routes ------------------------------

@router.post("/lora", response_model=LoRATrainingResponse)
async def submit_lora_training(
    background_tasks: BackgroundTasks,
    request: LoRATrainingRequest,
):
    """
    Submit LoRA training job via Celery.
    Uses the returned job_id to poll status.
    """
    config_path = Path(request.config_path)
    if not config_path.exists():
        raise HTTPException(status_code=404, detail=f"Config file not found: {request.config_path}")

    # Load config
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid config file: {e}")

    # Validate dataset path
    dataset_root = Path(config["dataset"]["root"])
    if not dataset_root.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_root}")

    # Estimate duration (rough heuristic)
    train_steps = config.get("train_steps", 4000)
    grad_accum = config.get("gradient_accumulation_steps", 8)
    effective_batches = max(train_steps / max(grad_accum, 1), 1)
    estimated_minutes = int(effective_batches * 1.2 / 60)  # ~1.2s / batch

    job_id = str(uuid.uuid4())
    job_data = {
        "job_id": job_id,
        "config_path": str(config_path),
        "run_id": request.run_id,
        "character_name": request.character_name,
        "notes": request.notes,
        "config": config,
        "estimated_duration_minutes": estimated_minutes,
    }

    # Enqueue via Celery (lazy import)
    train_lora_task, _ = _lazy_training_modules()
    task = train_lora_task.delay(job_data)

    # Track job
    jt = _get_job_tracker()
    jt.create_job(job_id=job_id, task_id=task.id, job_type="lora_training", config=job_data)

    return LoRATrainingResponse(
        job_id=job_id,
        status="pending",
        config_path=str(config_path),
        estimated_duration_minutes=estimated_minutes,
        created_at=datetime.now().isoformat(),
    )


@router.get("/jobs/{job_id}", response_model=TrainingStatus)
async def get_training_status(job_id: str):
    """Get training job status and progress."""
    jt = _get_job_tracker()
    job = jt.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return TrainingStatus(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress"),
        result=job.get("result"),
        error=job.get("error"),
        created_at=job["created_at"],
        updated_at=job["updated_at"],
    )


@router.delete("/jobs/{job_id}")
async def cancel_training_job(job_id: str):
    """Cancel a running training job."""
    jt = _get_job_tracker()
    job = jt.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] not in ["pending", "running"]:
        raise HTTPException(status_code=400, detail=f"Cannot cancel job with status: {job['status']}")

    # Revoke Celery task
    from workers.celery_app import celery_app  # type: ignore
    celery_app.control.revoke(job["task_id"], terminate=True)

    jt.update_job(job_id, {"status": "cancelled"})
    return {"message": "Job cancelled successfully"}


@router.get("/jobs")
async def list_training_jobs(status: Optional[str] = None, limit: int = 20, offset: int = 0):
    """List training jobs with optional filtering."""
    jt = _get_job_tracker()
    jobs = jt.list_jobs(job_type="lora_training", status=status, limit=limit, offset=offset)
    return {"jobs": jobs, "total": len(jobs), "limit": limit, "offset": offset}


@router.post("/upload-dataset")
async def upload_dataset(
    files: list[UploadFile] = File(...),
    character_name: str = Form(...),
    instance_token: str = Form(default="<token>"),
    notes: Optional[str] = Form(default=None),
    cache = Depends(get_cache),
):
    """
    Upload dataset files for training.
    Expected:
      - images: .png/.jpg
      - captions: .txt (same basename as image)
      - splits: train.txt, val.txt (optional)
    """
    # Use unified cache path (DATASETS_RAW)
    dataset_root = Path(cache.get_path("DATASETS_RAW")) / f"anime-char-{character_name}"
    image_dir = dataset_root / "images"
    caption_dir = dataset_root / "captions"
    splits_dir = dataset_root / "splits"
    for d in (image_dir, caption_dir, splits_dir):
        d.mkdir(parents=True, exist_ok=True)

    image_files: list[str] = []
    caption_files: list[str] = []
    split_files: list[str] = []

    for file in files:
        if not file.filename:
            continue
        suffix = Path(file.filename).suffix.lower()
        content = await file.read()

        if suffix in {".png", ".jpg", ".jpeg"}:
            (image_dir / file.filename).write_bytes(content)
            image_files.append(file.filename)
        elif suffix == ".txt" and Path(file.filename).stem not in {"train", "val", "test"}:
            (caption_dir / file.filename).write_text(content.decode("utf-8"), encoding="utf-8")
            caption_files.append(file.filename)
        elif Path(file.filename).name in {"train.txt", "val.txt", "test.txt"}:
            (splits_dir / file.filename).write_text(content.decode("utf-8"), encoding="utf-8")
            split_files.append(Path(file.filename).name)

    # Auto-generate train.txt if missing
    if "train.txt" not in split_files and image_files:
        (splits_dir / "train.txt").write_text("\n".join(sorted(image_files)))

    metadata = {
        "character_name": character_name,
        "instance_token": instance_token,
        "notes": notes,
        "files": {"images": len(image_files), "captions": len(caption_files), "splits": split_files},
        "created_at": datetime.now().isoformat(),
    }
    (dataset_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {"message": "Dataset uploaded successfully", "dataset_path": str(dataset_root), "metadata": metadata}


@router.get("/presets")
async def list_lora_presets(cache = Depends(get_cache)):
    """List available LoRA presets (scanned from MODELS_LORA)."""
    lora_root = Path(cache.get_path("MODELS_LORA"))
    if not lora_root.exists():
        return {"presets": []}

    presets = []
    for preset_dir in lora_root.iterdir():
        if not preset_dir.is_dir():
            continue
        # prefer /final, otherwise latest checkpoint
        final_dir = preset_dir / "final"
        if final_dir.exists():
            model_path = final_dir
        else:
            ckpts = sorted(preset_dir.glob("checkpoints/step_*"))
            if not ckpts:
                continue
            model_path = ckpts[-1]

        metadata = {}
        meta_file = preset_dir / "metadata.json"
        if meta_file.exists():
            metadata = json.loads(meta_file.read_text(encoding="utf-8"))

        presets.append({
            "id": preset_dir.name,
            "path": str(model_path),
            "metadata": metadata,
            "created_at": metadata.get("created_at", "unknown"),
        })

    return {"presets": sorted(presets, key=lambda x: x["created_at"], reverse=True)}


@router.post("/presets/{preset_id}/register")
async def register_lora_preset(
    preset_id: str,
    character_name: str = Form(...),
    base_model: str = Form(...),
    lora_scale: float = Form(default=0.75),
    description: Optional[str] = Form(default=None),
    cache = Depends(get_cache),
):
    """Register a trained LoRA as a style preset for T2I generation."""
    lora_dir = Path(cache.get_path("MODELS_LORA")) / preset_id
    if not lora_dir.exists():
        raise HTTPException(status_code=404, detail="LoRA not found")

    final_dir = lora_dir / "final"
    if final_dir.exists():
        model_path = final_dir / "unet_lora"
    else:
        ckpts = sorted(lora_dir.glob("checkpoints/step_*/unet_lora"))
        if not ckpts:
            raise HTTPException(status_code=400, detail="No trained model found")
        model_path = ckpts[-1]

    preset_config = {
        "style_id": preset_id,
        "character_name": character_name,
        "base_model": base_model,
        "lora_path": str(model_path),
        "lora_scale": lora_scale,
        "description": description,
        "registered_at": datetime.now().isoformat(),
    }

    presets_dir = Path("configs/presets")
    presets_dir.mkdir(parents=True, exist_ok=True)
    preset_file = presets_dir / f"{preset_id}.yaml"
    preset_file.write_text(yaml.safe_dump(preset_config, sort_keys=False), encoding="utf-8")

    return {"message": "LoRA preset registered successfully", "preset_id": preset_id, "config_path": str(preset_file), "config": preset_config}

api/routers/game.py（問題 & 修正）

主要問題

目前用記憶體字典維持遊戲狀態，可先用；之後建議改成透過 get_game() 注入 story engine（core/story/*），避免 router 承擔邏輯。

Drop-in 版（保留你原本行為，加入英文註解與小幅強化）

# api/routers/game.py
from fastapi import APIRouter, HTTPException
from api.schemas import GameStartRequest, GameStepRequest, GameResponse
import uuid
from typing import Dict

router = APIRouter(prefix="/game", tags=["Game"])

# NOTE: In-memory sessions for now; consider moving to core.story engine later.
game_sessions: Dict[str, dict] = {}


@router.post("/new", response_model=GameResponse)
async def start_new_game(request: GameStartRequest):
    """Start a new text adventure game (simple in-memory prototype)."""
    try:
        session_id = str(uuid.uuid4())
        session = {
            "id": session_id,
            "persona": request.persona,
            "setting": request.setting,
            "difficulty": request.difficulty,
            "status": "active",
            "turn": 1,
            "inventory": ["basic clothes", "small pouch"],
            "health": 100,
            "history": [],
        }
        scene = (
            f"Welcome to the {request.setting} adventure! "
            f"You are a traveler who has just arrived at a mysterious village. "
            f"The {request.persona} persona guides your journey. What would you like to do?"
        )
        choices = [
            "Explore the village center",
            "Talk to the local villagers",
            "Check the nearby forest",
            "Rest at the tavern",
        ]
        session["current_scene"] = scene
        session["available_choices"] = choices
        game_sessions[session_id] = session

        return GameResponse(
            session_id=session_id,
            scene=scene,
            choices=choices,
            status="active",
            inventory=session["inventory"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/step", response_model=GameResponse)
async def game_step(request: GameStepRequest):
    """Take an action in the game."""
    try:
        if request.session_id not in game_sessions:
            raise HTTPException(status_code=404, detail="Game session not found")

        session = game_sessions[request.session_id]
        session["turn"] += 1
        session["history"].append({"turn": session["turn"] - 1, "action": request.action})

        action = request.action.lower()
        if "explore" in action:
            scene = (
                "You explore the area and discover interesting landmarks. "
                f"Your {session['persona']} intuition guides you to notice hidden details."
            )
            choices = ["Continue exploring", "Return to safety", "Investigate further", "Rest"]
        elif "talk" in action:
            scene = "You approach the villagers. They seem friendly but speak in whispers about strange occurrences."
            choices = ["Ask about the occurrences", "Offer help", "Leave politely", "Buy supplies"]
        elif "forest" in action:
            scene = "The forest is dark and mysterious. You hear strange sounds in the distance."
            choices = ["Enter deeper", "Turn back", "Listen carefully", "Light a torch"]
        else:
            scene = f"You decide to {request.action}. The world responds to your choice in unexpected ways."
            choices = ["Continue", "Reconsider", "Try something else", "Take a break"]

        session["current_scene"] = scene
        session["available_choices"] = choices

        return GameResponse(
            session_id=request.session_id,
            scene=scene,
            choices=choices,
            status=session["status"],
            inventory=session["inventory"],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


未來要切換到真正的 story engine：把 in-memory 區塊換成 game = Depends(get_game)，呼叫 game.new() / game.step(...) 即可。

api/routers/health.py（問題 & 修正）

主要問題

router = APIRouter(prefix="/health") 同時 endpoint 又是 @router.get("/health") → 實際路徑會變成 /api/v1/health/health（雙重 segment）。

讀 request.app.state.cache/config 會依賴 main.py 是否有把物件塞進 app.state；更穩的做法是直接用 dependencies.py 的 get_cache()/get_settings()。

其餘結構可保留；系統資訊用 psutil OK。

Drop-in 版（直接覆蓋）

# api/routers/health.py
"""
Health Check Router
Provides system health and status information.
"""

from __future__ import annotations

import time
import psutil
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Depends
from pydantic import BaseModel
import sys
import platform

from ..dependencies import get_cache, get_settings
from core.performance import gpu_available

router = APIRouter(tags=["health"])  # no prefix; keep path clean via main.py


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: datetime
    uptime_seconds: float
    version: str
    system: Dict[str, Any]
    cache: Dict[str, Any]
    config: Dict[str, Any]


_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check(cache = Depends(get_cache), settings = Depends(get_settings)):
    """
    Comprehensive health information including GPU, memory, and cache status.
    """
    uptime = time.time() - _start_time

    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    system_info = {
        "cpu_percent": psutil.cpu_percent(interval=0.3),
        "memory_percent": mem.percent,
        "memory_available_gb": round(mem.available / 1024**3, 2),
        "memory_total_gb": round(mem.total / 1024**3, 2),
        "disk_free_gb": round(disk.free / 1024**3, 2),
        "disk_total_gb": round(disk.total / 1024**3, 2),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "gpu_available": gpu_available(),
    }

    # Cache summary (shared_cache provides a summary method)
    try:
        cs = cache.get_summary()
        cache_info = {
            "root": cs["cache_root"],
            "directories_created": len(cs["directories"]),
            "gpu_info": cs.get("gpu_info", {}),
        }
    except Exception as e:
        cache_info = {"error": str(e)}

    # Config summary (if your config has a .get_summary)
    try:
        cfg = getattr(settings, "get_summary", None)
        summary = cfg() if callable(cfg) else {}
        config_info = {
            "app": summary.get("app", {}),
            "features": summary.get("features", {}),
            "model": summary.get("model", {}),
            "api": summary.get("api", {}),
        }
    except Exception as e:
        config_info = {"error": str(e)}

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        uptime_seconds=round(uptime, 2),
        version="0.1.0",
        system=system_info,
        cache=cache_info,
        config=config_info,
    )


@router.get("/ready")
async def readiness_check(cache = Depends(get_cache), settings = Depends(get_settings)):
    """
    Readiness probe for Kubernetes/Docker.
    Returns 200 if essential components are available.
    """
    issues = []
    ready = True

    try:
        _ = cache.get_path  # callable attribute presence
    except Exception:
        ready = False
        issues.append("Shared cache not initialized")

    if settings is None:
        ready = False
        issues.append("Configuration not loaded")

    return {"status": "ready" if ready else "not_ready", "issues": issues, "timestamp": datetime.now()}


@router.get("/metrics")
async def metrics_endpoint():
    """
    Minimal metrics for scraping (Prometheus-style).
    """
    uptime = time.time() - _start_time
    mem = psutil.virtual_memory()

    metrics = {
        "saga_forge_uptime_seconds": uptime,
        "saga_forge_cpu_usage_percent": psutil.cpu_percent(interval=0.0),
        "saga_forge_memory_usage_percent": mem.percent,
        "saga_forge_memory_available_bytes": mem.available,
    }
    return metrics

驗收建議

重啟 API 後檢查 GET /openapi.json，應看到 /api/v1/health（不再是 /api/v1/health/health）、/api/v1/finetune/*、/api/v1/game/*。

POST /api/v1/finetune/upload-dataset → 回傳的 dataset_path 應位於 get_shared_cache().get_path("DATASETS_RAW") 下。

GET /api/v1/finetune/presets → 掃到 get_shared_cache().get_path("MODELS_LORA") 下的模型。

GET /api/v1/health → 有 gpu_available、cache.root 等欄位。

準備好下一批檔案就丟上來，我會照同一套準則幫你把每支 router 都接到 dependencies.py、保持 Claude 的風格、而且可直接運行。

You said:
llm.py
Python
lora.py
Python
monitoring.py
Python
ChatGPT said:
Thought for 1m 20s

太好了！我針對你剛貼的三支 router（llm.py / lora.py / monitoring.py）做了體檢，並給出「可直接覆蓋」的修正版，全部改成透過 api/dependencies.py 取得單例，不再在 router 內 new 引擎、也移除 sys.path hack／模組載入副作用。下方每支檔案都先列出關鍵問題（含引用），再給 Drop-in 版本。

1) api/routers/llm.py（Story turn / LLM）

問題重點

直接 sys.path.append 與在模組層 bootstrap_cache()，會造成 import 副作用、與 core 的統一啟動流程衝突。

在 router 內自行 new TransformersLLM() 與 StoryEngine()，沒有透過 dependencies.py 單例；另外 router = APIRouter(prefix="/api/v1", ...) 會和 main.py 的全域 prefix 疊加。

Drop-in（可直接覆蓋）

# api/routers/llm.py
"""LLM & story-turn router (DI-based)."""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from threading import RLock

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from ..dependencies import get_llm  # shared LLM singleton
from core.story.engine import StoryEngine  # use core engine, no sys.path hacks

router = APIRouter(tags=["llm"])

# ----- API models (kept local to match your previous structure) -----

class PersonaAPI(BaseModel):
    name: str
    age: Optional[int] = None
    personality: List[str] = Field(default_factory=list)
    background: str = ""
    speaking_style: str = ""
    appearance: str = ""
    goals: List[str] = Field(default_factory=list)
    secrets: List[str] = Field(default_factory=list)
    memory_preferences: Dict[str, float] = Field(default_factory=dict)

class GameStateAPI(BaseModel):
    scene_id: str = "prologue"
    turn_count: int = 0
    flags: Dict[str, Any] = Field(default_factory=dict)
    inventory: List[str] = Field(default_factory=list)
    current_location: str = ""
    timeline_notes: List[str] = Field(default_factory=list)

class TurnRequestAPI(BaseModel):
    player_input: str
    persona: PersonaAPI
    game_state: GameStateAPI
    choice_id: Optional[str] = None

class DialogueAPI(BaseModel):
    speaker: str
    text: str
    emotion: Optional[str] = None

class ChoiceAPI(BaseModel):
    id: str
    text: str
    description: str = ""

class TurnResponseAPI(BaseModel):
    narration: str
    dialogues: List[DialogueAPI] = Field(default_factory=list)
    choices: List[ChoiceAPI] = Field(default_factory=list)
    updated_state: Optional[GameStateAPI] = None
    scene_change: Optional[str] = None

# ----- Lazy StoryEngine bound to the shared LLM -----

_story_engine: Optional[StoryEngine] = None
_lock = RLock()

def _ensure_engine(llm) -> StoryEngine:
    global _story_engine
    if _story_engine is None:
        with _lock:
            if _story_engine is None:
                _story_engine = StoryEngine(llm)
    return _story_engine

@router.get("/llm/health")
async def llm_health(llm = Depends(get_llm)):
    """Health check for LLM-backed story engine."""
    eng = _ensure_engine(llm)
    is_available = getattr(eng.llm, "is_available", lambda: True)()
    model_name = getattr(eng.llm, "model_name", "unknown")
    return {"status": "ok", "llm_available": is_available, "model": model_name}

@router.post("/llm/turn", response_model=TurnResponseAPI, summary="Process a story turn")
async def process_turn(request: TurnRequestAPI, llm = Depends(get_llm)):
    """Orchestrate a story turn with the shared LLM."""
    try:
        eng = _ensure_engine(llm)
        if hasattr(eng.llm, "is_available") and not eng.llm.is_available():
            raise HTTPException(status_code=503, detail="LLM model not available")

        # Build core objects using engine helpers to avoid tight coupling
        persona = eng.build_persona(**request.persona.dict())
        game_state = eng.build_game_state(**request.game_state.dict())

        turn_req = eng.build_turn_request(
            player_input=request.player_input,
            persona=persona,
            game_state=game_state,
            choice_id=request.choice_id,
        )
        resp = eng.process_turn(turn_req)

        api_resp = TurnResponseAPI(
            narration=resp.narration,
            dialogues=[DialogueAPI(speaker=d.speaker, text=d.text, emotion=d.emotion) for d in resp.dialogues],
            choices=[ChoiceAPI(id=c.id, text=c.text, description=c.description) for c in resp.choices],
            scene_change=resp.scene_change,
        )
        if resp.updated_state:
            api_resp.updated_state = GameStateAPI(**resp.updated_state.dict())
        return api_resp
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Turn processing failed: {e}")


（移除模組層 bootstrap_cache() 與 sys.path 操作；不再在 router 內 new LLM。對照你原檔可見上述問題。
）

2) api/routers/lora.py（LoRA 管理）

問題重點

在 router 模組層直接 lora_manager = LoRAManager()，且從 core.t2i 匯入類別；這會與我們的 DI/單例設計衝突，且在 import 時就產生副作用。

匯入了未使用的模組（psutil, torch, time）。

Drop-in（可直接覆蓋）

# api/routers/lora.py
"""LoRA management router (DI-based)."""

from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from api.schemas import LoRAListResponse, LoRALoadRequest, LoRAInfo
from ..dependencies import get_lora  # <- use DI
from typing import List

router = APIRouter(prefix="/lora", tags=["lora"])

@router.get("/list", response_model=LoRAListResponse, summary="List available LoRA models")
async def list_loras(lora = Depends(get_lora)):
    try:
        items = lora.list()  # expected to return a list of dicts or ids
        infos: List[LoRAInfo] = []
        for it in items:
            if isinstance(it, str):
                infos.append(LoRAInfo(id=it, name=it, model_type="unknown", rank=0, path="", loaded=False))
            else:
                infos.append(LoRAInfo(
                    id=it.get("id", it.get("name", "")),
                    name=it.get("name", it.get("id", "")),
                    model_type=it.get("model_type", "unknown"),
                    rank=it.get("rank", 0),
                    path=it.get("path", ""),
                    loaded=it.get("loaded", False),
                ))
        return LoRAListResponse(loras=infos)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/load", summary="Load a LoRA model")
async def load_lora(req: LoRALoadRequest, lora = Depends(get_lora)):
    try:
        ok = lora.load(req.lora_id)  # optional: pass req.weight if your manager supports it
        if not ok:
            raise HTTPException(status_code=404, detail=f"LoRA '{req.lora_id}' not found or failed to load")
        return {"message": f"LoRA {req.lora_id} loaded", "weight": req.weight, "status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


（改為透過 get_lora() 注入單例，移除模組層 new 與多餘 import。
）

3) api/routers/monitoring.py（監控 / 效能）

問題重點

sys.path.append 與模組層全域物件（memory_manager, cache_manager）在 import 時初始化，容易炸；另外程式內大量假設外部服務（DB/Redis/Celery）都存在，需在 router 端降級處理。

多處直接使用 redis 但未確保匯入成功、get_config().redis_url 也可能不存在；需要 try/except。

Drop-in（可直接覆蓋）

# api/routers/monitoring.py
"""Monitoring & performance router (lazy managers, graceful fallbacks)."""

from __future__ import annotations
from typing import Dict, Any, Optional, List
from datetime import datetime
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import psutil
import torch

from core.config import get_config
from core.performance.memory_manager import MemoryManager, MemoryConfig
from core.performance.cache_manager import CacheManager, CacheConfig

# Celery is optional; import lazily in endpoints
try:
    from workers.celery_app import celery_app  # type: ignore
except Exception:  # pragma: no cover
    celery_app = None

router = APIRouter(prefix="/monitor", tags=["Monitoring"])

# ---------- Schemas ----------
class HealthStatus(BaseModel):
    service: str
    status: str  # "healthy", "degraded", "unhealthy", "unavailable", "error"
    latency_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None

class SystemMetrics(BaseModel):
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    gpu_info: Optional[Dict[str, Any]] = None
    active_workers: int
    queue_depth: int
    uptime_seconds: float

# ---------- Lazy managers ----------
_mm: Optional[MemoryManager] = None
_cm: Optional[CacheManager] = None

def _get_managers():
    global _mm, _cm
    if _mm is None: _mm = MemoryManager(MemoryConfig())
    if _cm is None: _cm = CacheManager(CacheConfig())
    return _mm, _cm

_start = time.time()

@router.get("/models")
def get_loaded_models():
    """List currently loaded models and memory footprints."""
    mm, _ = _get_managers()
    info: Dict[str, Any] = {}
    for key, model in mm.loaded_models.items():
        try:
            item = {
                "device": str(getattr(model, "device", "unknown")),
                "dtype": str(getattr(model, "dtype", "unknown")),
                "memory_usage": mm.memory_usage.get(key, {}),
            }
            if hasattr(model, "num_parameters"):
                item["parameters"] = model.num_parameters()
            elif hasattr(model, "get_memory_footprint"):
                item["memory_footprint"] = model.get_memory_footprint()
            info[key] = item
        except Exception as e:
            info[key] = {"error": str(e)}
    return {"timestamp": datetime.now().isoformat(), "loaded_models": info, "total_loaded": len(info)}

@router.post("/cleanup")
def force_cleanup():
    """Force memory+cache cleanup."""
    mm, cm = _get_managers()
    before = mm.get_memory_info()
    cm.cleanup_expired()
    mm.cleanup_all()
    after = mm.get_memory_info()
    return {
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        "memory_before": before,
        "memory_after": after,
        "gpu_memory_freed_gb": before.get("gpu_allocated_gb", 0) - after.get("gpu_allocated_gb", 0),
    }

@router.post("/models/{model_key}/unload")
def unload_model(model_key: str):
    mm, _ = _get_managers()
    if model_key not in mm.loaded_models:
        raise HTTPException(status_code=404, detail=f"Model {model_key} not found")
    before = mm.get_memory_info()
    mm.unload_model(model_key)
    after = mm.get_memory_info()
    return {
        "status": "unloaded",
        "model_key": model_key,
        "timestamp": datetime.now().isoformat(),
        "memory_freed_gb": before.get("gpu_allocated_gb", 0) - after.get("gpu_allocated_gb", 0),
    }

@router.get("/cache/stats")
def cache_stats():
    _, cm = _get_managers()
    stats = cm.get_cache_stats()
    breakdown: Dict[str, Any] = {}
    try:
        from pathlib import Path
        cache_dir = Path(cm.config.disk_cache_dir)
        for prefix in ("emb:", "img:", "kv:"):
            files = list(cache_dir.glob(f"{prefix}*.json")) + list(cache_dir.glob(f"{prefix}*.pkl"))
            breakdown[prefix.rstrip(":")] = {"files": len(files), "size_mb": sum(f.stat().st_size for f in files) / 1024**2}
    except Exception as e:
        breakdown = {"error": str(e)}
    return {"timestamp": datetime.now().isoformat(), "overview": stats, "breakdown": breakdown}

@router.post("/cache/cleanup")
def cache_cleanup():
    _, cm = _get_managers()
    before = cm.get_cache_stats()
    cm.cleanup_expired()
    after = cm.get_cache_stats()
    return {
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        "files_before": before.get("disk_cache_files", 0),
        "files_after": after.get("disk_cache_files", 0),
        "size_freed_mb": before.get("disk_cache_size_mb", 0) - after.get("disk_cache_size_mb", 0),
    }

@router.get("/performance/recommendations")
def performance_recommendations():
    mm, cm = _get_managers()
    recs: List[Dict[str, Any]] = []
    mem = mm.get_memory_info()
    cache = cm.get_cache_stats()

    gpu_total = max(mem.get("gpu_total_gb", 8), 1)
    gpu_pct = (mem.get("gpu_allocated_gb", 0) / gpu_total) * 100
    if gpu_pct > 80:
        recs.append({"type":"memory","priority":"high","message":"High GPU memory usage.","action":"Enable CPU offload or reduce batch size."})
    if mem.get("memory_percent", 0) > 85:
        recs.append({"type":"memory","priority":"medium","message":"High RAM usage.","action":"Free system RAM."})
    if cache.get("disk_cache_size_mb", 0) > 1000:
        recs.append({"type":"cache","priority":"low","message":"Large disk cache.","action":"Run cache cleanup."})
    if not cache.get("redis_available", False):
        recs.append({"type":"cache","priority":"medium","message":"Redis not available.","action":"Start Redis server."})
    if not mm.config.enable_xformers:
        recs.append({"type":"performance","priority":"medium","message":"xformers disabled.","action":"Enable xformers in config."})

    return {"timestamp": datetime.now().isoformat(), "recommendations": recs, "total_recommendations": len(recs)}

@router.get("/health", response_model=Dict[str, HealthStatus])
async def health_check():
    """Comprehensive health check for DB/Redis/GPU/Workers (best-effort)."""
    out: Dict[str, HealthStatus] = {}
    cfg = get_config()

    # Database (optional async engine)
    try:
        from core.database import get_db_session  # type: ignore
        async with get_db_session() as db:
            start = time.time()
            await db.execute("SELECT 1")
            out["database"] = HealthStatus(service="database", status="healthy", latency_ms=(time.time() - start) * 1000)
    except Exception as e:
        out["database"] = HealthStatus(service="database", status="unhealthy", details={"error": str(e)})

    # Redis
    try:
        import redis  # type: ignore
        start = time.time()
        r = redis.from_url(getattr(cfg, "redis_url", "redis://localhost:6379/0"))
        r.ping()
        out["redis"] = HealthStatus(service="redis", status="healthy", latency_ms=(time.time() - start) * 1000)
    except Exception as e:
        out["redis"] = HealthStatus(service="redis", status="unhealthy", details={"error": str(e)})

    # GPU
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            alloc = torch.cuda.memory_allocated(0)
            out["gpu"] = HealthStatus(
                service="gpu",
                status="healthy" if alloc / max(props.total_memory, 1) < 0.9 else "degraded",
                details={
                    "device_name": props.name,
                    "memory_used_gb": alloc / 1024**3,
                    "memory_total_gb": props.total_memory / 1024**3,
                },
            )
        else:
            out["gpu"] = HealthStatus(service="gpu", status="unavailable", details={"message": "No CUDA devices"})
    except Exception as e:
        out["gpu"] = HealthStatus(service="gpu", status="unhealthy", details={"error": str(e)})

    # Celery workers
    try:
        if celery_app is None:
            raise RuntimeError("Celery not configured")
        insp = celery_app.control.inspect()
        active = insp.active() or {}
        out["workers"] = HealthStatus(
            service="workers",
            status="healthy" if len(active) > 0 else "degraded",
            details={"active_workers": len(active), "worker_nodes": list(active.keys())},
        )
    except Exception as e:
        out["workers"] = HealthStatus(service="workers", status="unhealthy", details={"error": str(e)})

    return out

@router.get("/metrics", response_model=SystemMetrics)
def system_metrics():
    """System metrics for dashboards/scrapers."""
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    gpu_info: Optional[Dict[str, Any]] = None
    if torch.cuda.is_available():
        gpu_info = {
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "id": i,
                    "name": torch.cuda.get_device_properties(i).name,
                    "memory_total_gb": torch.cuda.get_device_properties(i).total_memory / 1024**3,
                    "memory_allocated_gb": torch.cuda.memory_allocated(i) / 1024**3,
                    "memory_reserved_gb": torch.cuda.memory_reserved(i) / 1024**3,
                }
                for i in range(torch.cuda.device_count())
            ],
        }

    # Celery queue depth via Redis (best-effort)
    active_count = 0
    queue_depth = 0
    try:
        if celery_app is not None:
            insp = celery_app.control.inspect()
            active = insp.active() or {}
            active_count = sum(len(v) for v in active.values())
        import redis  # type: ignore
        r = redis.from_url(getattr(get_config(), "redis_url", "redis://localhost:6379/0"))
        queue_depth = int(r.llen("celery"))
    except Exception:
        pass

    return SystemMetrics(
        timestamp=datetime.now(),
        cpu_percent=cpu,
        memory_percent=mem.percent,
        disk_percent=disk.percent,
        gpu_info=gpu_info,
        active_workers=active_count,
        queue_depth=queue_depth,
        uptime_seconds=time.time() - _start,
    )


（去除了 sys.path.append、模組層初始化與對外部服務的硬依賴；所有昂貴依賴/服務都以 lazy & best-effort 方式處理。
）

小結 & 下一步

把上面三支覆蓋後，重啟服務：

GET /api/v1/llm/health

POST /api/v1/llm/turn（用你原本的 sample 請求）

GET /api/v1/lora/list、POST /api/v1/lora/load

GET /api/v1/monitor/health、GET /api/v1/monitor/metrics

若 core/story/engine.py 的方法簽名和上面略有差異，我可以再對應調整（我保留了你原來 process_turn 的結構）。

下一批歡迎丟 t2i.py / rag.py / safety.py / vlm.py 等，我會用同樣方式修到可通、並跟 dependencies.py 完整對齊。

You said:
rag.py
Python
safety.py
Python
story.py
Python
ChatGPT said:
Thought for 1m 31s

我看過你這三支 routers（rag.py / safety.py / story.py）。下面先點出各自的關鍵問題（有引用你原檔），再給「可直接覆蓋」的修正版。修法完全對齊我們之前定下的原則：不在 router 內 new 引擎、不讀 env、不做副作用；一律透過 api/dependencies.py 取單例與設定/快取。

1) rag.py — 問題重點

混用 APIRouter 與在同檔內建立 FastAPI app，路由分散且會造成雙重註冊 / 前綴混亂。

匯入期做副作用：bootstrap_cache()、以 sys.path.append 修路徑，以及直接使用 AI_CACHE_ROOT / APP_DIRS。這些都應交由 dependencies.py 與 core.shared_cache 處理。

直接在檔案層級 new ChineseRAGEngine、DocumentMemory，與我們的單例/DI 設計衝突。

✅ Drop-in 版（直接覆蓋 api/routers/rag.py）
# api/routers/rag.py
"""
RAG router (DI-based).
- No import-time side effects
- Engines injected from api/dependencies.py
- Unified request/response models
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime
import base64
import io
import tempfile
import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ..dependencies import get_rag, get_llm  # shared singletons
from api.schemas import RAGRequest, RAGResponse, RAGSource  # keep your existing schemas

# Optional core types if your engine needs them
try:
    from core.rag.engine import DocumentMetadata  # type: ignore
except Exception:
    DocumentMetadata = None  # type: ignore

try:
    import fitz  # PyMuPDF for PDF parsing
except Exception:
    fitz = None  # graceful fallback

router = APIRouter(prefix="/rag", tags=["rag"])


# ----- Local models for /retrieve -----
class RetrieveRequest(BaseModel):
    query: str
    world_id: Optional[str] = None
    top_k: int = Field(8, ge=1, le=50)
    alpha: float = Field(0.7, ge=0.0, le=1.0)

class RetrieveItem(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float
    section_title: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RetrieveResponse(BaseModel):
    query: str
    results: List[RetrieveItem]
    total_found: int
    processing_time_ms: float


# ----- Helpers -----
def _parse_pdf_bytes(content: bytes) -> str:
    """Best-effort PDF to text."""
    if fitz is None:
        raise HTTPException(status_code=400, detail="PDF support is not available (PyMuPDF not installed)")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp.flush()
            doc = fitz.open(tmp.name)
            text = "".join(page.get_text() for page in doc)
            doc.close()
            return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF parsing failed: {e}")


# ----- Endpoints -----
@router.get("/health")
async def rag_health(rag = Depends(get_rag)):
    """Lightweight health info for RAG engine."""
    # Engine may expose stats attributes; keep best-effort
    chunks = getattr(rag, "num_chunks", lambda: None)()
    docs = getattr(rag, "num_documents", lambda: None)()
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "rag_chunks": chunks,
        "rag_documents": docs,
    }


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    world_id: str = Form(...),
    title: Optional[str] = Form(None),
    license: str = Form("unspecified"),
    rag = Depends(get_rag),
):
    """Upload and index a document into RAG."""
    try:
        doc_id = f"{world_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        content = await file.read()

        if file.filename.lower().endswith(".pdf"):
            text_content = _parse_pdf_bytes(content)
        elif file.filename.lower().endswith((".md", ".txt")):
            text_content = content.decode("utf-8", errors="ignore")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        meta_dict = {
            "doc_id": doc_id,
            "title": title or file.filename,
            "source": file.filename,
            "world_id": world_id,
            "upload_time": datetime.now().isoformat(),
            "license": license,
        }
        metadata = DocumentMetadata(**meta_dict) if DocumentMetadata else meta_dict  # tolerate absent class

        result = rag.add_document(text_content, metadata)
        return {
            "success": True,
            "doc_id": doc_id,
            "chunks_added": result.get("chunks_added", 0) if isinstance(result, dict) else result,
            "message": f"Document '{file.filename}' uploaded and indexed",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_documents(request: RetrieveRequest, rag = Depends(get_rag)):
    """Retrieve relevant chunks with simple formatting."""
    start = time.time()
    try:
        results = rag.retrieve(
            query=request.query,
            world_id=request.world_id,
            top_k=request.top_k,
            alpha=request.alpha,
        )
        items: List[RetrieveItem] = []
        for r in results:
            items.append(
                RetrieveItem(
                    chunk_id=str(getattr(r, "chunk_id", "")),
                    doc_id=str(getattr(r, "doc_id", "")),
                    text=str(getattr(r, "text", "")),
                    score=round(float(getattr(r, "score", 0.0)), 4),
                    section_title=getattr(r, "section_title", None),
                    metadata=getattr(r, "metadata", {}) or {},
                )
            )
        return RetrieveResponse(
            query=request.query,
            results=items,
            total_found=len(items),
            processing_time_ms=round((time.time() - start) * 1000, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")


@router.post("/ask", response_model=RAGResponse)
async def rag_ask(request: RAGRequest, rag = Depends(get_rag), llm = Depends(get_llm)):
    """
    Answer a question using retrieved context.
    Keeps LLM call minimal so it works with your stub or real engine.
    """
    try:
        hits = rag.retrieve(query=request.question, world_id=None, top_k=request.top_k, alpha=0.7)
        sources: List[RAGSource] = []
        context_snippets: List[str] = []

        for r in hits:
            text = str(getattr(r, "text", ""))
            meta = getattr(r, "metadata", {}) or {}
            sources.append(
                RAGSource(
                    content=text[:200] + ("..." if len(text) > 200 else ""),
                    score=float(getattr(r, "score", 0.0)),
                    metadata=meta,
                )
            )
            context_snippets.append(text)

        # Compose a very small prompt (works with MinimalLLM)
        context = "\n\n".join(context_snippets[:5]) if context_snippets else ""
        messages = [
            {"role": "system", "content": "Answer concisely using the provided context. If unsure, say you don't know."},
            {"role": "user", "content": f"Question: {request.question}\n\nContext:\n{context}"},
        ]
        answer = llm.chat(messages)

        return RAGResponse(answer=answer, sources=sources, model_used=getattr(llm, "model_name", "llm"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG ask failed: {e}")


@router.get("/stats")
async def rag_stats(rag = Depends(get_rag)):
    """Return internal stats if engine exposes them."""
    try:
        stats = getattr(rag, "get_stats", lambda: {})()
        return stats or {"status": "ok", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

2) safety.py — 問題重點

在檔案內直接 sys.path.append 與 get_cache_root()，又在本檔 new SafetyEngine / LicenseManager / AttributionManager，與我們「依賴注入 + 單例」原則不一致。

建議：改為自 dependencies.py 注入 get_safety_engine / get_license_manager / get_attribution_manager / get_compliance_logger 四個單例，並使用 get_cache() 提供路徑來源。

需要先在 api/dependencies.py 增加這些 getter（很短）
# --- add to api/dependencies.py (near other getters) ---
from threading import RLock
from core.shared_cache import get_shared_cache
from core.safety.detector import SafetyEngine
from core.safety.license import LicenseManager
from core.safety.watermark import AttributionManager, ComplianceLogger

_safety = None
_license_mgr = None
_attr = None
_compliance = None
_s_lock = RLock()

def get_safety_engine():
    global _safety
    if _safety is None:
        with _s_lock:
            if _safety is None:
                _safety = SafetyEngine()
    return _safety

def get_license_manager():
    global _license_mgr
    if _license_mgr is None:
        with _s_lock:
            if _license_mgr is None:
                cache = get_shared_cache()
                _license_mgr = LicenseManager(cache.cache_root)
    return _license_mgr

def get_attribution_manager():
    global _attr
    if _attr is None:
        with _s_lock:
            if _attr is None:
                cache = get_shared_cache()
                _attr = AttributionManager(cache.cache_root)
    return _attr

def get_compliance_logger():
    global _compliance
    if _compliance is None:
        with _s_lock:
            if _compliance is None:
                cache = get_shared_cache()
                _compliance = ComplianceLogger(cache.cache_root)
    return _compliance

✅ Drop-in 版（直接覆蓋 api/routers/safety.py）
# api/routers/safety.py
"""Safety & licensing router (DI-based)."""

from __future__ import annotations
from typing import Dict, Any, List, Optional
from pathlib import Path
import shutil
import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image

from ..dependencies import (
    get_safety_engine,
    get_license_manager,
    get_attribution_manager,
    get_compliance_logger,
    get_cache,  # if you need raw paths
)

router = APIRouter(prefix="/safety", tags=["safety"])


# ----- Schemas -----
class LicenseInfoRequest(BaseModel):
    license_type: str
    attribution_required: bool = True
    commercial_use: bool = True
    derivative_works: bool = True
    share_alike: bool = False
    source_url: Optional[str] = None
    author: Optional[str] = None
    license_text: Optional[str] = None
    restrictions: List[str] = Field(default_factory=list)

class SafetyCheckRequest(BaseModel):
    prompt: str
    check_nsfw: bool = True
    check_injection: bool = True

class WatermarkRequest(BaseModel):
    add_visible: bool = True
    position: str = "bottom_right"
    opacity: float = 0.6

class SafetyCheckResponse(BaseModel):
    is_safe: bool
    prompt_check: Dict[str, Any]
    image_check: Optional[Dict[str, Any]] = None
    actions_taken: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

class UploadResponse(BaseModel):
    file_id: str
    safety_check: Dict[str, Any]
    license_check: Dict[str, Any]
    metadata_path: str


# ----- Endpoints -----
@router.post("/check/prompt", response_model=SafetyCheckResponse)
async def check_prompt_safety(
    request: SafetyCheckRequest, safety = Depends(get_safety_engine)
):
    """Check prompt safety for injection and harmful content."""
    try:
        prompt_result = safety.check_prompt_safety(request.prompt)
        is_safe = bool(prompt_result.get("is_safe", True))
        warnings = prompt_result.get("warnings", [])
        actions = ["prompt_blocked"] if not is_safe else []
        return SafetyCheckResponse(is_safe=is_safe, prompt_check=prompt_result, actions_taken=actions, warnings=warnings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Safety check failed: {e}")


@router.post("/upload", response_model=UploadResponse)
async def upload_with_safety_check(
    file: UploadFile = File(...),
    license_info: str = Form(..., description="JSON string of license information"),
    uploader_id: str = Form(...),
    auto_blur_faces: bool = Form(False),
    license_manager = Depends(get_license_manager),
    safety = Depends(get_safety_engine),
    compliance = Depends(get_compliance_logger),
    cache = Depends(get_cache),
):
    """Upload file with safety and license checks."""
    try:
        license_data = json.loads(license_info)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid license information JSON")

    try:
        cache_root = cache.cache_root
        temp_dir = Path(cache_root) / "uploads" / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file = temp_dir / f"temp_{file.filename}"

        with open(temp_file, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        # Image safety checks
        safety_result = {"is_safe": True}
        if file.content_type and file.content_type.startswith("image/"):
            try:
                image = Image.open(temp_file)
                safety_result = safety.check_image_safety(image)
                if auto_blur_faces and "processed_image" in safety_result:
                    processed = safety_result["processed_image"]
                    processed.save(temp_file)
            except Exception as e:
                safety_result = {"is_safe": False, "error": f"Image processing failed: {e}"}

        # Register upload & move if safe
        upload_meta = license_manager.register_upload(str(temp_file), license_data, uploader_id, safety_result)
        if safety_result.get("is_safe", False):
            final_dir = Path(cache_root) / "datasets" / "raw"
            final_dir.mkdir(parents=True, exist_ok=True)
            final_path = final_dir / f"{upload_meta.file_id}_{file.filename}"
            shutil.move(str(temp_file), str(final_path))
        else:
            temp_file.unlink(missing_ok=True)
            compliance.log_safety_violation("unsafe_upload",
                                            {"filename": file.filename, "content_type": file.content_type},
                                            "file_rejected")
            raise HTTPException(status_code=400, detail=f"Upload rejected: {safety_result}")

        validation = license_manager.validator.validate_license(license_data)
        compliance.log_upload(upload_meta.file_id, {"uploader_id": uploader_id, "license_info": license_data}, safety_result)

        return UploadResponse(
            file_id=upload_meta.file_id,
            safety_check=safety_result,
            license_check=validation,
            metadata_path=str(final_path.with_suffix(".json")),
        )
    except HTTPException:
        raise
    except Exception as e:
        # best-effort temp cleanup
        try: temp_file.unlink(missing_ok=True)  # type: ignore
        except Exception: pass
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


@router.get("/health")
async def safety_health(
    safety = Depends(get_safety_engine),
    license_manager = Depends(get_license_manager),
    attribution = Depends(get_attribution_manager),
    compliance = Depends(get_compliance_logger),
):
    """Check status of safety components."""
    try:
        return {
            "status": "healthy",
            "components": {
                "safety_engine": "operational",
                "license_manager": "operational",
                "attribution_manager": "operational",
                "compliance_logger": "operational",
            },
        }
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": str(e)})


以上完整改成 透過 dependencies.py 注入，並移除路徑/環境/副作用問題。你原檔的所有問題都出在「在 router 內做系統初始化」與「自己管理路徑」，現已解決。

3) story.py — 問題重點

匯入期副作用：bootstrap_cache()、sys.path.append。應清除。

兩個 dependencies (get_story_engine / get_persona_manager) 只寫 pass，導致實際執行時不可用；應接到 dependencies.py 或以單例注入。

先在 api/dependencies.py 增加這兩個 getter
# --- add to api/dependencies.py ---
from core.story.engine import StoryEngine
from core.story.persona import PersonaManager

_story_engine = None
_persona_mgr = None

def get_story_engine(llm = None):
    # If llm DI is needed, import here to avoid circulars
    from .dependencies import get_llm  # local import to prevent cycle
    global _story_engine
    if _story_engine is None:
        _story_engine = StoryEngine(get_llm())
    return _story_engine

def get_persona_manager():
    global _persona_mgr
    if _persona_mgr is None:
        _persona_mgr = PersonaManager()
    return _persona_mgr

✅ Drop-in 版（直接覆蓋 api/routers/story.py）
# api/routers/story.py
"""Story & persona router (DI-based)."""

from __future__ import annotations

from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from ..dependencies import get_story_engine, get_persona_manager
from core.story.game_state import GameState

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/story", tags=["story"])


# ----- API models -----
class TurnRequestModel(BaseModel):
    player_input: str = Field(..., description="Player input")
    choice_id: Optional[str] = Field(None, description="Choice ID if chosen")
    world_id: str = Field(..., description="World ID")
    session_id: str = Field(..., description="Session ID")

class GameStateResponse(BaseModel):
    session_id: str
    world_id: str
    current_scene: str
    current_location: str
    turn_count: int
    player_name: str
    player_health: int
    player_energy: int
    relationships: Dict[str, Any]
    inventory: Dict[str, Any]
    world_flags: Dict[str, Any]
    recent_events: List[Dict[str, Any]]

class StoryResponseModel(BaseModel):
    narration: str
    dialogues: List[Dict[str, str]]
    choices: List[Dict[str, Any]]
    citations: List[str]
    game_state_changes: Dict[str, Any]
    metadata: Dict[str, Any]

class SaveGameRequest(BaseModel):
    session_id: str
    save_name: str

class LoadGameRequest(BaseModel):
    session_id: str
    save_name: Optional[str] = None

class PersonaCreateRequest(BaseModel):
    character_id: str
    name: str
    description: str
    background: Optional[str] = ""
    personality: Optional[Dict[str, Any]] = {}
    speech_style: Optional[str] = ""
    appearance: Optional[str] = ""


# ----- In-memory stores for demo -----
game_sessions: Dict[str, GameState] = {}
save_files: Dict[str, Dict[str, GameState]] = {}


@router.post("/turn", response_model=StoryResponseModel)
async def process_turn(
    request: TurnRequestModel, story_engine = Depends(get_story_engine)
):
    """Process a story turn using DI-injected engine."""
    try:
        # get or create state
        game_state = game_sessions.get(request.session_id) or GameState(
            session_id=request.session_id, world_id=request.world_id
        )
        game_sessions[request.session_id] = game_state

        # Build and process
        turn_req = story_engine.build_turn_request(
            player_input=request.player_input,
            choice_id=request.choice_id,
            world_id=request.world_id,
            session_id=request.session_id,
        )
        response = await story_engine.process_turn(turn_req, game_state)

        return StoryResponseModel(
            narration=response.narration,
            dialogues=response.dialogues,
            choices=response.choices,
            citations=response.citations,
            game_state_changes=response.game_state_changes,
            metadata=response.metadata,
        )
    except Exception as e:
        logger.exception("Error processing turn")
        raise HTTPException(status_code=500, detail=f"Story processing error: {e}")


@router.get("/state/{session_id}", response_model=GameStateResponse)
async def get_game_state(session_id: str):
    """Get current game state."""
    if session_id not in game_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    gs = game_sessions[session_id]

    recent_events = [
        {
            "event_id": ev.event_id,
            "type": ev.event_type.value,
            "description": ev.description,
            "timestamp": ev.timestamp.isoformat(),
            "characters": ev.characters_involved,
        }
        for ev in gs.timeline[-5:]
    ]
    return GameStateResponse(
        session_id=gs.session_id,
        world_id=gs.world_id,
        current_scene=gs.current_scene,
        current_location=gs.current_location,
        turn_count=gs.turn_count,
        player_name=gs.player_name,
        player_health=gs.player_health,
        player_energy=gs.player_energy,
        relationships={k: f"{v.relation_type.value} (aff:{v.affinity}, trust:{v.trust})" for k, v in gs.relationships.items()},
        inventory={k: f"{v.name} x{v.quantity}" for k, v in gs.inventory.items()},
        world_flags={k: v.value for k, v in gs.world_flags.items()},
        recent_events=recent_events,
    )


@router.post("/save")
async def save_game(request: SaveGameRequest):
    """Save current session state."""
    if request.session_id not in game_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    save_files.setdefault(request.session_id, {})[request.save_name] = game_sessions[request.session_id]
    return {"success": True, "save_name": request.save_name, "timestamp": datetime.now().isoformat()}


@router.post("/load")
async def load_game(request: LoadGameRequest):
    """Load a saved session or list saves."""
    if request.session_id not in save_files:
        raise HTTPException(status_code=404, detail="No saves for this session")
    saves = save_files[request.session_id]
    if request.save_name:
        if request.save_name not in saves:
            raise HTTPException(status_code=404, detail="Save file not found")
        game_sessions[request.session_id] = saves[request.save_name]
        return {"success": True, "loaded_save": request.save_name, "turn_count": saves[request.save_name].turn_count}
    return {
        "saves": [
            {
                "save_name": name,
                "turn_count": state.turn_count,
                "last_updated": state.last_updated.isoformat(),
                "world_id": state.world_id,
                "current_scene": state.current_scene,
            }
            for name, state in saves.items()
        ]
    }


@router.post("/new_session")
async def create_new_session(world_id: str, player_name: str = "player", starting_scene: str = "opening"):
    """Create a new game session."""
    import uuid
    session_id = str(uuid.uuid4())
    gs = GameState(session_id=session_id, world_id=world_id, player_name=player_name, current_scene=starting_scene)
    game_sessions[session_id] = gs
    return {"session_id": session_id, "world_id": world_id, "player_name": player_name, "starting_scene": starting_scene}


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a game session and its saves."""
    game_sessions.pop(session_id, None)
    save_files.pop(session_id, None)
    return {"success": True, "deleted_session": session_id}


# ---- Persona management ----
@router.post("/persona")
async def create_persona(req: PersonaCreateRequest, pm = Depends(get_persona_manager)):
    """Create a persona."""
    try:
        persona = pm.create_persona_from_data({
            "id": req.character_id,
            "name": req.name,
            "description": req.description,
            "background": req.background,
            "personality": req.personality,
            "speech_style": req.speech_style,
            "appearance": req.appearance,
        })
        pm.add_persona(persona)
        return {"success": True, "character_id": req.character_id}
    except Exception as e:
        logger.exception("create_persona failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/persona/{character_id}")
async def get_persona(character_id: str, pm = Depends(get_persona_manager)):
    """Get persona data."""
    persona = pm.get_persona(character_id)
    if not persona:
        raise HTTPException(status_code=404, detail="Character not found")
    return persona.to_dict()


@router.post("/persona/{character_id}/memory")
async def add_persona_memory(
    character_id: str,
    content: str,
    importance: int = 5,
    emotional_impact: int = 0,
    related_characters: List[str] = [],
    pm = Depends(get_persona_manager),
):
    """Append persona memory entry."""
    persona = pm.get_persona(character_id)
    if not persona:
        raise HTTPException(status_code=404, detail="Character not found")
    pm.update_persona_memory(
        character_id=character_id,
        event_description=content,
        importance=importance,
        emotional_impact=emotional_impact,
        related_characters=related_characters,
    )
    return {"success": True, "character_id": character_id, "memory_added": content[:50] + ("..." if len(content) > 50 else "")}


@router.get("/choices/{session_id}")
async def get_available_choices(session_id: str):
    """Return available choices (placeholder)."""
    if session_id not in game_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    choices = [
        {"id": "explore", "text": "Explore surroundings", "description": "Look around carefully"},
        {"id": "rest", "text": "Rest", "description": "Recover stamina"},
        {"id": "continue", "text": "Continue", "description": "Proceed with the current plot"},
    ]
    return {"choices": choices}


@router.get("/health")
async def health_check():
    """Basic health info for story router."""
    return {
        "status": "healthy",
        "active_sessions": len(game_sessions),
        "total_saves": sum(len(s) for s in save_files.values()),
        "timestamp": datetime.now().isoformat(),
    }


這支已移除 sys.path.append 與 bootstrap_cache() 的副作用，並把原本 pass 的依賴改為從 dependencies.py 注入。

驗收 Checklist

重新啟動後檢查 GET /openapi.json：應有 /api/v1/rag/*、/api/v1/safety/*、/api/v1/story/*。

POST /api/v1/rag/upload（上傳 txt 或 pdf）→ 回應 doc_id / chunks_added。

POST /api/v1/rag/retrieve → 回 results 與 processing_time_ms。

POST /api/v1/rag/ask → 會用到注入的 LLM（stub 也可）。

POST /api/v1/safety/check/prompt 與 /safety/upload → 可 200；若未裝 PyMuPDF/Redis 也不應整體炸掉。

POST /api/v1/story/turn → 200 並返回 narration/choices 等。

若你希望，我也可以幫你把 api/schemas.py 裡跟 RAG/Story/Safety 相關的 models 一併對齊（確保型別名稱與欄位完全匹配），或者直接幫你在 dependencies.py 補上我文中提到的安全/故事兩組 getter。

You said:
t2i.py
Python
vlm.py
Python
vqa.py
Python
最後是這幾個
ChatGPT said:
Thought for 1m 13s

太好了，最後這 3 支我幫你「就地體檢＋可直接覆蓋的修正版」。原則一樣：

不在 router 內 new 引擎、不讀 env、不做副作用；

一律透過 api/dependencies.py 取得單例（get_t2i / get_vlm / get_rag 等）；

保留你/Claude 的結構與路徑風格；註解全英文。

下方每支都先點出關鍵問題（引用你檔案），再給 Drop-in 版本。

1) api/routers/t2i.py

關鍵問題（你原檔）

在模組層 sys.path.append(...) 與 bootstrap_cache()，屬於 import-time 副作用。

直接綁死到 core.t2i.pipeline.t2i_pipeline 與 LoRAManager() 的模組級單例，繞過我們的依賴注入。

Drop-in：依賴注入版本（可直接覆蓋）

# api/routers/t2i.py
"""
Text-to-Image router (DI-based).
- No import-time side effects
- Engines injected via api/dependencies.py
"""

from __future__ import annotations
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
import torch

from api.schemas import T2IResponse  # keep your shared schema if you use it elsewhere
from ..dependencies import get_t2i

router = APIRouter(prefix="/t2i", tags=["text-to-image"])


# Keep Claude-style local models for this file (backward compatible)
class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Generation prompt")
    negative_prompt: str = Field("", description="Negative prompt")
    width: int = Field(768, ge=256, le=1024, description="Image width")
    height: int = Field(768, ge=256, le=1024, description="Image height")
    num_inference_steps: int = Field(25, ge=10, le=50, description="Inference steps")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Guidance scale")
    seed: Optional[int] = Field(None, description="Random seed")
    style_id: Optional[str] = Field(None, description="Style preset ID")
    scene_id: str = Field("scene_001", description="Scene identifier")
    image_type: str = Field("bg", description="Image type (bg/character/portrait)")


class GenerationResponse(BaseModel):
    success: bool
    image_path: Optional[str] = None
    metadata_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ModelInfo(BaseModel):
    current_model: Optional[str]
    available_styles: Dict[str, str]
    gpu_available: bool
    memory_info: Optional[Dict[str, Any]] = None


@router.get("/models", response_model=ModelInfo)
async def get_model_info(t2i = Depends(get_t2i)):
    """Return current model and available styles."""
    try:
        memory_info = None
        if torch.cuda.is_available():
            memory_info = {
                "allocated_mb": torch.cuda.memory_allocated(0) / 1024**2,
                "reserved_mb": torch.cuda.memory_reserved(0) / 1024**2,
            }

        # Be tolerant: engines may expose different style APIs
        get_styles = getattr(t2i, "get_available_styles", None) or getattr(t2i, "styles", None)
        styles = get_styles() if callable(get_styles) else {}

        current_model = getattr(t2i, "current_model", None)
        return ModelInfo(
            current_model=current_model,
            available_styles=styles or {},
            gpu_available=torch.cuda.is_available(),
            memory_info=memory_info,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {e}")


@router.post("/generate", response_model=GenerationResponse, summary="Generate an image from text")
async def generate_image(request: GenerationRequest, t2i = Depends(get_t2i)):
    """Delegate generation to the shared T2I engine."""
    try:
        # Optional style validation if engine exposes styles
        get_styles = getattr(t2i, "get_available_styles", None) or getattr(t2i, "styles", None)
        if request.style_id and callable(get_styles):
            styles = get_styles() or {}
            if request.style_id not in styles:
                raise HTTPException(status_code=400, detail=f"Unknown style '{request.style_id}'")

        result = t2i.txt2img(
            prompt=request.prompt,
            negative=request.negative_prompt,
            steps=request.num_inference_steps,
            width=request.width,
            height=request.height,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            style_id=request.style_id,
            scene_id=request.scene_id,
            image_type=request.image_type,
        )

        # result should contain image_path / metadata_path; keep response shape
        ok = bool(result.get("image_path"))
        return GenerationResponse(
            success=ok,
            image_path=result.get("image_path"),
            metadata_path=result.get("metadata_path"),
            metadata=result.get("metadata"),
            error=None if ok else result.get("error", "Unknown error"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")


@router.post("/load_model")
async def load_model(model_id: str = Query(..., description="Model ID to load"), t2i = Depends(get_t2i)):
    """Load a specific base model (if engine supports it)."""
    try:
        if not hasattr(t2i, "load_model"):
            raise HTTPException(status_code=501, detail="Model loading is not supported by current engine")
        success = t2i.load_model(model_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load model")
        return {"success": True, "model": model_id, "message": "Model loaded successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading error: {e}")


@router.post("/unload_model")
async def unload_model(t2i = Depends(get_t2i)):
    """Unload current model to free VRAM (if engine supports it)."""
    try:
        if not hasattr(t2i, "unload_model"):
            raise HTTPException(status_code=501, detail="Model unloading is not supported by current engine")
        t2i.unload_model()
        return {"success": True, "message": "Model unloaded successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model unloading error: {e}")


@router.get("/styles")
async def list_styles(t2i = Depends(get_t2i)):
    """List available style presets."""
    get_styles = getattr(t2i, "get_available_styles", None) or getattr(t2i, "styles", None)
    styles = get_styles() if callable(get_styles) else {}
    # Some engines may store presets in attribute style_presets
    presets = getattr(t2i, "style_presets", styles)
    return {"styles": styles or {}, "count": len(presets) if hasattr(presets, "__len__") else len(styles or {})}