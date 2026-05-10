# GPU 本地部署指南（RTX 5080 16GB / Qwen 7B / SDXL）

本文件目標：在「單卡 RTX 5080 16GB VRAM」環境中，讓 Story（含 agents）+ SDXL 圖像生成可完整串起來跑。

## 1) 建議目錄（符合 `~/Desktop/data_model_structure.md`）

本專案會依 `core/shared_cache.py` 使用以下預設根目錄：

- Cache：`/mnt/c/ai_cache`（HF/torch/XDG cache）
- Models：`/mnt/c/ai_models`
- Outputs：`/mnt/c/ai_output/anime-adventure-lab`

建議你把模型放這些路徑：

- Qwen（LLM）：`/mnt/c/ai_models/llm/qwen2.5-7b-instruct/`
- SDXL（base）：`/mnt/c/ai_models/stable-diffusion/xl/sdxl-base-1.0/`（diffusers 格式目錄）
- SDXL LoRA：`/mnt/c/ai_models/diffusion/lora/sdxl/<lora_id>/*.safetensors`
- RAG Embedding：`/mnt/c/ai_models/embeddings/bge-m3/`（可選：也可只用 HF cache）
- RAG Reranker：`/mnt/c/ai_models/reranker/bge-reranker-v2-m3/`（可選）
- LLM LoRA：`/mnt/c/ai_models/llm/lora/<lora_id>/`

> 也支援 SDXL 單檔 checkpoint（`.safetensors`），但建議優先用 diffusers 目錄格式，LoRA/管線相容性更穩。

## 2) conda 環境

```bash
conda create -n ai_env python=3.10 -y
conda activate ai_env
pip install -r requirements.txt
pip install -r requirements-test.txt
```

## 3) 建議環境變數（本地模型 / 避免寫入 $HOME）

```bash
# cache (避免寫到 ~/.cache)
export AI_CACHE_ROOT=/mnt/c/ai_cache
export AI_MODELS_ROOT=/mnt/c/ai_models
export AI_OUTPUT_ROOT=/mnt/c/ai_output/anime-adventure-lab
export HF_HOME=/mnt/c/ai_cache/huggingface
export TRANSFORMERS_CACHE=/mnt/c/ai_cache/huggingface
export TORCH_HOME=/mnt/c/ai_cache/torch
export XDG_CACHE_HOME=/mnt/c/ai_cache

# GPU
export CUDA_VISIBLE_DEVICES=0

# LLM（Story/Chat 預設會用 MODEL_CHAT_MODEL）
export MODEL_CHAT_MODEL=/mnt/c/ai_models/llm/qwen2.5-7b-instruct
export MODEL_DEVICE=cuda
export MODEL_DEVICE_MAP=auto
export MODEL_TORCH_DTYPE=float16
export MODEL_USE_4BIT_LOADING=true

# T2I（SDXL）
export MODEL_DEFAULT_SD_MODEL=/mnt/c/ai_models/stable-diffusion/xl/sdxl-base-1.0
export T2I_MOCK=0

# RAG（中英混合 + reranker）
export RAG_EMBEDDING_MODEL=BAAI/bge-m3
export RAG_DEVICE=cpu
export RAG_ENABLE_RERANK=1
export RAG_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
export RAG_RERANKER_DEVICE=cpu
```

## 4) 啟動方式（API + React）

### API
```bash
uvicorn api.main:app --reload
```

檢查：
- `GET http://localhost:8000/api/v1/health`
- `GET http://localhost:8000/api/v1/status`

### React（Vite）
```bash
cd frontend/react
npm install
npm run dev
```

## 5) GPU smoke（建議你第一次裝好模型後跑）

```bash
python scripts/gpu_smoke.py
```

會做：
- torch CUDA/VRAM 檢查
- 用 `MODEL_CHAT_MODEL` 跑一次簡短聊天（Qwen）
- 用 `MODEL_DEFAULT_SD_MODEL` 跑一次小尺寸 SDXL txt2img（會在 outputs 產生檔案）

## 6) 訓練（SDXL LoRA / LLM LoRA）

### 啟動 training worker（可選，但建議）
若你要把訓練/生成丟到背景跑，建議同時起 Celery：

```bash
export REDIS_URL=redis://localhost:6379/0
celery -A workers.celery_app:celery_app worker -l INFO -Q training
```

### 透過 API 提交訓練

SDXL LoRA（風格/角色視覺）：

```bash
curl -s http://localhost:8000/api/v1/finetune/lora \
  -H 'Content-Type: application/json' \
  -d '{
    "job_type": "lora_sdxl",
    "simulate": false,
    "base_model": "/mnt/c/ai_models/stable-diffusion/xl/sdxl-base-1.0",
    "dataset_path": "/mnt/c/ai_datasets/.../my_lora_dataset",
    "output_name": "my_sdxl_lora_v1",
    "config": {
      "resolution": 1024,
      "batch_size": 1,
      "gradient_accumulation_steps": 4,
      "max_steps": 1000,
      "learning_rate": 1e-4,
      "lora_rank": 16,
      "mixed_precision": "fp16",
      "save_steps": 500
    }
  }'
```

LLM LoRA（Qwen 對話規則/世界觀語氣）：

```bash
curl -s http://localhost:8000/api/v1/finetune/lora \
  -H 'Content-Type: application/json' \
  -d '{
    "job_type": "llm_lora",
    "simulate": false,
    "base_model": "/mnt/c/ai_models/llm/qwen2.5-7b-instruct",
    "dataset_path": "/mnt/c/ai_datasets/.../train.jsonl",
    "output_name": "my_qwen_lora_v1",
    "config": {
      "max_length": 2048,
      "batch_size": 1,
      "gradient_accumulation_steps": 8,
      "max_steps": 500,
      "learning_rate": 0.0002,
      "use_4bit": true
    }
  }'
```

查詢狀態：

```bash
curl -s http://localhost:8000/api/v1/jobs/<job_id>
```

完成後：
- SDXL LoRA 會出現在：`/mnt/c/ai_models/diffusion/lora/sdxl/<output_name>/`
- LLM LoRA 會出現在：`/mnt/c/ai_models/llm/lora/<output_name>/`
