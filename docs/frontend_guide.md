# 前端操作指南（Gradio）

本指南說明 `frontend/gradio/app.py` 的使用方式、可用功能，以及對應的後端 API 端點。

## 環境與啟動

- 後端預設位址：`http://localhost:8000/api/v1`。可用環境變數覆寫：
  - `API_BASE` 或 `API_URL`（會自動補上 `/api/v1`）。
- 模型 mock：
  - `VLM_MOCK=1`、`T2I_MOCK=1` 可避免下載大模型；要真實推理改為 `0` 並確保模型/網路可用。
- 啟動前端：
  ```bash
  API_BASE=http://localhost:8000 python frontend/gradio/app.py
  ```
  預設埠 `7860`（可透過 Gradio 參數調整）。

### 一鍵啟動範例（本機）

1. 安裝依賴（已完成可略）：
   ```bash
   pip install -r requirements.txt -r requirements-test.txt
   ```
2. 啟動後端（mock 模式避免下載模型）：
   ```bash
   export VLM_MOCK=1 T2I_MOCK=1 AI_CACHE_ROOT=/mnt/c/AI_LLM_projects/ai_warehouse
   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```
   - 健康檢查：`http://localhost:8000/api/v1/health`
3. 另開終端啟動前端：
   ```bash
   export API_BASE=http://localhost:8000
   python frontend/gradio/app.py  # 預設 http://localhost:7860
   ```
4. 如需 Celery/Redis（批次/訓練非模擬）：
   ```bash
   # 需先啟動 redis-server
   REDIS_URL=redis://localhost:6379/0 celery -A workers.celery_app:celery_app worker -Q default,vision,text,training -l info
   ```
5. 真實推理：將 `VLM_MOCK=0`、`T2I_MOCK=0`，並確保模型已放入 `AI_CACHE_ROOT` 或可連網下載。

## Celery / Redis 使用說明

### 原理簡述
- **Redis**：作為 Celery 的 broker（傳遞任務）與 backend（儲存任務結果/狀態）。本專案預設 `REDIS_URL=redis://localhost:6379/0`。
- **Celery Worker**：負責從 Redis 取出任務執行，如批次 caption/VQA/chat、訓練/LoRA 任務等。前端/後端提交任務後，Worker 在背景處理並回寫狀態。
- **批次/訓練流程**：
  1. 前端或 API 呼叫 `/batch/submit` 或 `/finetune/lora` 建立任務。
  2. 產生 `job_id`（內部 sqlite/檔案紀錄）與可選的 `task_id`（Celery 任務 ID）。
  3. Worker 從 Redis 取任務執行，狀態可查 `/batch/status/{job_id}`、`/batch/progress/{task_id}` 或 `/jobs/{job_id}`。

### 啟動步驟
1. 啟動 Redis（本機示例）：
   ```bash
   redis-server --port 6379
   ```
2. 啟動後端 API（確保 REDIS_URL 環境一致）：
   ```bash
   export REDIS_URL=redis://localhost:6379/0
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```
3. 啟動 Celery Worker（可多進程/多機）：
   ```bash
   export REDIS_URL=redis://localhost:6379/0
   celery -A workers.celery_app:celery_app worker -Q default,vision,text,training -l info
   ```
   - `-Q default,vision,text,training` 對應路由分類；如只需要批次文字任務，可精簡為 `-Q default,text`。
4. （可選）啟動 Flower 監控：
   ```bash
   celery -A workers.celery_app:celery_app flower --port=5555
   ```
   Flower Web UI 可觀察任務狀態、隊列與 Worker 健康。

### 前端對應
- Batch 分頁：`/batch/submit`、`/batch/status/{job_id}`、`/batch/progress/{task_id}`、`/batch/queues`、`/batch/results/{job_id}`。
- LoRA/訓練：`/finetune/lora`、`/jobs`、`/jobs/{job_id}`。
- 監控：`/monitoring/health` 會檢查 Redis 連線與 Celery worker 回應數；`/monitoring/metrics`、`/monitoring/performance` 顯示系統/性能。

## 分頁與功能對照

### 🩺 系統狀態
- 健康檢查：`GET /health`
- 顯示 CPU/Memory/Disk/GPU/Cache 等資訊。

### 🧠 Agent 任務
- 任務執行：`POST /agent/task`
- 工具列表：`GET /agent/tools`
- 指定工具呼叫：`POST /agent/tools/call`

### 🖼️ 圖像理解
- 圖片 Caption：`POST /caption`（表單檔案 + 參數）
- VQA 問答：`POST /vqa`（表單檔案 + question）

### 🎨 文字生圖
- T2I 生成：`POST /t2i/txt2img`
- 參數：prompt/negative_prompt/width/height/steps/guidance/seed。

### 📖 Story 冒險
- 建立故事：`POST /story/session`
- 行動回合：`POST /story/turn`
- 會話詳情：`GET /story/session/{session_id}`
- 上下文快照：`GET /story/session/{session_id}/context`

### 📚 RAG 管理
- 上傳文件：`POST /rag/upload`（world_id, tags, file）
- 搜尋：`POST /rag/search`
- RAG 問答：`POST /rag/ask`
- 索引統計：`GET /rag/stats`
- 重建：`POST /rag/rebuild`
- 清空：`DELETE /rag/clear`
- 列出文件：`GET /rag/documents`（並可下拉預覽 doc/chunk metadata）
- 刪除文件/Chunk：`DELETE /rag/documents/{doc_id}`

### 📦 批次任務
- 提交 caption/vqa/chat 批次：`POST /batch/submit`（支援 simulate=true）
- 查詢狀態：`GET /batch/status/{job_id}`
- 查詢進度：`GET /batch/progress/{task_id}`
- Queue 統計：`GET /batch/queues`
- 下載/預覽結果：`GET /batch/results/{job_id}`（提供 download_url `/batch/download/{job_id}`；前端會載入 JSON 並顯示前 5 筆）

### 🔬 LoRA / 訓練
- 提交 LoRA Job：`POST /finetune/lora`
- 列出 Jobs：`GET /jobs`
- 查詢 Job：`GET /jobs/{job_id}`

### 📈 監控與管理
- 系統指標：`GET /monitoring/metrics`
- 性能報告：`GET /monitoring/performance`
- Celery/Redis/RAG 健康：`GET /monitoring/health`
- 系統資訊：`GET /admin/system`
- 模型卸載：`POST /admin/models/control`（action=unload_all）

## 操作流程示例

1) **啟動後端**（建議 mock 模式）：`VLM_MOCK=1 T2I_MOCK=1 uvicorn api.main:app --reload`
2) **啟動前端**：`API_BASE=http://localhost:8000 python frontend/gradio/app.py`
3) 在 Gradio UI：
   - 「系統狀態」檢查健康。
   - 「Agent 任務」輸入任務描述並執行；必要時展開工具列表或直接呼叫工具。
   - 「圖像理解」上傳圖片做 Caption/VQA。
   - 「文字生圖」輸入提示，檢視生成圖。
   - 「Story 冒險」建立會話、提交行動並查看詳情/上下文。
   - 「RAG 管理」上傳文件、搜尋/問答、重建/清空索引、預覽或刪除 doc。
   - 「批次任務」提交 caption/vqa/chat 批次，可查狀態、進度、Queue、下載/預覽結果。
   - 「LoRA/訓練」提交與查詢 Jobs。
   - 「監控與管理」查看系統/性能、Celery/Redis/RAG 健康，或卸載模型。

## 常見注意事項

- 無網路或無 GPU 時，請用 mock 環境變數避免模型下載/初始化失敗。
- RAG/批次/LoRA 需要對 `AI_CACHE_ROOT` 有寫入權限。
- 若批次結果檔路徑不可讀，前端預覽會顯示錯誤訊息；需確認檔案路徑及權限。
