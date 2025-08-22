# SagaForge 部署指南

## 快速開始 (Quick Start)

### 1. 系統需求

**最低配置**
- CPU: 4 cores, 8GB RAM
- GPU: NVIDIA GTX 1060 6GB+ (或 CPU-only 模式)
- 硬碟: 50GB 可用空間
- 作業系統: Ubuntu 20.04+, Windows 10+, macOS 12+

**推薦配置**
- CPU: 8+ cores, 16GB+ RAM
- GPU: NVIDIA RTX 3070 8GB+ 或 RTX 4060 Ti 16GB+
- 硬碟: 100GB+ SSD
- Docker 與 Docker Compose 已安裝

### 2. 環境準備

```bash
# 1. 克隆專案
git clone https://github.com/your-org/saga-forge.git
cd saga-forge

# 2. 建立共用模型倉儲
mkdir -p ../ai_warehouse/cache
export AI_CACHE_ROOT="$(pwd)/../ai_warehouse/cache"

# 3. 複製環境配置
cp .env.example .env
# 編輯 .env 檔案，設定你的配置
```

**關鍵環境變數 (.env)**
```bash
# 必填
AI_CACHE_ROOT=/path/to/ai_warehouse/cache
POSTGRES_PASSWORD=your_secure_password
MINIO_PASSWORD=your_minio_password

# 可選
CUDA_VISIBLE_DEVICES=0
API_CORS_ORIGINS=http://localhost:7860
HUGGINGFACE_TOKEN=your_hf_token  # 用於私有模型
```

### 3. 一鍵部署

**生產環境 (推薦)**
```bash
# 使用 Docker Compose 部署所有服務
docker-compose -f docker-compose.prod.yml up -d

# 等待所有服務啟動 (約 2-5 分鐘)
docker-compose -f docker-compose.prod.yml logs -f

# 檢查健康狀態
curl http://localhost:8000/monitoring/health
```

**開發環境**
```bash
# 安裝 Python 依賴
conda create -n adventure-lab python=3.10 -y
conda activate adventure-lab
pip install -r requirements.txt

# 啟動基礎服務
docker-compose up postgres redis -d

# 啟動 API 伺服器
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 另開終端啟動 worker
celery -A workers.celery_app worker --loglevel=INFO

# 啟動 WebUI
python frontend/gradio/app.py
```

### 4. 驗證部署

```bash
# API 健康檢查
curl http://localhost:8000/healthz

# Web 介面
open http://localhost:7860

# 監控面板
open http://localhost:5555  # Celery Flower

# 系統指標
curl http://localhost:8000/monitoring/metrics
```

---

## 完整配置指南

### 資料庫設定

**PostgreSQL with pgvector**
```sql
-- 初始化腳本 (scripts/init_db.sql)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- 用於全文搜尋

-- 建立向量索引
CREATE INDEX CONCURRENTLY embeddings_vector_idx
ON vectors USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

**Redis 配置優化**
```redis
# redis.conf 建議設定
maxmemory 4gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

### GPU 與記憶體優化

**NVIDIA Docker 設定**
```bash
# 安裝 NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

**低 VRAM 優化 (configs/app.yaml)**
```yaml
performance:
  low_vram_mode: true
  enable_xformers: true
  enable_attention_slicing: true
  enable_cpu_offload: true
  max_batch_size: 1

models:
  default_precision: "fp16"  # 或 "int8" 用於 8-bit 推理
  use_safetensors: true
  cache_models: true
```

### 安全強化

**API 安全**
```python
# 加入認證中介軟體
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def verify_token(token: str = Depends(security)):
    # 實作 JWT 驗證邏輯
    if not verify_jwt(token.credentials):
        raise HTTPException(401, "Invalid token")
    return token

# 保護敏感端點
@router.post("/admin/models")
async def admin_endpoint(token: str = Depends(verify_token)):
    pass
```

**資料加密**
```yaml
# 加密敏感資料
postgres:
  environment:
    POSTGRES_INITDB_ARGS: "--auth-host=scram-sha-256 --auth-local=scram-sha-256"
  volumes:
    - postgres_data:/var/lib/postgresql/data:Z  # SELinux 標籤
```

**網路隔離**
```yaml
# 內部網路配置
networks:
  backend:
    internal: true  # 僅限內部通訊
  frontend:
    # 對外網路
services:
  postgres:
    networks: [backend]
  api:
    networks: [backend, frontend]
```

---

## 維運手冊

### 日常維護

**每日檢查清單**
- [ ] 檢查系統健康狀態: `curl localhost:8000/monitoring/health`
- [ ] 查看錯誤日誌: `docker-compose logs --tail=100 api worker`
- [ ] 監控資源使用: `curl localhost:8000/monitoring/metrics`
- [ ] 檢查任務隊列: `celery -A workers.celery_app inspect active`

**每週維護**
- [ ] 清理舊日誌檔案
- [ ] 檢查磁碟空間使用率
- [ ] 更新系統套件: `apt update && apt upgrade`
- [ ] 備份資料庫和重要檔案

**每月維護**
- [ ] 更新 Docker 映像: `docker-compose pull && docker-compose up -d`
- [ ] 檢查並更新 AI 模型版本
- [ ] 效能基準測試和調優
- [ ] 安全性掃描和更新

### 緊急應變

**服務停機處理**
```bash
# 1. 快速診斷
docker-compose ps
docker-compose logs api worker

# 2. 重啟服務
docker-compose restart api worker

# 3. 如果問題持續，回滾到穩定版本
git checkout v1.0.0
docker-compose down && docker-compose up -d
```

**資料復原**
```bash
# 從備份恢復資料庫
gunzip < backup_20241201.sql.gz | docker-compose exec -T postgres psql -U saga sagaforge

# 恢復模型檔案
rsync -av /backup/ai_warehouse/ /path/to/ai_warehouse/
```

---

## API 文檔

### 核心端點

**健康檢查**
```http
GET /healthz
Response: {"status": "healthy", "version": "1.0.0", "uptime": 3600}
```

**LLM 對話**
```http
POST /llm/turn
Content-Type: application/json

{
    "message": "告訴我關於 Alice 的故事",
    "world_id": "neo_taipei",
    "character_id": "alice",
    "use_rag": true,
    "max_tokens": 500,
    "temperature": 0.7
}

Response:
{
    "narration": "在霓虹燈閃爍的新台北街頭，Alice Chen...",
    "choices": [
        {"id": "choice_1", "text": "繼續探索城市"},
        {"id": "choice_2", "text": "進入咖啡廳"}
    ],
    "citations": ["doc_123@section_1:chunk_2"],
    "game_state": {...}
}
```

**圖像生成**
```http
POST /t2i/generate
Content-Type: application/json

{
    "prompt": "anime girl, cyberpunk city, neon lights",
    "negative_prompt": "blurry, low quality",
    "width": 1024,
    "height": 1024,
    "steps": 30,
    "cfg_scale": 7.5,
    "seed": 42,
    "style_preset": "anime_style"
}

Response:
{
    "image_path": "/warehouse/outputs/20241201/img_001.png",
    "metadata": {
        "prompt": "...",
        "seed": 42,
        "model": "sdxl-base-1.0",
        "generation_time": 15.2
    }
}
```

**RAG 檢索**
```http
POST /rag/retrieve
Content-Type: application/json

{
    "world_id": "neo_taipei",
    "query": "Alice 的背景故事",
    "top_k": 5,
    "rerank": true
}

Response:
{
    "chunks": [
        {
            "id": "doc_123@section_1:chunk_2",
            "text": "Alice Chen 是一位年輕的程式設計師...",
            "score": 0.89,
            "metadata": {"source": "characters.yaml"}
        }
    ],
    "query_time_ms": 45
}
```

### 批次作業

**提交批次任務**
```http
POST /batch/submit
Content-Type: application/json

{
    "world_id": "neo_taipei",
    "job_type": "image_generation",
    "config": {
        "prompts": ["prompt1", "prompt2", "prompt3"],
        "style_preset": "anime_style",
        "generation_params": {
            "width": 512,
            "height": 512,
            "steps": 25
        }
    }
}

Response:
{
    "batch_id": "batch_123456",
    "status": "submitted",
    "total_tasks": 3,
    "estimated_time": 180
}
```

**查詢批次狀態**
```http
GET /batch/{batch_id}/status

Response:
{
    "batch_id": "batch_123456",
    "status": "running",  // pending, running, completed, failed, cancelled
    "progress": {
        "total": 3,
        "completed": 1,
        "failed": 0,
        "remaining": 2
    },
    "results": [
        {
            "task_id": "task_001",
            "status": "completed",
            "output_path": "/warehouse/outputs/batch_123456/img_001.png"
        }
    ]
}
```

### LoRA 訓練

**提交訓練任務**
```http
POST /finetune/lora
Content-Type: application/json

{
    "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
    "dataset_config": {
        "character_name": "alice",
        "data_path": "/warehouse/datasets/alice_images",
        "caption_file": "captions.txt"
    },
    "training_config": {
        "rank": 16,
        "learning_rate": 1e-4,
        "max_steps": 2000,
        "batch_size": 1,
        "resolution": 1024
    },
    "output_name": "alice_v1"
}

Response:
{
    "job_id": "lora_train_789",
    "status": "submitted",
    "estimated_time": 7200,
    "config_path": "/warehouse/configs/lora_train_789.yaml"
}
```

---

## 範例與教學

### 基本使用流程

**1. 上傳世界觀資料**
```python
import requests

# 上傳 worldpack
with open('my_world.zip', 'rb') as f:
    files = {'file': f}
    data = {'world_id': 'my_world', 'license': 'CC-BY-SA-4.0'}
    response = requests.post('http://localhost:8000/rag/upload',
                           files=files, data=data)
print(f"上傳結果: {response.json()}")
```

**2. 開始對話**
```python
# 與 AI 角色對話
conversation = {
    "message": "你好，我想探索這個世界",
    "world_id": "my_world",
    "character_id": "protagonist",
    "use_rag": True
}

response = requests.post('http://localhost:8000/llm/turn', json=conversation)
result = response.json()
print(f"旁白: {result['narration']}")
print(f"選擇: {result['choices']}")
```

**3. 生成場景圖片**
```python
# 根據故事情節生成圖片
image_request = {
    "prompt": "medieval castle, sunset, fantasy landscape",
    "width": 1024,
    "height": 768,
    "steps": 30,
    "style_preset": "fantasy_art"
}

response = requests.post('http://localhost:8000/t2i/generate', json=image_request)
result = response.json()
print(f"圖片已生成: {result['image_path']}")
```

### 進階自動化

**批次生成角色立繪**
```python
import json

# 準備角色資料
characters = [
    {"name": "Alice", "desc": "short black hair, blue eyes, casual clothes"},
    {"name": "Bob", "desc": "tall, brown hair, formal suit"},
    {"name": "Carol", "desc": "long red hair, green dress, friendly smile"}
]

# 批次生成
prompts = []
for char in characters:
    prompt = f"anime style portrait, {char['desc']}, high quality"
    prompts.append(prompt)

batch_job = {
    "world_id": "my_world",
    "job_type": "image_generation",
    "config": {
        "prompts": prompts,
        "style_preset": "anime_portrait",
        "generation_params": {
            "width": 512,
            "height": 768,
            "steps": 25,
            "cfg_scale": 7.0
        }
    }
}

response = requests.post('http://localhost:8000/batch/submit', json=batch_job)
batch_id = response.json()['batch_id']
print(f"批次任務 ID: {batch_id}")

# 監控進度
import time
while True:
    status_response = requests.get(f'http://localhost:8000/batch/{batch_id}/status')
    status = status_response.json()

    print(f"進度: {status['progress']['completed']}/{status['progress']['total']}")

    if status['status'] in ['completed', 'failed']:
        break
    time.sleep(10)
```

### Webhook 整合

**設定完成通知**
```python
# 在批次任務完成時發送通知
webhook_config = {
    "webhook_url": "https://your-app.com/api/batch_complete",
    "events": ["batch_completed", "batch_failed"],
    "secret": "your_webhook_secret"
}

# 系統會在任務完成時發送 POST 請求到你的 webhook
# POST https://your-app.com/api/batch_complete
# {
#   "event": "batch_completed",
#   "batch_id": "batch_123456",
#   "timestamp": "2024-12-01T10:30:00Z",
#   "results": [...]
# }
```

---

## 社群與支援

### 問題回報

**Bug 回報模板**
```markdown
**環境資訊:**
- OS: Ubuntu 22.04
- Docker 版本: 24.0.5
- GPU: NVIDIA RTX 4070
- SagaForge 版本: v1.0.0

**重現步驟:**
1. 執行 `docker-compose up -d`
2. 訪問 http://localhost:8000/healthz
3. 錯誤發生...

**預期行為:**
應該返回 200 OK

**實際行為:**
返回 500 錯誤

**日誌:**
```bash
docker-compose logs api
```

**功能請求模板**
```markdown
**功能描述:**
希望添加支援 Stable Video Diffusion 的影片生成功能

**使用場景:**
為互動故事生成短影片片段

**可能的實作方式:**
- 新增 `/t2v/generate` 端點
- 整合 SVD 模型管道
- 支援關鍵幀控制
```

### 貢獻指南

**開發流程**
1. Fork 專案並建立功能分支
2. 遵循程式碼風格指南 (Black + Ruff)
3. 添加對應的測試案例
4. 更新相關文檔
5. 提交 Pull Request

**程式碼審查標準**
- [ ] 功能正確且完整
- [ ] 測試覆蓋率 > 80%
- [ ] 文檔完整且清晰
- [ ] 遵循專案架構原則
- [ ] 無安全漏洞

### 版本發布

**發布週期**
- **補丁版本** (v1.0.x): 每月，修復 bug 和小改進
- **次要版本** (v1.x.0): 每季，新功能和 API 擴充
- **主要版本** (vX.0.0): 每年，重大架構變更

**升級指南**
```bash
# 備份當前版本
docker-compose down
cp -r ai_warehouse ai_warehouse_backup

# 拉取新版本
git pull origin main
docker-compose pull

# 執行資料庫遷移 (如需要)
python scripts/migrate_database.py

# 重新啟動服務
docker-compose up -d

# 驗證升級
curl http://localhost:8000/healthz
```

**相容性承諾**
- API 端點在主要版本內保持向後相容
- 資料庫遷移腳本確保資料安全升級
- 配置檔案格式變更會提供轉換工具

---

## 附錄

### 預設配置檔案

**configs/app.yaml (完整範例)**
```yaml
# SagaForge 主要配置檔案
app:
  name: "SagaForge"
  version: "1.0.0"
  debug: false
  cors_origins:
    - "http://localhost:7860"
    - "http://localhost:3000"

models:
  llm:
    default: "Qwen/Qwen2.5-7B-Instruct"
    alternatives: ["microsoft/DialoGPT-medium"]
    max_tokens: 2048
    temperature: 0.7

  embedding:
    default: "BAAI/bge-m3"
    dimension: 1024
    normalize: true

  t2i:
    default: "stabilityai/stable-diffusion-xl-base-1.0"
    scheduler: "DPMSolverMultistepScheduler"
    safety_checker: true

  vlm:
    default: "Salesforce/blip2-opt-2.7b"
    max_image_size: 1024

performance:
  low_vram_mode: false
  enable_xformers: true
  enable_attention_slicing: false
  max_batch_size: 4
  cache_models: true

  redis:
    max_connections: 20
    ttl_default: 3600

  celery:
    worker_concurrency: 2
    task_timeout: 1800

security:
  enable_auth: false
  api_rate_limit: "100/minute"
  upload_max_size: "100MB"
  allowed_file_types: [".zip", ".txt", ".md", ".pdf", ".png", ".jpg"]

  content_filter:
    enable_nsfw_filter: true
    enable_face_detection: true
    watermark_generated_images: true

storage:
  backend: "filesystem"  # 或 "minio"
  retention_days: 90
  cleanup_schedule: "0 2 * * *"  # 每日凌晨 2 點清理
```

### 常用命令參考

```bash
# === Docker 相關 ===
# 查看服務狀態
docker-compose ps

# 查看即時日誌
docker-compose logs -f api worker

# 重啟單一服務
docker-compose restart api

# 清理未使用的映像
docker system prune -f

# === 資料庫操作 ===
# 進入 PostgreSQL 控制台
docker-compose exec postgres psql -U saga sagaforge

# 備份資料庫
docker-compose exec postgres pg_dump -U saga sagaforge > backup.sql

# 恢復資料庫
cat backup.sql | docker-compose exec -T postgres psql -U saga sagaforge

# === Celery 任務管理 ===
# 查看活躍任務
celery -A workers.celery_app inspect active

# 清空任務隊列
celery -A workers.celery_app purge

# 監控任務執行
celery -A workers.celery_app events

# === 模型管理 ===
# 下載預設模型
python scripts/download_models.py --all

# 清理模型快取
python scripts/clean_cache.py --models

# 檢查模型完整性
python scripts/verify_models.py

# === 日誌與監控 ===
# 查看 API 錯誤日誌
grep "ERROR" logs/api.log | tail -20

# 監控 GPU 使用率
watch -n 1 nvidia-smi

# 檢查磁碟使用率
df -h /path/to/ai_warehouse
```

---

**🎯 恭喜！** 你現在已經擁有一個完整運行的 SagaForge 系統。開始創建你的互動故事世界吧！

如需更多幫助，請查看：
- 📖 [API 文檔](http://localhost:8000/docs)
- 💬 [社群討論](https://github.com/your-org/saga-forge/discussions)
- 🐛 [問題回報](https://github.com/your-org/saga-forge/issues)
- 📺 [影片教學](https://youtube.com/playlist?list=your-tutorials)與網路配置

**Nginx 反向代理**
```nginx
# nginx.conf
upstream saga_api {
    server api:8000;
}

upstream saga_webui {
    server webui:7860;
}

server {
    listen 80;
    server_name your-domain.com;

    # API 路由
    location /api/ {
        proxy_pass http://saga_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # WebUI 路由
    location / {
        proxy_pass http://saga_webui;
        proxy_set_header Host $host;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

**SSL/TLS 設定**
```bash
# 使用 Let's Encrypt
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

## 故障排除

### 常見問題

**1. GPU 記憶體不足 (CUDA OOM)**
```bash
# 檢查 GPU 使用狀況
nvidia-smi

# 解決方案
# 1. 降低批次大小
# 2. 啟用記憶體優化選項
# 3. 使用 CPU fallback
export CUDA_VISIBLE_DEVICES=""  # 強制使用 CPU
```

**2. 模型下載失敗**
```bash
# 手動下載模型
python scripts/download_models.py --model bge-m3 --cache /path/to/cache

# 或設定 HuggingFace 鏡像
export HF_ENDPOINT=https://hf-mirror.com
```

**3. 資料庫連線問題**
```bash
# 檢查 PostgreSQL 狀態
docker-compose logs postgres

# 重置資料庫
docker-compose down -v
docker-compose up postgres -d
python scripts/init_database.py
```

**4. Celery 任務卡住**
```bash
# 查看 worker 狀態
celery -A workers.celery_app inspect active

# 清空任務隊列
celery -A workers.celery_app purge

# 重啟 worker
docker-compose restart worker
```

### 效能調校

**API 最佳化**
```python
# configs/app.yaml
api:
  workers: 4  # 根據 CPU 核心數調整
  max_requests: 1000
  max_requests_jitter: 50
  timeout: 300

cache:
  embedding_ttl: 3600  # 1小時
  model_cache_size: 5
  redis_max_connections: 20
```

**資料庫最佳化**
```sql
-- 調整 PostgreSQL 設定
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
SELECT pg_reload_conf();
```

### 監控與日誌

**日誌收集**
```yaml
# docker-compose.yml 增加日誌配置
services:
  api:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

**Prometheus 監控 (選配)**
```yaml
# docker-compose.monitoring.yml
services:
  prometheus:
    image: prom/prometheus
    ports: ["9090:9090"]
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports: ["3000:3000"]
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
```

---

## 擴展與客製化

### 新增自訂模型

**1. 註冊新的 LLM 後端**
```python
# core/llm/custom_llm.py
from core.llm.adapter import LLMAdapter

class CustomLLMAdapter(LLMAdapter):
    def __init__(self, model_id: str):
        self.model_id = model_id
        # 初始化你的自訂模型

    async def generate(self, messages: List[dict], **kwargs):
        # 實作生成邏輯
        pass
```

**2. 新增 LoRA 風格預設**
```yaml
# configs/presets/my_style.yaml
style_id: "my_custom_style"
base_model: "stabilityai/stable-diffusion-xl-base-1.0"
lora_path: "../ai_warehouse/models/lora/my_style"
lora_scale: 0.8
trigger_words: ["my_style", "custom_art"]
negative_prompt: "low quality, blurry"
```

### 開發工作流程

**設定開發環境**
```bash
# 安裝開發依賴
pip install -r requirements-dev.txt

# 設定 pre-commit
pre-commit install

# 執行測試
pytest tests/ -v

# 程式碼格式化
black . && ruff --fix .
```

**新功能開發**
```bash
# 建立功能分支
git checkout -b feature/my-new-feature

# 提交遵循 Conventional Commits
git commit -m "feat(api): add new custom endpoint"

# 執行完整測試套件
python -m pytest tests/test_e2e_complete.py::TestE2EWorkflow -v
```

---

## 生產環境最佳實務

### 備份策略

**資料庫備份**
```bash
# 每日自動備份腳本
#!/bin/bash
docker-compose exec postgres pg_dump -U saga sagaforge | gzip > backup_$(date +%Y%m%d).sql.gz

# 保留最近 30 天的備份
find ./backups -name "backup_*.sql.gz" -mtime +30 -delete
```

**模型與快取備份**
```bash
# 同步模型倉儲到備份位置
rsync -av --progress /path/to/ai_warehouse/ /backup/ai_warehouse/

# 或使用 MinIO 的跨區域複製
mc mirror local/models s3/backup-bucket/models
```

### 高可用性部署

**負載平衡器設定**
```yaml
# docker-compose.ha.yml
services:
  api:
    deploy:
      replicas: 3
    depends_on:
      - postgres
      - redis

  worker:
    deploy:
      replicas: 2
```

**資料庫集群 (進階)**
```bash
# PostgreSQL 主從複製設定
# 1. 設定主節點
# 2. 配置從節點
# 3. 設定自動故障轉移
```

### 安全