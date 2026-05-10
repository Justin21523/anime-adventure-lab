# Testing Modes Configuration

本專案支援兩種運行模式：**Mock/Stub 模式（預設）** 和 **真實模型模式**。

## 🎭 Mock/Stub 模式（預設，推薦用於開發和測試）

Mock 模式使用輕量級的模擬實現，不會載入任何真實的 AI 模型。

### 優點：
- ✅ 不下載大型 AI 模型（節省數十 GB 空間）
- ✅ 快速啟動（秒級啟動 vs 分鐘級）
- ✅ 低資源消耗（不需要 GPU，最小化 CPU/內存使用）
- ✅ 適合 API 開發、前端開發、單元測試
- ✅ 適合 CI/CD 環境

### 配置：

編輯 `.env` 文件：

```bash
# Mock 模式配置（預設）
T2I_MOCK=1              # 文生圖模擬模式
VLM_MOCK=1              # 視覺語言模型模擬模式
LLM_MOCK=1              # 語言模型模擬模式
MODEL_DEVICE="cpu"      # 使用 CPU
MODEL_TORCH_DTYPE="float32"
```

### Mock 模式行為：

1. **LLM (語言模型)**
   - 返回格式化的模擬回應：`"[mock reply] {user_input}"`
   - 不載入 Qwen/Llama 等模型

2. **T2I (文生圖)**
   - 返回模擬圖片路徑
   - 不載入 Stable Diffusion 模型

3. **VLM (視覺語言模型)**
   - 返回模擬的圖片描述和 VQA 回答
   - 不載入 BLIP/LLaVA 等模型

4. **RAG (檢索增強生成)**
   - 返回模擬的搜尋結果
   - 不載入嵌入模型

---

## 🚀 真實模型模式（用於生產和完整測試）

真實模式會載入完整的 AI 模型進行推理。

### 要求：
- ⚠️ 需要 NVIDIA GPU（推薦 16GB+ VRAM）
- ⚠️ 需要大量磁碟空間（50-100GB）
- ⚠️ 首次運行會自動下載模型（可能需要 30 分鐘到數小時）

### 配置：

編輯 `.env` 文件：

```bash
# 真實模式配置
T2I_MOCK=0              # 啟用真實文生圖
VLM_MOCK=0              # 啟用真實 VLM
LLM_MOCK=0              # 啟用真實 LLM
MODEL_DEVICE="cuda:0"   # 使用 GPU
MODEL_TORCH_DTYPE="float16"  # 使用 FP16 精度
CUDA_VISIBLE_DEVICES="0"     # 指定 GPU 設備
```

### 真實模式會載入的模型：

1. **LLM**: `Qwen/Qwen2.5-7B-Instruct` (~14GB)
2. **T2I**: `stabilityai/stable-diffusion-xl-base-1.0` (~7GB)
3. **VLM**: `Salesforce/blip2-opt-2.7b` (~5GB)
4. **Embedding**: `BAAI/bge-m3` (~2GB)

---

## 📋 快速切換模式

### 切換到 Mock 模式：
```bash
# 在 .env 中設定
T2I_MOCK=1
VLM_MOCK=1
LLM_MOCK=1
MODEL_DEVICE="cpu"

# 重啟後端
conda activate ai_env
python api/main.py
```

### 切換到真實模式：
```bash
# 在 .env 中設定
T2I_MOCK=0
VLM_MOCK=0
LLM_MOCK=0
MODEL_DEVICE="cuda:0"

# 重啟後端
conda activate ai_env
python api/main.py
```

---

## 🧪 測試建議

### 開發階段（使用 Mock 模式）：
- ✅ API 端點開發和測試
- ✅ 前端 UI 開發
- ✅ 資料庫操作測試
- ✅ 單元測試和整合測試
- ✅ CI/CD 流程

### 整合測試階段（使用真實模式）：
- ✅ AI 生成品質驗證
- ✅ 性能基準測試
- ✅ 端到端測試
- ✅ 用戶驗收測試

---

## 🔍 驗證當前模式

### 檢查後端狀態：
```bash
curl http://localhost:8000/health
```

回應會顯示當前模式：
```json
{
  "status": "healthy",
  "models": {
    "llm": "mock-llm",      // Mock 模式
    "t2i": "mock-sd",       // Mock 模式
    "vlm": "mock"           // Mock 模式
  }
}
```

或者（真實模式）：
```json
{
  "status": "healthy",
  "models": {
    "llm": "Qwen/Qwen2.5-7B-Instruct",
    "t2i": "stabilityai/stable-diffusion-xl-base-1.0",
    "vlm": "Salesforce/blip2-opt-2.7b"
  }
}
```

---

## ⚙️ 環境變數完整參考

| 變數 | Mock 模式 | 真實模式 | 說明 |
|------|----------|----------|------|
| `T2I_MOCK` | `1` | `0` | 文生圖模式 |
| `VLM_MOCK` | `1` | `0` | VLM 模式 |
| `LLM_MOCK` | `1` | `0` | LLM 模式 |
| `MODEL_DEVICE` | `cpu` | `cuda:0` | 計算設備 |
| `MODEL_TORCH_DTYPE` | `float32` | `float16` | 模型精度 |
| `CUDA_VISIBLE_DEVICES` | `""` | `0` | GPU 設備 |

---

## 🎯 預設行為

- **新克隆的專案**：預設使用 Mock 模式（根據 `.env.example`）
- **無 `.env` 文件**：後端會嘗試自動判斷（無 GPU 時自動使用 Mock）
- **CI/CD 環境**：建議明確設定為 Mock 模式

---

## 💡 提示

1. **開發時建議使用 Mock 模式**，可以大幅提升開發效率
2. **提交 PR 前建議使用真實模式**測試一次，確保 AI 功能正常
3. **生產環境**必須使用真實模式
4. 可以混合使用，例如：`T2I_MOCK=0, VLM_MOCK=1, LLM_MOCK=1`
