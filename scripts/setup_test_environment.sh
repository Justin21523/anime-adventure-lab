#!/bin/bash
# scripts/setup_test_environment.sh
# 測試環境自動設置腳本

set -e

echo "🔧 設置 Multi-Modal Lab 測試環境"
echo "=================================="

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 檢查 Python 版本
echo -e "${BLUE}檢查 Python 版本...${NC}"
PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then
    echo -e "${GREEN}✅ Python 版本符合要求: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}❌ Python 版本過舊: $PYTHON_VERSION < $REQUIRED_VERSION${NC}"
    exit 1
fi

# 創建虛擬環境
if [ ! -d "venv" ]; then
    echo -e "${BLUE}創建虛擬環境...${NC}"
    python -m venv venv
    echo -e "${GREEN}✅ 虛擬環境已創建${NC}"
else
    echo -e "${YELLOW}⚠️ 虛擬環境已存在${NC}"
fi

# 激活虛擬環境
echo -e "${BLUE}激活虛擬環境...${NC}"
source venv/bin/activate || source venv/Scripts/activate

# 升級 pip
echo -e "${BLUE}升級 pip...${NC}"
pip install --upgrade pip

# 安裝依賴
echo -e "${BLUE}安裝測試依賴...${NC}"
pip install -r requirements.txt
pip install -r requirements-test.txt

# 設置測試快取目錄
TEST_CACHE_DIR="${AI_CACHE_ROOT:-/tmp/test_cache}"
echo -e "${BLUE}設置測試快取目錄: $TEST_CACHE_DIR${NC}"

mkdir -p "$TEST_CACHE_DIR"/{hf,torch,models,datasets,outputs}
mkdir -p "$TEST_CACHE_DIR"/models/{lora,blip2,qwen,llava,embeddings}
mkdir -p "$TEST_CACHE_DIR"/datasets/{raw,processed,metadata}
mkdir -p "$TEST_CACHE_DIR"/outputs/multi-modal-lab

# 設置環境變數
echo -e "${BLUE}設置環境變數...${NC}"
export AI_CACHE_ROOT="$TEST_CACHE_DIR"
export API_PREFIX="/api/v1"
export ALLOWED_ORIGINS="http://localhost:3000,http://localhost:7860"
export DEVICE="cpu"
export DEBUG="true"

# 創建 .env.test 文件
cat > .env.test << EOF
AI_CACHE_ROOT=$TEST_CACHE_DIR
API_PREFIX=/api/v1
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:7860
DEVICE=cpu
MAX_WORKERS=2
MAX_BATCH_SIZE=4
DEBUG=true
EOF

echo -e "${GREEN}✅ 環境變數配置完成${NC}"

# 創建測試結果目錄
mkdir -p test_results/{logs,reports,coverage}
echo -e "${GREEN}✅ 測試結果目錄已創建${NC}"

# 運行健康檢查
echo -e "${BLUE}運行系統健康檢查...${NC}"
python scripts/health_check.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}🎉 測試環境設置完成！${NC}"
    echo -e "${BLUE}現在可以運行測試：${NC}"
    echo -e "  make test-smoke    # 快速煙霧測試"
    echo -e "  make test          # 完整測試套件"
    echo -e "  python tests/run_tests.py --smoke  # 使用Python腳本"
else
    echo -e "${RED}❌ 健康檢查失敗，請檢查配置${NC}"
    exit 1
fi
