#!/bin/bash

# 前端改進驗證腳本
# 此腳本檢查所有 3 週改進是否正確實施

echo "======================================"
echo "前端 UI 改進驗證腳本"
echo "======================================"
echo ""

# 顏色定義
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 計數器
passed=0
failed=0

# 檢查函數
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} 文件存在: $1"
        ((passed++))
        return 0
    else
        echo -e "${RED}✗${NC} 文件缺失: $1"
        ((failed++))
        return 1
    fi
}

check_directory() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} 目錄存在: $1"
        ((passed++))
        return 0
    else
        echo -e "${RED}✗${NC} 目錄缺失: $1"
        ((failed++))
        return 1
    fi
}

check_content() {
    if grep -q "$2" "$1" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} 內容驗證: $3"
        ((passed++))
        return 0
    else
        echo -e "${RED}✗${NC} 內容缺失: $3"
        ((failed++))
        return 1
    fi
}

echo "=== Week 1: 基礎設施改進 ==="
echo ""

echo "檢查 CORS 修復..."
check_content "frontend/react/.env" "VITE_API_BASE=/api/v1" ".env 使用相對路徑"
check_content "frontend/react/src/api/client.ts" "baseURL: import.meta.env.VITE_API_BASE" "API client 配置正確"
echo ""

echo "檢查日誌系統..."
check_file "frontend/react/src/utils/logger.ts"
check_content "frontend/react/src/utils/logger.ts" "export const logger" "logger 導出正確"
echo ""

echo "=== Week 2: Agent 工具系統 ==="
echo ""

echo "檢查 Agent 組件..."
check_file "frontend/react/src/features/agent/components/AgentToolSelector.tsx"
check_file "frontend/react/src/features/agent/components/ToolParameterForm.tsx"
check_file "frontend/react/src/features/agent/components/ToolExecutionPanel.tsx"
check_file "frontend/react/src/features/agent/components/AgentSystem.tsx"
echo ""

echo "檢查 Agent hooks..."
check_file "frontend/react/src/features/agent/hooks/useToolExecutionMonitor.ts"
echo ""

echo "檢查 Story 組件..."
check_file "frontend/react/src/features/story/components/AgentActionsPanel.tsx"
echo ""

echo "=== Week 3: 性能優化 ==="
echo ""

echo "檢查 React Query 配置..."
check_file "frontend/react/src/config/query.config.ts"
check_content "frontend/react/src/config/query.config.ts" "staleTime: 2 \* 60_000" "staleTime 優化為 2 分鐘"
check_content "frontend/react/src/config/query.config.ts" "gcTime: 10 \* 60_000" "gcTime 優化為 10 分鐘"
echo ""

echo "檢查懶加載實現..."
check_content "frontend/react/src/App.tsx" "lazy" "App.tsx 使用 lazy loading"
check_content "frontend/react/src/App.tsx" "Suspense" "App.tsx 使用 Suspense"
check_content "frontend/react/src/features/story/components/StoryGameScreen.tsx" "lazy" "StoryGameScreen 使用 lazy loading"
echo ""

echo "檢查 Vite 配置優化..."
check_file "frontend/react/vite.config.ts"
check_content "frontend/react/vite.config.ts" "optimizeDeps" "Vite 依賴優化配置"
check_content "frontend/react/vite.config.ts" "warmup" "Vite warmup 配置"
check_content "frontend/react/vite.config.ts" "manualChunks" "手動 chunk 配置"
echo ""

echo "檢查性能監控工具..."
check_file "frontend/react/src/hooks/usePerformance.ts"
check_content "frontend/react/src/hooks/usePerformance.ts" "usePerformance" "usePerformance hook"
check_content "frontend/react/src/hooks/usePerformance.ts" "useDebounce" "useDebounce hook"
check_content "frontend/react/src/hooks/usePerformance.ts" "useThrottle" "useThrottle hook"
echo ""

echo "檢查其他高級功能..."
check_file "frontend/react/src/hooks/useSmartRetry.ts"
check_file "frontend/react/src/hooks/useResumableUpload.ts"
check_file "frontend/react/src/hooks/useSSE.ts"
echo ""

echo "=== 文檔檢查 ==="
echo ""

echo "檢查文檔文件..."
check_file "frontend/react/PERFORMANCE.md"
check_file "frontend/react/FRONTEND_IMPROVEMENTS.md"
echo ""

echo "======================================"
echo "驗證結果總結"
echo "======================================"
echo -e "通過: ${GREEN}$passed${NC}"
echo -e "失敗: ${RED}$failed${NC}"
echo ""

if [ $failed -eq 0 ]; then
    echo -e "${GREEN}✓ 所有檢查通過!${NC}"
    echo "前端改進已成功實施。"
    exit 0
else
    echo -e "${YELLOW}⚠ 部分檢查失敗${NC}"
    echo "請檢查上述失敗項目。"
    exit 1
fi
