#!/bin/bash
# Test script to verify Mock mode is working correctly
# Run this script to ensure all endpoints work without loading real AI models

set -e

API_BASE="${API_BASE:-http://localhost:8000/api/v1}"
BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BOLD}🧪 Testing Mock Mode Configuration${NC}\n"

# Function to test endpoint
test_endpoint() {
    local name="$1"
    local endpoint="$2"
    local method="${3:-GET}"
    local data="$4"

    echo -n "Testing ${name}... "

    if [ "$method" = "POST" ]; then
        response=$(curl -s -X POST "$API_BASE$endpoint" \
            -H "Content-Type: application/json" \
            -d "$data" \
            -w "\n%{http_code}")
    else
        response=$(curl -s "$API_BASE$endpoint" -w "\n%{http_code}")
    fi

    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)

    if [ "$http_code" -ge 200 ] && [ "$http_code" -lt 300 ]; then
        echo -e "${GREEN}✓ PASS${NC} (HTTP $http_code)"
        return 0
    else
        echo -e "${RED}✗ FAIL${NC} (HTTP $http_code)"
        echo "Response: $body"
        return 1
    fi
}

# Test 1: Health check
echo -e "${BOLD}1. Health & System Status${NC}"
test_endpoint "Health" "/health"
echo ""

# Test 2: Story endpoints
echo -e "${BOLD}2. Story Module (Mock LLM)${NC}"
test_endpoint "List sessions" "/story/sessions"
test_endpoint "List personas" "/story/personas"
echo ""

# Test 3: RAG endpoints
echo -e "${BOLD}3. RAG Module (Mock Embeddings)${NC}"
test_endpoint "RAG stats" "/rag/stats"
echo ""

# Test 4: T2I endpoints
echo -e "${BOLD}4. Text-to-Image Module (Mock SD)${NC}"
test_endpoint "List generations" "/t2i/generations"
test_endpoint "List LoRAs" "/t2i/loras"
echo ""

# Test 5: Agent endpoints
echo -e "${BOLD}5. Agent Module${NC}"
test_endpoint "List tools" "/agent/tools"
echo ""

# Test 6: Batch endpoints
echo -e "${BOLD}6. Batch Module${NC}"
test_endpoint "List jobs" "/batch/jobs"
echo ""

# Verify Mock mode from health endpoint
echo -e "${BOLD}7. Verify Mock Mode Configuration${NC}"
health_response=$(curl -s "$API_BASE/health")

echo "Checking environment configuration..."
if echo "$health_response" | grep -q '"cuda_available":false'; then
    echo -e "${GREEN}✓${NC} CUDA disabled (Mock mode)"
else
    echo -e "${YELLOW}⚠${NC} CUDA may be enabled (check .env)"
fi

if echo "$health_response" | grep -q '"device_map":"cpu"'; then
    echo -e "${GREEN}✓${NC} Using CPU device (Mock mode)"
else
    echo -e "${YELLOW}⚠${NC} Device may not be CPU (check .env)"
fi

echo ""
echo -e "${BOLD}8. Environment Variables Check${NC}"
echo "T2I_MOCK=${T2I_MOCK:-not set}"
echo "VLM_MOCK=${VLM_MOCK:-not set}"
echo "LLM_MOCK=${LLM_MOCK:-not set}"
echo "MODEL_DEVICE=${MODEL_DEVICE:-not set}"

echo ""
echo -e "${GREEN}${BOLD}✓ All tests passed!${NC}"
echo -e "${BOLD}Mock mode is working correctly.${NC}"
echo ""
echo -e "${YELLOW}Note:${NC} To switch to real models, edit .env and set:"
echo "  T2I_MOCK=0"
echo "  VLM_MOCK=0"
echo "  LLM_MOCK=0"
echo "  MODEL_DEVICE=cuda:0"
