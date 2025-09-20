#!/bin/bash

# 1. Install additional dependencies for batch processing
pip install celery[redis] redis psutil plotly

# 2. Start Redis server (if not using Docker)
# Option A: Using apt (Ubuntu/Debian)
sudo apt-get install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Option B: Using Docker
docker run -d --name redis-server -p 6379:6379 redis:7-alpine

# Option C: Using Homebrew (macOS)
brew install redis
brew services start redis

# 3. Verify Redis connection
redis-cli ping
# Should return: PONG

# 4. Create required directories
mkdir -p logs
mkdir -p /mnt/ai_warehouse/cache/outputs/multi-modal-lab/batch_results

# tests/smoke_test_phase3.sh

set -e

echo "🧪 Phase 3 Smoke Tests - Batch Processing & Monitoring"
echo "======================================================="

API_URL="http://localhost:8000/api/v1"
REDIS_URL="redis://localhost:6379/0"

# Test 1: Redis connectivity
echo "1. Testing Redis connectivity..."
if redis-cli ping > /dev/null 2>&1; then
    echo "   ✅ Redis is running"
else
    echo "   ❌ Redis is not running"
    exit 1
fi

# Test 2: API health check
echo "2. Testing API health..."
response=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/health")
if [ "$response" = "200" ]; then
    echo "   ✅ API is healthy"
else
    echo "   ❌ API health check failed (HTTP $response)"
    exit 1
fi

# Test 3: Monitoring endpoints
echo "3. Testing monitoring endpoints..."

endpoints=(
    "/monitoring/health"
    "/monitoring/metrics"
    "/monitoring/tasks"
    "/monitoring/workers"
)

for endpoint in "${endpoints[@]}"; do
    response=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL$endpoint")
    if [ "$response" = "200" ]; then
        echo "   ✅ $endpoint: OK"
    else
        echo "   ❌ $endpoint: Failed (HTTP $response)"
    fi
done

# Test 4: Batch job submission (mock)
echo "4. Testing batch job submission..."
batch_payload='{
    "job_type": "caption",
    "inputs": ["/tmp/test1.jpg", "/tmp/test2.jpg"],
    "config": {"max_length": 50}
}'

response=$(curl -s -X POST "$API_URL/batch/submit" \
    -H "Content-Type: application/json" \
    -d "$batch_payload" \
    -o /dev/null -w "%{http_code}")

if [ "$response" = "200" ] || [ "$response" = "202" ]; then
    echo "   ✅ Batch job submission endpoint works"
else
    echo "   ❌ Batch job submission failed (HTTP $response)"
fi

# Test 5: Batch job listing
echo "5. Testing batch job listing..."
response=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/batch/list")
if [ "$response" = "200" ]; then
    echo "   ✅ Batch job listing works"
else
    echo "   ❌ Batch job listing failed (HTTP $response)"
fi

# Test 6: Check Celery worker status (if running)
echo "6. Testing Celery worker connectivity..."
if pgrep -f "celery.*worker" > /dev/null; then
    echo "   ✅ Celery worker process is running"

    # Test worker status endpoint
    response=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/monitoring/workers")
    if [ "$response" = "200" ]; then
        echo "   ✅ Worker status endpoint works"
    else
        echo "   ❌ Worker status endpoint failed (HTTP $response)"
    fi
else
    echo "   ⚠️  Celery worker not running (expected for basic tests)"
fi

# Test 7: Database initialization
echo "7. Testing database initialization..."
if python -c "
import sys, os
sys.path.insert(0, '.')
from backend.core.batch.manager import BatchManager
from backend.core.monitoring.metrics import MetricsCollector

try:
    bm = BatchManager()
    mc = MetricsCollector()
    print('✅ Database initialization successful')
except Exception as e:
    print(f'❌ Database initialization failed: {e}')
    sys.exit(1)
"; then
    echo "   ✅ Database tables created successfully"
else
    echo "   ❌ Database initialization failed"
    exit 1
fi

echo ""
echo "🎉 Phase 3 smoke tests completed successfully!"
echo ""
echo "Next steps:"
echo "1. Start Celery worker: python scripts/start_worker.py"
echo "2. Start monitoring dashboard: python frontend/gradio_app/monitoring_dashboard.py"
echo "3. Test batch processing: python scripts/batch_test.py --monitor"
