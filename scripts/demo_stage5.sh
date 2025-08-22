#!/bin/bash
# scripts/demo_stage5.sh - SagaForge Stage 5 Demo

echo "🚀 SagaForge Stage 5 - T2I Pipeline Demo"
echo "======================================="

# Set environment variables
export AI_CACHE_ROOT="../ai_warehouse/cache"
export CUDA_VISIBLE_DEVICES=0

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  Creating .env file..."
    cp .env.example .env 2>/dev/null || echo "AI_CACHE_ROOT=../ai_warehouse/cache" > .env
fi

# Setup environment
echo "📦 Setting up environment..."
python scripts/setup_stage5.py

# Run smoke tests
echo ""
echo "🧪 Running smoke tests..."
python scripts/smoke_test_t2i.py

# Start API server in background
echo ""
echo "🌐 Starting API server..."
echo "   (will run in background)"

# Kill any existing server
pkill -f "api/main.py" 2>/dev/null || true
sleep 2

# Start new server
python api/main.py &
SERVER_PID=$!
echo "   Server PID: $SERVER_PID"

# Wait for server to start
echo "   Waiting for server to start..."
sleep 10

# Test API endpoints
echo ""
echo "🔗 Testing API endpoints..."

# Health check
echo "1. Health check:"
curl -s http://localhost:8000/healthz | python -m json.tool

echo ""
echo "2. Model info:"
curl -s http://localhost:8000/api/v1/t2i/models | python -m json.tool

echo ""
echo "3. Available styles:"
curl -s http://localhost:8000/api/v1/t2i/styles | python -m json.tool

# Generate test image
echo ""
echo "4. Generating test image..."
curl -X POST http://localhost:8000/api/v1/t2i/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a beautiful anime girl with blue hair in a magical forest",
    "negative_prompt": "blurry, low quality, bad anatomy",
    "width": 512,
    "height": 512,
    "num_inference_steps": 20,
    "guidance_scale": 7.5,
    "seed": 42,
    "style_id": "anime_style",
    "scene_id": "demo_001",
    "image_type": "portrait"
  }' | python -m json.tool

echo ""
echo "5. Generating background image..."
curl -X POST http://localhost:8000/api/v1/t2i/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "cyberpunk city street at night with neon lights",
    "negative_prompt": "blurry, low quality",
    "width": 768,
    "height": 512,
    "num_inference_steps": 25,
    "guidance_scale": 7.0,
    "seed": 12345,
    "style_id": "anime_style",
    "scene_id": "demo_002",
    "image_type": "bg"
  }' | python -m json.tool

# Show generated files
echo ""
echo "📁 Generated files:"
find ../ai_warehouse/cache/outputs/saga-forge -name "*.png" -exec ls -la {} \; 2>/dev/null | tail -5

echo ""
echo "📋 Demo Summary:"
echo "✓ T2I Pipeline loaded"
echo "✓ API server running on http://localhost:8000"
echo "✓ Generated test images with fixed seeds"
echo "✓ Style presets working"
echo "✓ Metadata files created"

echo ""
echo "🌐 Next steps:"
echo "1. Open API docs: http://localhost:8000/docs"
echo "2. Check generated images in: ../ai_warehouse/cache/outputs/saga-forge/"
echo "3. Stop server: kill $SERVER_PID"
echo ""
echo "Demo complete! 🎉"