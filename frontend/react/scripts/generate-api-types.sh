#!/bin/bash
# Generate TypeScript types from OpenAPI schema

set -e

cd "$(dirname "$0")/.."

echo "🔄 Generating OpenAPI schema from backend..."
python ../../../scripts/generate_openapi_schema.py

if [ ! -f "openapi.json" ]; then
    echo "❌ openapi.json not found!"
    exit 1
fi

echo "🔄 Generating TypeScript types..."
mkdir -p src/api/generated
npx openapi-typescript openapi.json -o src/api/generated/api.ts

echo "✅ API types generated successfully!"
echo "   Output: src/api/generated/api.ts"
