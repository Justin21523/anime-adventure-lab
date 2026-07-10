#!/usr/bin/env python3
"""
Generate OpenAPI JSON schema for frontend TypeScript type generation
"""
import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

def main():
    """Generate OpenAPI schema from FastAPI app"""
    try:
        from api.main import app

        # Generate OpenAPI schema
        openapi_schema = app.openapi()

        # Output path
        output_path = project_root / "frontend" / "react" / "openapi.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write schema
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(openapi_schema, f, indent=2, ensure_ascii=False)

        print(f"✅ OpenAPI schema generated: {output_path}")
        print(f"   - Title: {openapi_schema.get('info', {}).get('title', 'N/A')}")
        print(f"   - Version: {openapi_schema.get('info', {}).get('version', 'N/A')}")
        print(f"   - Paths: {len(openapi_schema.get('paths', {}))}")

        return 0

    except Exception as e:
        print(f"❌ Error generating OpenAPI schema: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
