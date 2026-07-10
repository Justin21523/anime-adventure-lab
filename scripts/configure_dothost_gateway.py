#!/usr/bin/env python3
"""Idempotently add SagaForge routes to the existing portfolio Nginx gateway."""
from __future__ import annotations

import argparse
from pathlib import Path

UPSTREAM_MARKER = "# ---- SagaForge upstreams ----"
ROUTE_MARKER = "# ---- SagaForge / anime-adventure-lab ----"

UPSTREAMS = """    # ---- SagaForge upstreams ----
    set $upstream_29 http://sagaforge-deploy-portfolio-1:80;
    set $upstream_30 http://sagaforge-deploy-frontend-1:80;
    set $upstream_31 http://sagaforge-deploy-api-1:8000;

"""

ROUTES = """    # ---- SagaForge / anime-adventure-lab ----
    location = /p/anime-adventure-lab {
        return 301 /p/anime-adventure-lab/;
    }
    location ^~ /p/anime-adventure-lab/api/v2/ {
        rewrite ^/p/anime-adventure-lab/api/v2/(.*)$ /api/v2/$1 break;
        set $upstream_31 http://sagaforge-deploy-api-1:8000;
        proxy_pass $upstream_31;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Request-ID $request_id;
        proxy_read_timeout 300s;
    }
    location ^~ /p/anime-adventure-lab/app/ {
        rewrite ^/p/anime-adventure-lab/app/(.*)$ /$1 break;
        set $upstream_30 http://sagaforge-deploy-frontend-1:80;
        proxy_pass $upstream_30;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
    }
    location ^~ /p/anime-adventure-lab/ {
        rewrite ^/p/anime-adventure-lab/(.*)$ /$1 break;
        set $upstream_29 http://sagaforge-deploy-portfolio-1:80;
        proxy_pass $upstream_29;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }

"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    text = args.config.read_text(encoding="utf-8")
    if UPSTREAM_MARKER not in text:
        anchor = "    # Resolve Docker service names"
        if anchor not in text:
            anchor = "    # ---- agentic-bi-dataops-copilot ----"
        if anchor not in text:
            raise SystemExit("Could not find the upstream insertion point")
        text = text.replace(anchor, UPSTREAMS + anchor, 1)
    if ROUTE_MARKER not in text:
        anchor = "    location / {"
        if anchor not in text:
            raise SystemExit("Could not find the route insertion point")
        text = text.replace(anchor, ROUTES + anchor, 1)
    args.config.write_text(text, encoding="utf-8")
    print(f"SagaForge gateway routes configured in {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
