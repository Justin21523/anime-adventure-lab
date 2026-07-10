from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse


def problem_response(
    request: Request,
    *,
    status_code: int,
    title: str,
    detail: str,
    code: str,
    errors: Any | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    request_id = str(getattr(request.state, "request_id", "") or "")
    content: dict[str, Any] = {
        "type": f"https://docs.sagaforge.local/problems/{code.lower().replace('_', '-')}",
        "title": title,
        "status": status_code,
        "detail": detail,
        "instance": request.url.path,
        "code": code,
    }
    if request_id:
        content["request_id"] = request_id
    if errors is not None:
        content["errors"] = errors
    response_headers = dict(headers or {})
    if request_id:
        response_headers["X-Request-ID"] = request_id
    return JSONResponse(
        status_code=status_code,
        content=content,
        headers=response_headers,
        media_type="application/problem+json",
    )
