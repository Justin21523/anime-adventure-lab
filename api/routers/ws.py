"""WebSocket endpoint for realtime training progress.

Adapted from charaforge-T2I-Lab/api/routers/ws.py for anime-adventure-lab.

Channel: ``/api/v1/ws/train/{job_id}``
Protocol: workers publish to Redis pubsub channel ``anime_adventure:train:{job_id}``
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.jwt_tokens import verify_access_token
from api.security import parse_api_keys, resolve_api_key, scope_allows
from api.train_access import read_train_access_owner
from api.ws_tickets import verify_ws_ticket
from core.config import get_config

logger = logging.getLogger(__name__)

router = APIRouter(tags=["WebSocket"])


def _redis_url() -> str:
    return (
        os.getenv("REDIS_URL")
        or os.getenv("CELERY_BROKER_URL")
        or "redis://localhost:6379/0"
    )


def _parse_ws_protocols(value: str | None) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


@router.websocket("/ws/train/{job_id}")
async def ws_train_progress(websocket: WebSocket, job_id: str) -> None:
    config = get_config()

    # Resolve API keys from config
    admin_keys = set()
    user_keys = set()
    try:
        api_cfg = getattr(config, "api", None)
        if api_cfg:
            admin_raw = getattr(api_cfg, "admin_keys", getattr(api_cfg, "api_admin_keys", None)) or []
            user_raw = getattr(api_cfg, "keys", getattr(api_cfg, "api_keys", None)) or []
            admin_keys = set(str(k) for k in admin_raw if k)
            user_keys = set(str(k) for k in user_raw if k)
            api_key_single = getattr(api_cfg, "api_key", None)
            if api_key_single:
                admin_keys.add(str(api_key_single))
    except Exception:
        pass

    protocols = _parse_ws_protocols(websocket.headers.get("sec-websocket-protocol"))
    selected_subprotocol = "anime_adventure" if "anime_adventure" in protocols else None

    auth_enabled = bool(admin_keys or user_keys)
    ws_ticket_payload = None

    if auth_enabled:
        required_scope = "train:manage"

        subject = ""
        proto_access_token = ""
        proto_api_key = ""
        for proto in protocols:
            if proto.startswith("access_token."):
                proto_access_token = proto.split(".", 1)[1]
            elif proto.startswith("api_key."):
                proto_api_key = proto.split(".", 1)[1]
            elif proto.startswith("ws_ticket."):
                ws_ticket_payload = verify_ws_ticket(proto.split(".", 1)[1])

        access_token = proto_access_token or websocket.query_params.get("access_token")
        role = ""
        scopes: set[str] = set()

        if access_token:
            payload = verify_access_token(access_token)
            if not payload:
                await websocket.accept(subprotocol=selected_subprotocol)
                await websocket.send_json({"topic": "ws.error", "message": "Unauthorized"})
                await websocket.close(code=4401)
                return
            role = str(payload.get("role") or "user")
            subject = str(payload.get("sub") or "")
            scopes_raw = payload.get("scopes") or []
            if isinstance(scopes_raw, str):
                scopes_raw = [scopes_raw]
            scopes = {str(s).strip() for s in scopes_raw if str(s).strip()}
            if scopes and not scope_allows(scopes, required_scope):
                await websocket.accept(subprotocol=selected_subprotocol)
                await websocket.send_json({"topic": "ws.error", "message": "Forbidden"})
                await websocket.close(code=4403)
                return
        elif ws_ticket_payload:
            if str(ws_ticket_payload.get("job_id") or "") != str(job_id):
                await websocket.accept(subprotocol=selected_subprotocol)
                await websocket.send_json({"topic": "ws.error", "message": "Forbidden"})
                await websocket.close(code=4403)
                return
            role = str(ws_ticket_payload.get("role") or "user")
            subject = str(ws_ticket_payload.get("sub") or "")
            scopes_raw = ws_ticket_payload.get("scopes") or []
            if isinstance(scopes_raw, str):
                scopes_raw = [scopes_raw]
            scopes = {str(s).strip() for s in scopes_raw if str(s).strip()}
        else:
            # API key via header
            presented = (
                websocket.headers.get("X-API-Key")
                or proto_api_key
                or websocket.query_params.get("api_key")
                or websocket.query_params.get("token")
            )
            auth = resolve_api_key(presented, admin_keys=admin_keys, user_keys=user_keys)
            if not auth:
                await websocket.accept(subprotocol=selected_subprotocol)
                await websocket.send_json({"topic": "ws.error", "message": "Unauthorized"})
                await websocket.close(code=4401)
                return
            presented_key = str(presented or "")
            digest = hashlib.sha256(presented_key.encode("utf-8")).hexdigest()[:32]
            subject = f"key:{digest}"
            role = auth.role
            scopes = set(auth.scopes or set())
            if scopes and not scope_allows(scopes, required_scope):
                await websocket.accept(subprotocol=selected_subprotocol)
                await websocket.send_json({"topic": "ws.error", "message": "Forbidden"})
                await websocket.close(code=4403)
                return

        # Owner check for non-admin
        if role != "admin":
            owner = read_train_access_owner(job_id)
            if owner and owner != subject:
                await websocket.accept(subprotocol=selected_subprotocol)
                await websocket.send_json({"topic": "ws.error", "message": "Forbidden"})
                await websocket.close(code=4403)
                return

    await websocket.accept(subprotocol=selected_subprotocol)

    try:
        import redis.asyncio as redis
    except Exception:
        await websocket.send_json({
            "topic": "ws.error",
            "message": "redis-py is not installed; realtime progress unavailable",
        })
        await websocket.close(code=1011)
        return

    url = _redis_url()
    channel = f"anime_adventure:train:{job_id}"
    client = redis.from_url(url, decode_responses=True)
    pubsub = client.pubsub()

    # Ticket replay protection
    if ws_ticket_payload:
        jti = str(ws_ticket_payload.get("jti") or "")
        exp = int(ws_ticket_payload.get("exp") or 0)
        now = int(time.time())
        ttl = max(1, exp - now + 5)
        ticket_key = f"anime_adventure:ws_ticket:{jti}"
        try:
            ok = await client.set(ticket_key, "1", nx=True, ex=ttl)
        except Exception:
            await websocket.send_json({"topic": "ws.error", "message": "Service unavailable"})
            await websocket.close(code=1011)
            return
        if not ok:
            await websocket.send_json({"topic": "ws.error", "message": "Unauthorized"})
            await websocket.close(code=4401)
            return

    disconnect_code: str | None = None
    try:
        await pubsub.subscribe(channel)
        await websocket.send_json({"topic": "ws.subscribed", "channel": channel})

        while True:
            try:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if message and message.get("type") == "message":
                    raw = message.get("data")
                    try:
                        payload = json.loads(raw) if isinstance(raw, str) else {"data": raw}
                    except Exception:
                        payload = {"topic": "ws.message", "data": raw}
                    await websocket.send_json(payload)
                else:
                    await asyncio.sleep(0.1)
            except WebSocketDisconnect as exc:
                disconnect_code = str(getattr(exc, "code", "") or "1000")
                break
    finally:
        try:
            await pubsub.unsubscribe(channel)
        except Exception:
            pass
        try:
            await pubsub.close()
        except Exception:
            pass
        try:
            await client.close()
        except Exception:
            pass
