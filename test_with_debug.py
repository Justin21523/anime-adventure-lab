#!/usr/bin/env python3
"""帶有詳細 DEBUG 日誌的測試"""
import logging
import requests
import json

# 設置 DEBUG 級別日誌
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

API_BASE = "http://localhost:8000/api/v1"

print("=" * 60)
print("測試 Turn Endpoint (DEBUG 模式)")
print("=" * 60)

# 創建 session
print("\n步驟 1: 創建 session")
response = requests.post(
    f"{API_BASE}/story/session",
    json={
        "player_name": "Debug測試",
        "persona_id": "wise_sage",
        "setting": "測試環境",
        "difficulty": "medium",
        "use_agent": False,
        "enrich_with_rag": False
    },
    timeout=30
)

if response.status_code != 200:
    print(f"✗ Session 創建失敗: {response.text}")
    exit(1)

session_id = response.json()["session_id"]
print(f"✓ Session 創建成功: {session_id}")

# 發送 turn
print("\n步驟 2: 發送 turn")
print(f"Request: POST {API_BASE}/story/turn")
print(f"Body: {json.dumps({'session_id': session_id, 'player_input': 'test', 'use_agent': False, 'enrich_with_rag': False}, indent=2)}")

response = requests.post(
    f"{API_BASE}/story/turn",
    json={
        "session_id": session_id,
        "player_input": "test",
        "use_agent": False,
        "enrich_with_rag": False
    },
    timeout=30
)

print(f"\nResponse Status: {response.status_code}")
print(f"Response Headers: {dict(response.headers)}")

if response.status_code == 200:
    print("✓ Turn 成功!")
    data = response.json()
    print(f"Narrative: {data['narrative'][:100]}...")
else:
    print(f"✗ Turn 失敗!")
    print(f"Response Body: {response.text}")

    # 嘗試從後端日誌獲取更多信息
    print("\n嘗試讀取後端錯誤...")

print("\n" + "=" * 60)
