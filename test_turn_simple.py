#!/usr/bin/env python3
"""簡化的 turn 測試腳本，加入更多調試信息"""
import requests
import json
import sys

API_BASE = "http://localhost:8000/api/v1"

print("=" * 60)
print("測試 1: 創建 session")
print("=" * 60)

# 創建 session
response = requests.post(
    f"{API_BASE}/story/session",
    json={
        "player_name": "測試",
        "persona_id": "wise_sage",
        "setting": "測試環境",
        "difficulty": "medium",
        "use_agent": False,
        "enrich_with_rag": False
    },
    timeout=30
)

print(f"Status: {response.status_code}")
if response.status_code != 200:
    print(f"錯誤: {response.text}")
    sys.exit(1)

data = response.json()
session_id = data["session_id"]
print(f"✓ Session 創建成功: {session_id}")
print(f"✓ 初始 choices 數量: {len(data['choices'])}")

print("\n" + "=" * 60)
print("測試 2: 發送 turn（簡單輸入，無 agent, 無 RAG）")
print("=" * 60)

# 測試最簡單的 turn
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

print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"✓ Turn 成功!")
    print(f"✓ Narrative: {data['narrative'][:100]}...")
    print(f"✓ Choices: {len(data['choices'])}")
else:
    print(f"✗ Turn 失敗!")
    print(f"Response: {response.text}")

    # 嘗試獲取更多信息
    print("\n嘗試獲取 session 狀態...")
    session_response = requests.get(f"{API_BASE}/story/session/{session_id}")
    if session_response.status_code == 200:
        session_data = session_response.json()
        print(f"Session 狀態: {json.dumps(session_data, indent=2, ensure_ascii=False)[:500]}...")

    sys.exit(1)

print("\n" + "=" * 60)
print("✓ 所有測試通過!")
print("=" * 60)
