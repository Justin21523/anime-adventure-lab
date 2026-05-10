#!/usr/bin/env python3
"""Debug script to test story turn endpoint"""
import requests
import json
import traceback

API_BASE = "http://localhost:8000/api/v1"

def test_turn():
    try:
        # 1. Create session
        print("=" * 60)
        print("STEP 1: Creating session...")
        print("=" * 60)

        session_payload = {
            "player_name": "測試玩家",
            "persona_id": "wise_sage",
            "setting": "測試環境",
            "difficulty": "medium",
            "use_agent": False,  # Start with simpler test
            "enrich_with_rag": False
        }

        print(f"\nRequest payload:")
        print(json.dumps(session_payload, indent=2, ensure_ascii=False))

        response = requests.post(
            f"{API_BASE}/story/session",
            json=session_payload,
            timeout=30
        )

        print(f"\nResponse status: {response.status_code}")
        print(f"Response body:")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))

        response.raise_for_status()
        session_id = response.json()["session_id"]
        print(f"\n✓ Session created: {session_id}")

        # 2. Send turn
        print("\n" + "=" * 60)
        print("STEP 2: Sending turn...")
        print("=" * 60)

        turn_payload = {
            "session_id": session_id,
            "player_input": "環顧四周",
            "use_agent": False,
            "enrich_with_rag": False
        }

        print(f"\nRequest payload:")
        print(json.dumps(turn_payload, indent=2, ensure_ascii=False))

        response = requests.post(
            f"{API_BASE}/story/turn",
            json=turn_payload,
            timeout=30
        )

        print(f"\nResponse status: {response.status_code}")
        print(f"Response body:")

        if response.status_code == 200:
            data = response.json()
            print(json.dumps(data, indent=2, ensure_ascii=False))
            print("\n✓ Turn successful!")
        else:
            print(response.text)
            print("\n✗ Turn failed!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_turn()
