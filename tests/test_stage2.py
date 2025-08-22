# scripts/test_stage2.py
import requests
import json
import time
from typing import Dict, Any


def test_api_health():
    """Test API health endpoints"""
    print("🔍 Testing API health...")

    try:
        # Test root health
        response = requests.get("http://localhost:8000/healthz", timeout=10)
        assert response.status_code == 200
        print("✅ Root health check passed")

        # Test LLM health
        response = requests.get("http://localhost:8000/api/v1/health", timeout=10)
        assert response.status_code == 200
        data = response.json()
        print(
            f"✅ LLM health check passed - Model: {data.get('model')}, Available: {data.get('llm_available')}"
        )

        return True

    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_sample_data():
    """Test sample persona and game state endpoints"""
    print("\n🔍 Testing sample data endpoints...")

    try:
        # Test sample persona
        response = requests.get(
            "http://localhost:8000/api/v1/persona/sample", timeout=10
        )
        assert response.status_code == 200
        persona = response.json()
        assert "name" in persona
        assert "personality" in persona
        print(f"✅ Sample persona loaded: {persona['name']}")

        # Test sample game state
        response = requests.get(
            "http://localhost:8000/api/v1/gamestate/sample", timeout=10
        )
        assert response.status_code == 200
        state = response.json()
        assert "scene_id" in state
        assert "turn_count" in state
        print(f"✅ Sample game state loaded: Scene '{state['scene_id']}'")

        return persona, state

    except Exception as e:
        print(f"❌ Sample data test failed: {e}")
        return None, None


def test_turn_processing(persona: Dict[str, Any], game_state: Dict[str, Any]):
    """Test story turn processing"""
    print("\n🔍 Testing turn processing...")

    try:
        # Prepare turn request
        request_data = {
            "player_input": "我想探索這個神秘的圖書館，尋找有關古老魔法的書籍。",
            "persona": persona,
            "game_state": game_state,
            "choice_id": None,
        }

        print("📤 Sending turn request...")
        start_time = time.time()

        response = requests.post(
            "http://localhost:8000/api/v1/turn", json=request_data, timeout=30
        )

        elapsed = time.time() - start_time

        assert response.status_code == 200
        result = response.json()

        # Validate response structure
        assert "narration" in result
        assert "dialogues" in result
        assert "choices" in result
        assert isinstance(result["dialogues"], list)
        assert isinstance(result["choices"], list)

        print(f"✅ Turn processing successful (took {elapsed:.1f}s)")
        print(f"📖 Narration: {result['narration'][:100]}...")

        if result["dialogues"]:
            print(f"💬 Dialogues: {len(result['dialogues'])} entries")
            for d in result["dialogues"][:2]:  # Show first 2
                print(f"   - {d['speaker']}: {d['text'][:50]}...")

        if result["choices"]:
            print(f"🎯 Choices: {len(result['choices'])} options")
            for i, c in enumerate(result["choices"][:3]):  # Show first 3
                print(f"   {i+1}. {c['text']}")

        return result

    except Exception as e:
        print(f"❌ Turn processing test failed: {e}")
        return None


def test_json_format_robustness():
    """Test JSON format handling with various inputs"""
    print("\n🔍 Testing JSON format robustness...")

    test_cases = [
        {"input": "簡短回應測試", "description": "Simple short input"},
        {
            "input": "我想要詳細探索這個地方，檢查每一個角落，並且與任何可能遇到的角色進行深入對話，了解這個世界的背景故事和隱藏的秘密。",
            "description": "Long complex input",
        },
        {"input": "使用魔法", "description": "Action-oriented input"},
    ]

    # Get sample data first
    persona_resp = requests.get("http://localhost:8000/api/v1/persona/sample")
    state_resp = requests.get("http://localhost:8000/api/v1/gamestate/sample")

    if persona_resp.status_code != 200 or state_resp.status_code != 200:
        print("❌ Cannot get sample data for robustness test")
        return False

    persona = persona_resp.json()
    game_state = state_resp.json()

    passed = 0
    for i, test_case in enumerate(test_cases):
        try:
            request_data = {
                "player_input": test_case["input"],
                "persona": persona,
                "game_state": game_state,
            }

            response = requests.post(
                "http://localhost:8000/api/v1/turn", json=request_data, timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if "narration" in result and "choices" in result:
                    passed += 1
                    print(f"✅ Test {i+1} passed: {test_case['description']}")
                else:
                    print(
                        f"⚠️ Test {i+1} incomplete response: {test_case['description']}"
                    )
            else:
                print(
                    f"❌ Test {i+1} failed: {test_case['description']} (HTTP {response.status_code})"
                )

        except Exception as e:
            print(f"❌ Test {i+1} error: {test_case['description']} - {e}")

    print(f"📊 Robustness test: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


def main():
    """Run all Stage 2 smoke tests"""
    print("🚀 Stage 2 Smoke Test - LLM Core & Persona")
    print("=" * 50)

    # Test sequence
    tests_passed = 0
    total_tests = 4

    # 1. Health checks
    if test_api_health():
        tests_passed += 1

    # 2. Sample data
    persona, game_state = test_sample_data()
    if persona and game_state:
        tests_passed += 1

        # 3. Turn processing
        if test_turn_processing(persona, game_state):
            tests_passed += 1

        # 4. Robustness
        if test_json_format_robustness():
            tests_passed += 1

    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Stage 2 Test Results: {tests_passed}/{total_tests} passed")

    if tests_passed == total_tests:
        print("🎉 All tests passed! Stage 2 is ready.")
        print("\n✨ Ready for Stage 3: 中文 RAG implementation")
    else:
        print("⚠️ Some tests failed. Please check the issues above.")

    return tests_passed == total_tests


if __name__ == "__main__":
    main()
