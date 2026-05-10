#!/usr/bin/env python3
"""
Story Integration Test Script
Tests all story system features with mock AI models
"""

import json
import requests
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Configuration
API_BASE = "http://localhost:8000/api/v1"
TEST_SCENARIOS_PATH = "tests/mock_data/story_test_scenarios.json"

class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    BOLD = '\033[1m'
    NC = '\033[0m'  # No Color

class StoryTester:
    def __init__(self):
        self.api_base = API_BASE
        self.session_id = None
        self.test_results = []

    def print_header(self, text: str):
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.NC}")
        print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.NC}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.NC}\n")

    def print_success(self, text: str):
        print(f"{Colors.GREEN}✓ {text}{Colors.NC}")

    def print_error(self, text: str):
        print(f"{Colors.RED}✗ {text}{Colors.NC}")

    def print_info(self, text: str):
        print(f"{Colors.YELLOW}ℹ {text}{Colors.NC}")

    def check_backend(self) -> bool:
        """Check if backend is running"""
        try:
            response = requests.get(f"{self.api_base}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def load_scenarios(self) -> Dict[str, Any]:
        """Load test scenarios from JSON"""
        try:
            with open(TEST_SCENARIOS_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.print_error(f"Failed to load scenarios: {e}")
            return {"scenarios": [], "test_personas": []}

    def create_session(self, persona_id: str, initial_prompt: str, use_agent: bool = True, use_rag: bool = True) -> str:
        """Create a new story session"""
        try:
            response = requests.post(
                f"{self.api_base}/story/session",
                json={
                    "player_name": "測試玩家",
                    "persona_id": persona_id,
                    "setting": initial_prompt,
                    "difficulty": "medium",
                    "enhanced_mode": True,
                    "use_agent": use_agent,
                    "enrich_with_rag": use_rag
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data.get("session_id")
        except Exception as e:
            raise Exception(f"Failed to create session: {e}")

    def send_turn(self, session_id: str, user_input: str, agent_action: Dict = None) -> Dict:
        """Send a turn/dialogue to the session"""
        try:
            payload = {
                "session_id": session_id,
                "player_input": user_input,
                "use_agent": True,
                "enrich_with_rag": True
            }

            if agent_action:
                payload["scenario_type"] = agent_action.get("type", "autonomous_action")
                payload["scenario_data"] = agent_action

            response = requests.post(
                f"{self.api_base}/story/turn",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Failed to send turn: {e}")

    def get_session(self, session_id: str) -> Dict:
        """Get session details"""
        try:
            response = requests.get(
                f"{self.api_base}/story/session/{session_id}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Failed to get session: {e}")

    def check_agent_tools(self) -> List[str]:
        """Check available agent tools"""
        try:
            response = requests.get(f"{self.api_base}/agent/tools", timeout=10)
            response.raise_for_status()
            data = response.json()
            tools = data.get("tools", [])
            return [tool["name"] for tool in tools]
        except Exception as e:
            self.print_error(f"Failed to get agent tools: {e}")
            return []

    def test_scenario(self, scenario: Dict) -> bool:
        """Test a complete scenario"""
        scenario_name = scenario["name"]
        scenario_id = scenario["id"]

        print(f"\n{Colors.BOLD}Testing Scenario: {scenario_name}{Colors.NC}")
        print(f"Description: {scenario['description']}")
        print(f"Persona: {scenario['persona_id']}\n")

        try:
            # Create session
            self.print_info("Creating session...")
            session_id = self.create_session(
                scenario["persona_id"],
                scenario["initial_prompt"]
            )
            self.print_success(f"Session created: {session_id}")

            # Execute turns
            for turn_data in scenario["turns"]:
                turn_num = turn_data["turn"]
                user_input = turn_data["user_input"]
                agent_action = turn_data.get("agent_action")
                expected_elements = turn_data.get("expected_elements", [])

                print(f"\n{Colors.BOLD}Turn {turn_num}:{Colors.NC}")
                print(f"  Input: {user_input}")

                if agent_action:
                    print(f"  Agent Action: {agent_action.get('type', 'N/A')}")
                    if agent_action.get('tool'):
                        print(f"  Tool: {agent_action['tool']}")

                # Send turn
                result = self.send_turn(session_id, user_input, agent_action)

                # Check response
                narration = result.get("narration", "")
                choices = result.get("choices", [])
                agent_insights = result.get("agent_insights", {})

                print(f"  Narration: {narration[:100]}...")
                print(f"  Choices: {len(choices)}")

                if agent_insights:
                    print(f"  Agent Insights: {agent_insights.get('summary', 'N/A')[:80]}...")

                # Verify expected elements
                found_elements = []
                for element in expected_elements:
                    if element.lower() in narration.lower():
                        found_elements.append(element)

                if expected_elements:
                    coverage = len(found_elements) / len(expected_elements) * 100
                    if coverage >= 50:  # At least 50% of expected elements
                        self.print_success(f"Content check: {coverage:.0f}% match")
                    else:
                        self.print_error(f"Content check: Only {coverage:.0f}% match")

            # Get final session state
            final_session = self.get_session(session_id)
            total_turns = len(final_session.get("turns", []))

            print(f"\n{Colors.BOLD}Scenario Complete!{Colors.NC}")
            print(f"Total turns: {total_turns}")
            self.print_success(f"Scenario '{scenario_name}' passed")

            return True

        except Exception as e:
            self.print_error(f"Scenario '{scenario_name}' failed: {e}")
            return False

    def run_all_tests(self):
        """Run all test scenarios"""
        self.print_header("Story System Integration Test")

        # Check backend
        self.print_info("Checking backend status...")
        if not self.check_backend():
            self.print_error("Backend is not running!")
            self.print_info("Start it with: conda run -n ai_env python api/main.py")
            return False
        self.print_success("Backend is running")

        # Check agent tools
        self.print_info("Checking agent tools...")
        tools = self.check_agent_tools()
        if tools:
            self.print_success(f"Found {len(tools)} agent tools")
            print(f"  Available tools: {', '.join(tools[:5])}...")
        else:
            self.print_error("No agent tools found")

        # Load scenarios
        self.print_info("Loading test scenarios...")
        data = self.load_scenarios()
        scenarios = data.get("scenarios", [])

        if not scenarios:
            self.print_error("No scenarios found!")
            return False

        self.print_success(f"Loaded {len(scenarios)} scenarios")

        # Run tests
        print(f"\n{Colors.BOLD}Starting scenario tests...{Colors.NC}\n")

        passed = 0
        failed = 0

        for i, scenario in enumerate(scenarios, 1):
            print(f"\n{Colors.BOLD}[{i}/{len(scenarios)}]{Colors.NC}")
            if self.test_scenario(scenario):
                passed += 1
            else:
                failed += 1

        # Print summary
        self.print_header("Test Summary")
        print(f"Total scenarios: {len(scenarios)}")
        print(f"{Colors.GREEN}Passed: {passed}{Colors.NC}")
        print(f"{Colors.RED}Failed: {failed}{Colors.NC}")

        success_rate = (passed / len(scenarios)) * 100 if scenarios else 0
        print(f"\nSuccess rate: {success_rate:.1f}%")

        if success_rate >= 70:
            self.print_success("Integration test PASSED!")
            return True
        else:
            self.print_error("Integration test FAILED")
            return False

def main():
    tester = StoryTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
