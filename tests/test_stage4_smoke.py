# tests/test_stage4_smoke.py
"""
Stage 4 Story Engine Smoke Test
Tests the complete story system including GameState, ChoiceResolver, and API endpoints
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Shared cache bootstrap
import pathlib, torch

AI_CACHE_ROOT = os.getenv("AI_CACHE_ROOT", "../ai_warehouse/cache")
for k, v in {
    "HF_HOME": f"{AI_CACHE_ROOT}/hf",
    "TRANSFORMERS_CACHE": f"{AI_CACHE_ROOT}/hf/transformers",
    "HF_DATASETS_CACHE": f"{AI_CACHE_ROOT}/hf/datasets",
    "HUGGINGFACE_HUB_CACHE": f"{AI_CACHE_ROOT}/hf/hub",
    "TORCH_HOME": f"{AI_CACHE_ROOT}/torch",
}.items():
    os.environ[k] = v
    pathlib.Path(v).mkdir(parents=True, exist_ok=True)
print("[cache]", AI_CACHE_ROOT, "| GPU:", torch.cuda.is_available())

from core.story.game_state import GameState, RelationType, EventType
from core.story.choice_resolver import (
    ChoiceResolver,
    Choice,
    ChoiceCondition,
    ChoiceConsequence,
)
from core.story.persona import PersonaManager, Persona, PersonaTrait
from core.story.engine import StoryEngine, TurnRequest


# Mock LLM and RAG for testing
class MockLLMAdapter:
    async def generate(
        self, prompt: str, max_tokens: int = 1000, temperature: float = 0.8
    ) -> str:
        # Return a mock story response in JSON format
        mock_response = {
            "narration": "測試敘述：你發現自己在一個神秘的圖書館中，古老的書籍散發著淡淡的魔法光芒。",
            "dialogues": [
                {
                    "speaker": "圖書管理員",
                    "text": "歡迎來到知識之殿，旅者。你在尋找什麼？",
                }
            ],
            "choices": [
                {
                    "id": "ask_about_books",
                    "text": "詢問關於魔法書籍",
                    "consequences": {
                        "type": "relationship",
                        "target": "librarian",
                        "value": {"affinity_delta": 5},
                    },
                },
                {
                    "id": "explore_library",
                    "text": "自由探索圖書館",
                    "consequences": {
                        "type": "flag",
                        "target": "library_explored",
                        "value": True,
                    },
                },
            ],
            "citations": ["library_lore_001", "character_librarian_intro"],
            "state_changes": {
                "flags": [{"flag_id": "visited_library", "value": True}],
                "relationships": [],
            },
        }
        return json.dumps(mock_response, ensure_ascii=False, indent=2)


class MockRAGEngine:
    async def retrieve(self, query: str, world_id: str, scopes: list, top_k: int = 8):
        # Return mock RAG results
        return [
            {
                "doc_id": "library_lore_001",
                "text": "魔法圖書館是一個充滿古老知識的地方，由神秘的圖書管理員守護。",
                "score": 0.9,
            },
            {
                "doc_id": "character_librarian_intro",
                "text": "圖書管理員是一位博學的長者，對所有求知者都很友善。",
                "score": 0.8,
            },
        ]

    async def write_memory(
        self, world_id: str, scope: str, text: str, metadata: dict = None
    ):
        print(f"[RAG] Writing memory to {scope}: {text[:50]}...")


async def test_game_state():
    """Test GameState functionality"""
    print("=== Testing GameState ===")

    game_state = GameState(
        world_id="test_world", player_name="測試玩家", current_scene="library_entrance"
    )

    # Test relationship management
    game_state.update_relationship(
        character_id="librarian",
        relation_type=RelationType.FRIEND,
        affinity_delta=10,
        trust_delta=5,
        notes="初次見面，印象良好",
    )

    # Test inventory
    success = game_state.add_item(
        item_id="magic_book", name="魔法書", quantity=1, description="一本古老的魔法書"
    )
    print(f"Added magic book: {success}")

    # Test flags
    game_state.set_flag("visited_library", True, "已經參觀過圖書館")

    # Test location discovery
    game_state.discover_location(
        location_id="library_main_hall",
        name="圖書館大廳",
        properties={"ambient": "mystical", "light_level": "dim"},
    )

    print(f"Player: {game_state.player_name}")
    print(f"Relationships: {game_state.get_relationship_summary()}")
    print(f"Inventory: {list(game_state.inventory.keys())}")
    print(f"Flags: {list(game_state.world_flags.keys())}")
    print(f"Timeline events: {len(game_state.timeline)}")

    return game_state


def test_choice_resolver():
    """Test ChoiceResolver functionality"""
    print("\n=== Testing ChoiceResolver ===")

    resolver = ChoiceResolver()

    # Create test game state
    game_state = GameState(world_id="test", player_name="Test Player")
    game_state.add_item("healing_potion", "治療藥水", 2)
    game_state.player_health = 30  # Low health
    game_state.set_flag("has_key", True)

    # Test template choices
    available_choices = resolver.get_available_choices(game_state)
    print(f"Available template choices: {len(available_choices)}")
    for choice in available_choices:
        print(f"  - {choice.id}: {choice.text}")

    # Test dynamic choice for healing potion
    dynamic_choices = resolver._generate_dynamic_choices(game_state)
    print(f"Dynamic choices: {len(dynamic_choices)}")
    for choice in dynamic_choices:
        print(f"  - {choice.id}: {choice.text}")

    # Test choice resolution
    if dynamic_choices:
        choice = dynamic_choices[0]
        print(f"\nResolving choice: {choice.id}")
        changes = resolver.resolve_choice(choice.id, game_state)
        print(f"Changes applied: {changes}")


def test_persona_manager():
    """Test PersonaManager functionality"""
    print("\n=== Testing PersonaManager ===")

    manager = PersonaManager()

    # Create test persona
    librarian_data = {
        "id": "librarian",
        "name": "艾莉安娜",
        "description": "古老圖書館的神秘管理員",
        "background": "守護知識超過百年的學者",
        "personality": {
            "wisdom": {"value": 9, "description": "極其博學"},
            "patience": {"value": 8, "description": "非常有耐心"},
            "mystery": {"value": 7, "description": "帶有神秘感"},
        },
        "speech_style": "用詞優雅，富有詩意",
        "goals": ["保護古老知識", "引導求知者"],
        "appearance": "銀髮長袍，眼中閃爍智慧之光",
    }

    persona = manager.create_persona_from_data(librarian_data)
    manager.add_persona(persona)

    # Test memory addition
    manager.update_persona_memory(
        character_id="librarian",
        event_description="遇到了一位新的求知者，對方似乎對魔法很感興趣",
        importance=6,
        emotional_impact=2,
        related_characters=["player"],
    )

    # Test dialogue style generation
    style_prompt = manager.generate_dialogue_style_prompt(
        "librarian", "玩家詢問關於魔法書籍"
    )
    print(f"Dialogue style prompt:\n{style_prompt}")

    # Test personality summary
    print(f"Personality: {persona.get_personality_summary()}")

    return manager


async def test_story_engine():
    """Test complete StoryEngine workflow"""
    print("\n=== Testing StoryEngine ===")

    # Initialize components
    llm_adapter = MockLLMAdapter()
    rag_engine = MockRAGEngine()
    persona_manager = test_persona_manager()

    story_engine = StoryEngine(llm_adapter, rag_engine, persona_manager)

    # Create test game state
    game_state = GameState(
        world_id="fantasy_library",
        player_name="勇敢的冒險者",
        current_scene="library_entrance",
        current_location="entrance_hall",
    )

    # Process a turn
    turn_request = TurnRequest(
        player_input="我想了解這個圖書館的歷史",
        world_id="fantasy_library",
        session_id="test_session_001",
    )

    print("Processing story turn...")
    response = await story_engine.process_turn(turn_request, game_state)

    print(f"Narration: {response.narration}")
    print(f"Dialogues: {len(response.dialogues)}")
    for dialogue in response.dialogues:
        print(f"  {dialogue['speaker']}: {dialogue['text']}")

    print(f"Choices: {len(response.choices)}")
    for choice in response.choices:
        print(f"  - {choice['text']}")

    print(f"Citations: {response.citations}")
    print(f"Turn count: {game_state.turn_count}")

    return story_engine, game_state


async def test_api_integration():
    """Test API integration (mock)"""
    print("\n=== Testing API Integration ===")

    # This would test the actual FastAPI endpoints
    # For now, just verify the request/response models work

    from api.routers.story import TurnRequestModel, StoryResponseModel

    # Test request model
    turn_request = TurnRequestModel(
        player_input="測試輸入", world_id="test_world", session_id="test_session"
    )

    print(f"API Request: {turn_request.dict()}")

    # Mock response
    response_data = {
        "narration": "測試敘述",
        "dialogues": [{"speaker": "NPC", "text": "測試對話"}],
        "choices": [{"id": "test", "text": "測試選擇"}],
        "citations": ["test_citation"],
        "game_state_changes": {"flags": []},
        "metadata": {"turn_count": 1},
    }

    response = StoryResponseModel(**response_data)
    print(f"API Response validated: {response.narration[:30]}...")


async def main():
    """Run all tests"""
    print("🚀 Starting Stage 4 Story Engine Smoke Tests\n")

    try:
        # Test individual components
        game_state = test_game_state()
        test_choice_resolver()
        persona_manager = test_persona_manager()

        # Test integrated story engine
        story_engine, final_game_state = await test_story_engine()

        # Test API models
        await test_api_integration()

        print("\n✅ All Story Engine tests passed!")
        print(f"Final game state: Turn {final_game_state.turn_count}")
        print(f"Timeline events: {len(final_game_state.timeline)}")

        # Show game state summary
        print("\n📊 Final Game State Summary:")
        print(f"  Session: {final_game_state.session_id}")
        print(f"  Player: {final_game_state.player_name}")
        print(f"  Location: {final_game_state.current_location}")
        print(f"  Relationships: {len(final_game_state.relationships)}")
        print(f"  Inventory: {len(final_game_state.inventory)} items")
        print(f"  Flags: {len(final_game_state.world_flags)} flags")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
