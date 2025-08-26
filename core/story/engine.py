# core/story/engine.py
import json
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append("../..")

from core.llm.adapter import LLMAdapter
from core.llm.prompt_templates import PromptTemplates
from .game_state import GameState, EventType, RelationType
from .choice_resolver import ChoiceResolver
from .persona import PersonaManager
from core.rag.engine import ChineseRAGEngine
from .data_structures import (
    Persona,
    GameState,
    TurnRequest,
    TurnResponse,
    DialogueEntry,
    Choice,
    Relationship,
    RelationType,
)
from ..llm.base import MinimalLLM
from ..shared_cache import get_shared_cache


@dataclass
class GameState:
    """Game state data structure"""

    session_id: str
    world_id: str
    player_name: str
    current_scene: str
    inventory: List[str]
    stats: Dict[str, int]
    flags: Dict[str, bool]
    turn_count: int
    last_action: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TurnRequest:
    """Turn processing request"""

    session_id: str
    player_input: str
    choice_id: Optional[str] = None
    metadata: Dict[str, Any] = None


class StoryEngine:
    """Core story engine that manages narrative flow and game state"""

    def __init__(self, llm_adapter: LLMAdapter):
        self.cache = get_shared_cache()
        self.sessions = {}  # In-memory session storage
        self._personas = self._load_personas()
        self.llm = llm_adapter
        self.templates = PromptTemplates()

    def _load_personas(self) -> dict:
        """Load persona configurations"""
        # Mock personas - in real implementation, load from configs/game_persona.json
        return {
            "friendly_guide": {
                "name": "Friendly Guide",
                "personality": "Helpful, encouraging, patient",
                "speaking_style": "Warm and supportive",
                "knowledge_areas": ["general", "adventure", "problem_solving"],
                "safety_rules": ["no_violence", "no_adult_content"],
                "memory_slots": 10,
            },
            "mysterious_narrator": {
                "name": "Mysterious Narrator",
                "personality": "Enigmatic, wise, slightly cryptic",
                "speaking_style": "Poetic and atmospheric",
                "knowledge_areas": ["fantasy", "mystery", "lore"],
                "safety_rules": ["no_violence", "no_adult_content"],
                "memory_slots": 15,
            },
        }

    def _load_game_templates(self) -> dict:
        """Load game scenario templates"""
        return {
            "fantasy_adventure": {
                "title": "Fantasy Adventure",
                "description": "A magical world of quests and discovery",
                "starting_scene": "village_entrance",
                "available_actions": ["explore", "talk", "inventory", "help"],
            },
            "mystery_investigation": {
                "title": "Mystery Investigation",
                "description": "Solve puzzles and uncover secrets",
                "starting_scene": "investigation_start",
                "available_actions": ["investigate", "question", "analyze", "notes"],
            },
        }

    def build_persona(self, persona_id: str = "friendly_guide", **kwargs) -> dict:
        """Build persona configuration"""
        if persona_id not in self._personas:
            persona_id = "friendly_guide"

        persona = self._personas[persona_id].copy()
        persona.update(kwargs)
        return persona

    def build_game_state(
        self, world_id: str = "fantasy_adventure", player_name: str = "Player", **kwargs
    ) -> GameState:
        """Build initial game state"""
        session_id = str(uuid.uuid4())
        template = self._game_templates.get(
            world_id, self._game_templates["fantasy_adventure"]
        )

        return GameState(
            session_id=session_id,
            world_id=world_id,
            player_name=player_name,
            current_scene=template["starting_scene"],
            inventory=[],
            stats={"health": 100, "energy": 100, "knowledge": 0},
            flags={"game_started": True},
            turn_count=0,
        )

    def build_turn_request(
        self,
        *,
        player_input: str,
        persona=None,
        game_state=None,
        choice_id: Optional[str] = None,
        **kw,
    ) -> TurnRequest:
        """Build turn processing request"""
        session_id = game_state.session_id if game_state else str(uuid.uuid4())

        return TurnRequest(
            session_id=session_id,
            player_input=player_input,
            choice_id=choice_id,
            metadata=kw,
        )

    async def process_turn(
        self, turn_req: TurnRequest, game_state: GameState
    ) -> Dict[str, Any]:
        """Process a game turn"""
        try:
            # Update turn count
            game_state.turn_count += 1
            game_state.last_action = turn_req.player_input

            # Build context for LLM
            context = self._build_context(game_state, turn_req)

            # Generate LLM response
            messages = [
                {"role": "system", "content": self._build_system_prompt(game_state)},
                {"role": "user", "content": context},
            ]

            llm_response = self.llm.chat(messages)

            # Parse response into structured format
            result = self._parse_llm_response(llm_response, game_state)

            # Update game state based on response
            self._update_game_state(game_state, result)

            # Store session
            self.sessions[game_state.session_id] = game_state

            return result

        except Exception as e:
            raise RuntimeError(f"Turn processing failed: {str(e)}")

    def _build_context(self, game_state: GameState, turn_req: TurnRequest) -> str:
        """Build context string for LLM"""
        context = f"""
Current Scene: {game_state.current_scene}
Player: {game_state.player_name}
Turn: {game_state.turn_count}
Player Action: {turn_req.player_input}
Inventory: {', '.join(game_state.inventory) if game_state.inventory else 'empty'}
Stats: {game_state.stats}
"""
        return context.strip()

    def _build_system_prompt(self, game_state: GameState) -> str:
        """Build system prompt for LLM"""
        return f"""You are a text adventure game narrator for a {game_state.world_id} setting.
Respond with engaging narrative and provide 2-3 action choices.
Keep responses family-friendly and encourage exploration.
Format your response as narrative followed by numbered choices."""

    def _parse_llm_response(self, response: str, game_state: GameState) -> dict:
        """Parse LLM response into structured format"""
        # Simple parsing - in real implementation, could use more sophisticated parsing
        lines = response.split("\n")
        narration = []
        choices = []

        in_choices = False
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if any(
                choice_indicator in line.lower()
                for choice_indicator in ["1.", "2.", "3.", "choice", "option"]
            ):
                in_choices = True
                choices.append(line)
            elif not in_choices:
                narration.append(line)

        # Default choices if none parsed
        if not choices:
            choices = [
                "1. Look around carefully",
                "2. Continue forward",
                "3. Check inventory",
            ]

        return {
            "narration": (
                " ".join(narration) if narration else "You continue your adventure..."
            ),
            "dialogues": [],  # Could extract dialogue from narration
            "choices": choices,
            "scene_change": game_state.current_scene,  # Default to same scene
            "inventory_changes": [],
            "stat_changes": {},
            "flags_changed": {},
        }

    def _update_game_state(self, game_state: GameState, result: dict):
        """Update game state based on turn result"""
        # Update scene if changed
        if (
            result.get("scene_change")
            and result["scene_change"] != game_state.current_scene
        ):
            game_state.current_scene = result["scene_change"]

        # Update inventory
        for item in result.get("inventory_changes", []):
            if item.startswith("+"):
                game_state.inventory.append(item[1:])
            elif item.startswith("-"):
                if item[1:] in game_state.inventory:
                    game_state.inventory.remove(item[1:])

        # Update stats
        for stat, change in result.get("stat_changes", {}).items():
            if stat in game_state.stats:
                game_state.stats[stat] += change

        # Update flags
        game_state.flags.update(result.get("flags_changed", {}))

    def get_session(self, session_id: str) -> Optional[GameState]:
        """Get game session by ID"""
        return self.sessions.get(session_id)

    def save_session(self, game_state: GameState):
        """Save game session"""
        self.sessions[game_state.session_id] = game_state

    def process_turn(self, request: TurnRequest) -> TurnResponse:
        """Process a single story turn"""
        try:
            # Build prompt with context
            system_prompt = self.templates.SYSTEM_PROMPT
            user_prompt = self.templates.build_user_prompt(
                request.player_input,
                request.persona,
                request.game_state,
                request.choice_id,  # type: ignore
            )

            # Format messages for LLM
            messages = self.llm.format_messages(system_prompt, user_prompt)

            # Generate response
            raw_response = self.llm.generate(messages, max_tokens=1024, temperature=0.8)

            # Parse JSON response
            response_data = self.llm.extract_json_response(raw_response)

            # Create structured response
            turn_response = self._create_turn_response(
                response_data, request.game_state
            )

            # Update game state
            updated_state = self._update_game_state(request, turn_response)
            turn_response.updated_state = updated_state

            return turn_response

        except Exception as e:
            print(f"Story engine error: {e}")
            # Return fallback response
            return TurnResponse(
                narration=f"故事引擎遇到錯誤，但冒險仍在繼續... 錯誤: {str(e)}",
                choices=[
                    Choice(id="continue", text="繼續", description="嘗試繼續故事")
                ],
            )

    def _create_turn_response(
        self, data: Dict[str, Any], game_state: GameState
    ) -> TurnResponse:
        """Create TurnResponse from parsed JSON data"""
        # Extract narration
        narration = data.get("narration", "故事繼續進行...")

        # Extract dialogues
        dialogues = []
        for d in data.get("dialogues", []):
            if isinstance(d, dict) and "speaker" in d and "text" in d:
                dialogues.append(
                    DialogueEntry(
                        speaker=d["speaker"], text=d["text"], emotion=d.get("emotion")
                    )
                )

        # Extract choices
        choices = []
        for c in data.get("choices", []):
            if isinstance(c, dict) and "id" in c and "text" in c:
                choices.append(
                    Choice(
                        id=c["id"], text=c["text"], description=c.get("description", "")
                    )
                )

        # Default choice if none provided
        if not choices:
            choices.append(Choice(id="continue", text="繼續", description="繼續故事"))

        return TurnResponse(
            narration=narration,
            dialogues=dialogues,
            choices=choices,
            scene_change=data.get("scene_change"),
        )

    def _update_game_state(
        self, request: TurnRequest, response: TurnResponse
    ) -> GameState:
        """Update game state based on turn results"""
        state = request.game_state

        # Increment turn count
        state.turn_count += 1

        # Record choice if made
        if request.choice_id:
            state.choice_history.append(
                {
                    "turn": state.turn_count - 1,
                    "choice_id": request.choice_id,
                    "text": request.player_input,
                }
            )

        # Update scene if changed
        if response.scene_change:
            state.scene_id = response.scene_change

        # Add timeline note
        if response.narration:
            # Simple summary of the turn
            summary = (
                response.narration[:100] + "..."
                if len(response.narration) > 100
                else response.narration
            )
            state.timeline_notes.append(f"回合 {state.turn_count}: {summary}")

        # Keep only recent history to prevent memory bloat
        if len(state.choice_history) > 10:
            state.choice_history = state.choice_history[-10:]
        if len(state.timeline_notes) > 15:
            state.timeline_notes = state.timeline_notes[-15:]

        return state

    def create_sample_persona(self) -> Persona:
        """Create a sample persona for testing"""
        return Persona(
            name="艾莉絲",
            age=22,
            personality=["好奇", "勇敢", "善良", "有些衝動"],
            background="來自小鎮的冒險家，夢想探索古老的遺跡和發現失落的魔法",
            speaking_style="活潑直接，偶爾會用一些現代用語，但在嚴肅時刻會變得深思熟慮",
            appearance="棕色長髮，綠色眼眸，身穿實用的冒險裝備",
            goals=["尋找失落的魔法神器", "保護無辜的人", "證明自己的勇氣"],
            secrets=["擁有微弱的魔法感知能力", "害怕黑暗"],
            memory_preferences={"adventure": 0.9, "magic": 0.8, "friendship": 0.7},
        )

    def create_sample_game_state(self) -> GameState:
        """Create a sample game state for testing"""
        return GameState(
            scene_id="古老圖書館",
            turn_count=0,
            current_location="神秘圖書館的入口大廳",
            flags={"first_visit": True, "has_torch": True, "library_key": False},
            inventory=["冒險家背包", "小型手電筒", "地圖"],
            timeline_notes=["進入了傳說中的古老圖書館"],
        )
