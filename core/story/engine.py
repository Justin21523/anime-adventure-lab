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


class StoryEngine:
    """Core story engine that manages narrative flow and game state"""

    def __init__(self, llm_adapter: LLMAdapter):
        self.llm = llm_adapter
        self.templates = PromptTemplates()

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
