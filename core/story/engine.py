# core/story/engine.py
"""
Text Adventure Game Engine
Manages game state, narrative generation, and player interactions
"""

import json
import uuid
import logging
import pathlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from ..llm.adapter import get_llm_adapter, ChatMessage
from ..exceptions import (
    GameError,
    SessionNotFoundError,
    InvalidChoiceError,
    ValidationError,
)
from ..config import get_config
from ..shared_cache import get_shared_cache

logger = logging.getLogger(__name__)


@dataclass
class GameState:
    """Game state data structure"""

    session_id: str
    player_name: str
    current_scene: str
    inventory: List[str]
    stats: Dict[str, int]  # health, energy, etc.
    flags: Dict[str, bool]  # story flags
    relationships: Dict[str, int]  # NPC relationship levels
    turn_count: int
    last_action: Optional[str] = None
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict"""
        data = asdict(self)
        if self.created_at:
            data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GameState":
        """Create from dict"""
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class Choice:
    """Player choice option"""

    id: str
    text: str
    description: str
    requirements: Optional[Dict[str, Any]] = None  # stat/item requirements
    consequences: Optional[Dict[str, Any]] = None  # stat/flag changes


@dataclass
class TurnResponse:
    """Game turn response"""

    narration: str
    dialogues: List[
        Dict[str, str]
    ]  # [{"speaker": "NPC", "text": "...", "emotion": "happy"}]
    choices: List[Choice]
    scene_change: bool = False
    state_updates: Optional[Dict[str, Any]] = None


class Persona:
    """Game persona/character definition"""

    def __init__(self, config: Dict[str, Any]):
        self.name = config.get("name", "Game Master")
        self.description = config.get("description", "A helpful game master")
        self.personality = config.get("personality", [])
        self.speech_style = config.get("speech_style", "friendly")
        self.world_knowledge = config.get("world_knowledge", {})
        self.safety_rules = config.get("safety_rules", [])

    def get_system_prompt(self, game_state: GameState) -> str:
        """Generate system prompt for current game state"""
        prompt_parts = [
            f"你是 {self.name}，{self.description}",
            "",
            "角色設定：",
            f"- 個性：{', '.join(self.personality)}",
            f"- 說話風格：{self.speech_style}",
            "",
            f"當前遊戲狀態：",
            f"- 玩家：{game_state.player_name}",
            f"- 場景：{game_state.current_scene}",
            f"- 回合數：{game_state.turn_count}",
            f"- 道具：{', '.join(game_state.inventory) if game_state.inventory else '無'}",
        ]

        if game_state.stats:
            prompt_parts.append(
                f"- 狀態：{', '.join(f'{k}={v}' for k, v in game_state.stats.items())}"
            )

        prompt_parts.extend(
            [
                "",
                "遊戲規則：",
                "1. 每個回合提供 2-4 個選擇",
                "2. 保持故事連貫性和沉浸感",
                "3. 根據玩家選擇動態調整劇情",
                "4. 回應必須是有效的 JSON 格式",
            ]
        )

        if self.safety_rules:
            prompt_parts.extend(
                ["", "安全規則：", *[f"- {rule}" for rule in self.safety_rules]]
            )

        return "\n".join(prompt_parts)


class StoryEngine:
    """Core story engine that manages narrative flow"""

    def __init__(self):
        self.config = get_config()
        self.cache = get_shared_cache()
        self.llm_adapter = get_llm_adapter()

        # Session storage (in-memory for MVP)
        self.sessions: Dict[str, GameState] = {}
        self.personas: Dict[str, Persona] = {}

        # Load personas
        self._load_personas()

    def _load_personas(self) -> None:
        """Load game personas from configuration"""
        try:
            persona_config_path = Path("configs/game_persona.json")
            if persona_config_path.exists():
                with open(persona_config_path, "r", encoding="utf-8") as f:
                    personas_data = json.load(f)

                for persona_id, persona_config in personas_data.items():
                    self.personas[persona_id] = Persona(persona_config)

                logger.info(f"Loaded {len(self.personas)} personas")
            else:
                # Create default persona
                default_persona = {
                    "default": {
                        "name": "智慧導師",
                        "description": "一位博學且友善的遊戲引導者",
                        "personality": ["智慧", "耐心", "創意", "幽默"],
                        "speech_style": "溫和且富有啟發性",
                        "safety_rules": [
                            "避免暴力或成人內容",
                            "保持正面且教育性的互動",
                            "尊重玩家的選擇但引導向善",
                        ],
                    }
                }

                persona_config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(persona_config_path, "w", encoding="utf-8") as f:
                    json.dump(default_persona, f, indent=2, ensure_ascii=False)

                self.personas["default"] = Persona(default_persona["default"])
                logger.info("Created default persona configuration")

        except Exception as e:
            logger.error(f"Failed to load personas: {e}")
            # Fallback persona
            self.personas["default"] = Persona(
                {"name": "遊戲大師", "description": "友善的遊戲引導者"}
            )

    def create_session(
        self,
        player_name: str,
        persona_id: str = "default",
        setting: str = "fantasy",
        difficulty: str = "normal",
    ) -> GameState:
        """Create new game session"""

        if not player_name.strip():
            raise ValidationError(
                "player_name", player_name, "Player name cannot be empty"
            )

        session_id = str(uuid.uuid4())

        # Initialize game state
        game_state = GameState(
            session_id=session_id,
            player_name=player_name.strip(),
            current_scene="開始",
            inventory=[],
            stats={"health": 100, "energy": 100, "experience": 0},
            flags={"game_started": True},
            relationships={},
            turn_count=0,
            created_at=datetime.now(),
        )

        self.sessions[session_id] = game_state

        logger.info(f"Created game session {session_id} for player {player_name}")
        return game_state

    def get_session(self, session_id: str) -> GameState:
        """Get game session by ID"""
        if session_id not in self.sessions:
            raise SessionNotFoundError(session_id)
        return self.sessions[session_id]

    def process_turn(
        self, session_id: str, player_input: str, choice_id: Optional[str] = None
    ) -> TurnResponse:
        """Process player input and generate next turn"""

        game_state = self.get_session(session_id)

        try:
            # Get persona
            persona = self.personas.get("default")  # Simplified for MVP
            if not persona:
                raise GameError("No persona available", session_id)

            # Build prompt for LLM
            system_prompt = persona.get_system_prompt(game_state)

            # Format player action
            if choice_id:
                user_prompt = f"玩家選擇：{choice_id} - {player_input}"
            else:
                user_prompt = f"玩家行動：{player_input}"

            # Add context about expected response format
            response_format_prompt = """
請以 JSON 格式回應，包含以下欄位：
{
    "narration": "場景描述和事件敘述",
    "dialogues": [{"speaker": "角色名", "text": "對話內容", "emotion": "情緒"}],
    "choices": [{"id": "choice1", "text": "選項文字", "description": "選項說明"}],
    "scene_change": false,
    "state_updates": {"stats": {"health": 95}, "flags": {"met_wizard": true}}
}
"""

            full_user_prompt = f"{user_prompt}\n\n{response_format_prompt}"

            # Generate response using LLM
            messages = [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=full_user_prompt),
            ]

            response = self.llm_adapter.chat(
                messages=messages, max_length=800, temperature=0.8  # type: ignore
            )

            # Parse LLM response as JSON
            try:
                response_data = self._parse_game_response(response.content)
            except Exception as e:
                logger.warning(f"Failed to parse LLM response as JSON: {e}")
                # Fallback response
                response_data = {
                    "narration": response.content,
                    "dialogues": [],
                    "choices": [
                        {"id": "continue", "text": "繼續", "description": "繼續故事"}
                    ],
                    "scene_change": False,
                }

            # Update game state
            self._update_game_state(game_state, player_input, response_data)

            # Build turn response
            choices = [
                Choice(
                    id=choice["id"],
                    text=choice["text"],
                    description=choice.get("description", ""),
                )
                for choice in response_data.get("choices", [])
            ]

            turn_response = TurnResponse(
                narration=response_data.get("narration", ""),
                dialogues=response_data.get("dialogues", []),
                choices=choices,
                scene_change=response_data.get("scene_change", False),
                state_updates=response_data.get("state_updates"),
            )

            logger.info(
                f"Turn processed for session {session_id}, turn {game_state.turn_count}"
            )
            return turn_response

        except Exception as e:
            logger.error(f"Turn processing failed for session {session_id}: {e}")
            raise GameError(f"Turn processing failed: {str(e)}", session_id)

    def _parse_game_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response as game JSON"""
        # Use the JSON extraction from LLM adapter
        llm = (
            list(self.llm_adapter._models.values())[0]
            if self.llm_adapter._models
            else None
        )
        if llm:
            return llm.extract_json_response(response_text)

        # Fallback parsing
        import json

        response_text = response_text.strip()

        # Remove markdown code blocks
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        return json.loads(response_text)

    def _update_game_state(
        self, game_state: GameState, player_input: str, response_data: Dict[str, Any]
    ) -> None:
        """Update game state based on turn results"""

        # Update basic state
        game_state.last_action = player_input
        game_state.turn_count += 1

        # Apply state updates from LLM response
        state_updates = response_data.get("state_updates", {})

        if "stats" in state_updates:
            game_state.stats.update(state_updates["stats"])

        if "flags" in state_updates:
            game_state.flags.update(state_updates["flags"])

        if "inventory_add" in state_updates:
            for item in state_updates["inventory_add"]:
                if item not in game_state.inventory:
                    game_state.inventory.append(item)

        if "inventory_remove" in state_updates:
            for item in state_updates["inventory_remove"]:
                if item in game_state.inventory:
                    game_state.inventory.remove(item)

        if "scene" in state_updates:
            game_state.current_scene = state_updates["scene"]

        # Apply relationships updates
        if "relationships" in state_updates:
            game_state.relationships.update(state_updates["relationships"])

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:  # type: ignore
        """Get comprehensive session summary"""
        game_state = self.get_session(session_id)

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive session summary"""
        game_state = self.get_session(session_id)

        return {
            "session_id": session_id,
            "player_name": game_state.player_name,
            "current_scene": game_state.current_scene,
            "turn_count": game_state.turn_count,
            "inventory_count": len(game_state.inventory),
            "stats": game_state.stats.copy(),
            "active_flags": [k for k, v in game_state.flags.items() if v],
            "relationships": game_state.relationships.copy(),
            "created_at": (
                game_state.created_at.isoformat() if game_state.created_at else None
            ),
            "last_action": game_state.last_action,
        }

    def save_session(self, session_id: str) -> bool:
        """Save session to persistent storage"""
        try:
            game_state = self.get_session(session_id)

            # Save to cache directory
            session_dir = self.cache.get_output_path("game_sessions")
            session_file = session_dir / f"{session_id}.json"

            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(game_state.to_dict(), f, indent=2, ensure_ascii=False)

            logger.info(f"Session {session_id} saved to {session_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
            return False

    def load_session(self, session_id: str) -> bool:
        """Load session from persistent storage"""
        try:
            session_dir = self.cache.get_output_path("game_sessions")
            session_file = session_dir / f"{session_id}.json"

            if not session_file.exists():
                return False

            with open(session_file, "r", encoding="utf-8") as f:
                session_data = json.load(f)

            game_state = GameState.from_dict(session_data)
            self.sessions[session_id] = game_state

            logger.info(f"Session {session_id} loaded from storage")
            return True

        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active and saved sessions"""
        sessions = []

        # Add active sessions
        for session_id, game_state in self.sessions.items():
            sessions.append(
                {**self.get_session_summary(session_id), "status": "active"}
            )

        # Add saved sessions not in memory
        try:
            session_dir = self.cache.get_output_path("game_sessions")
            if session_dir.exists():
                for session_file in session_dir.glob("*.json"):
                    session_id = session_file.stem
                    if session_id not in self.sessions:
                        try:
                            with open(session_file, "r", encoding="utf-8") as f:
                                session_data = json.load(f)

                            sessions.append(
                                {
                                    "session_id": session_id,
                                    "player_name": session_data.get(
                                        "player_name", "Unknown"
                                    ),
                                    "current_scene": session_data.get(
                                        "current_scene", "Unknown"
                                    ),
                                    "turn_count": session_data.get("turn_count", 0),
                                    "created_at": session_data.get("created_at"),
                                    "status": "saved",
                                }
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to read session file {session_file}: {e}"
                            )

        except Exception as e:
            logger.warning(f"Failed to list saved sessions: {e}")

        return sorted(sessions, key=lambda x: x.get("created_at", ""), reverse=True)

    def delete_session(self, session_id: str) -> bool:
        """Delete session from memory and storage"""
        try:
            # Remove from memory
            if session_id in self.sessions:
                del self.sessions[session_id]

            # Remove from storage
            session_dir = self.cache.get_output_path("game_sessions")
            session_file = session_dir / f"{session_id}.json"

            if session_file.exists():
                session_file.unlink()

            logger.info(f"Session {session_id} deleted")
            return True

        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False

    def get_personas(self) -> Dict[str, Dict[str, Any]]:
        """Get available personas"""
        return {
            persona_id: {
                "name": persona.name,
                "description": persona.description,
                "personality": persona.personality,
                "speech_style": persona.speech_style,
            }
            for persona_id, persona in self.personas.items()
        }


# Global story engine instance
_story_engine: Optional[StoryEngine] = None


def get_story_engine() -> StoryEngine:
    """Get global story engine instance"""
    global _story_engine
    if _story_engine is None:
        _story_engine = StoryEngine()
    return _story_engine
