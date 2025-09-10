# =============================================================================
# core/story/engine.py
"""
Main Story Engine
Coordinates all story components and manages game flow
"""

import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .game_state import GameSession, GameState, PlayerStats
from .persona import PersonaManager, GamePersona
from .narrative import NarrativeGenerator, StoryContext
from .choices import ChoiceManager, GameChoice
from ..exceptions import GameError, SessionNotFoundError, InvalidChoiceError
from ..config import get_config
from ..shared_cache import get_shared_cache

logger = logging.getLogger(__name__)


class StoryEngine:
    """Main story engine coordinating all game systems"""

    def __init__(self, config_dir: Optional[Path] = None):
        self.config = get_config()
        self.cache = get_shared_cache()

        # Initialize component managers
        self.persona_manager = PersonaManager(
            config_dir / "game_persona.json" if config_dir else None
        )
        self.narrative_generator = NarrativeGenerator()
        self.choice_manager = ChoiceManager()

        # Session storage
        self.sessions: Dict[str, GameSession] = {}
        self.sessions_path = self.cache / "game_sessions"
        self.sessions_path.mkdir(exist_ok=True)

        # Load existing sessions
        self._load_sessions()

        logger.info("Story engine initialized")

    def _load_sessions(self):
        """Load existing game sessions from storage"""
        try:
            for session_file in self.sessions_path.glob("*.json"):
                with open(session_file, "r", encoding="utf-8") as f:
                    session_data = json.load(f)
                    session = GameSession.from_dict(session_data)
                    self.sessions[session.session_id] = session
            logger.info(f"Loaded {len(self.sessions)} existing sessions")
        except Exception as e:
            logger.error(f"Failed to load sessions: {e}")

    def _save_session(self, session: GameSession):
        """Save session to storage"""
        try:
            session_file = self.sessions_path / f"{session.session_id}.json"
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")

    def create_session(
        self,
        player_name: str,
        persona_id: str = "wise_sage",
        setting: str = "fantasy",
        difficulty: str = "medium",
    ) -> GameSession:
        """Create new game session"""

        # Validate persona
        persona = self.persona_manager.get_persona(persona_id)
        if not persona:
            raise GameError(f"Unknown persona: {persona_id}")

        # Create initial game state
        initial_state = GameState(
            scene_id="opening",
            scene_description="",
            available_choices=[],
            story_context={
                "setting": setting,
                "difficulty": difficulty,
                "current_location": "起點",
                "time_of_day": "清晨",
                "weather": "晴朗",
                "mood": "充滿希望",
            },
        )

        # Create session
        session = GameSession(
            session_id=f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{player_name}",
            player_name=player_name,
            persona_id=persona_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            turn_count=0,
            stats=PlayerStats(),
            inventory=[],
            current_state=initial_state,
        )

        # Store session
        self.sessions[session.session_id] = session
        self._save_session(session)

        logger.info(f"Created new session: {session.session_id}")
        return session

    def get_session(self, session_id: str) -> GameSession:
        """Get game session by ID"""
        session = self.sessions.get(session_id)
        if not session:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        return session

    async def process_turn(
        self, session_id: str, player_input: str, choice_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a player turn and generate response"""

        session = self.get_session(session_id)
        persona = self.persona_manager.get_persona(session.persona_id)

        if not persona:
            raise GameError(f"Persona not found: {session.persona_id}")

        try:
            # Handle choice execution if provided
            choice_result = None
            if choice_id and session.current_state.available_choices:
                choice_result = await self._execute_choice(session, choice_id)

            # Generate story context
            story_context = StoryContext(
                current_scene=session.current_state.scene_id,
                previous_scenes=[
                    h["scene_id"] for h in session.history[-3:]
                ],  # Last 3 scenes
                active_characters=[persona.name],
                current_location=session.current_state.story_context.get(
                    "current_location", "unknown"
                ),
                time_of_day=session.current_state.story_context.get(
                    "time_of_day", "unknown"
                ),
                weather=session.current_state.story_context.get("weather", "unknown"),
                mood=session.current_state.story_context.get("mood", "neutral"),
                plot_points=session.current_state.story_context.get("plot_points", []),
                world_state=session.current_state.flags,
            )

            # Generate persona prompt
            persona_prompt = self.persona_manager.generate_persona_prompt(
                persona, session.current_state.story_context
            )

            # Generate narrative response
            scene_type = (
                "opening"
                if session.turn_count == 0
                else "resolution" if choice_result else "exploration"
            )
            narrative_response = await self.narrative_generator.generate_scene(
                story_context, persona_prompt, player_input, scene_type
            )

            # Generate new choices
            new_choices = self.choice_manager.generate_choices(
                session.current_state.story_context,
                session.stats.to_dict(),
                session.inventory,
                session.current_state.flags,
            )

            # Update game state
            session.current_state.scene_description = narrative_response
            session.current_state.available_choices = [
                {
                    "choice_id": choice.choice_id,
                    "text": choice.text,
                    "type": choice.choice_type.value,
                    "difficulty": choice.difficulty.value,
                    "can_choose": choice.can_choose(
                        session.stats.to_dict(),
                        session.inventory,
                        session.current_state.flags,
                    )[0],
                }
                for choice in new_choices
            ]

            # Add to history
            session.add_to_history(player_input, narrative_response, choice_id)

            # Save session
            self._save_session(session)

            # Prepare response
            response = {
                "session_id": session.session_id,
                "turn_count": session.turn_count,
                "narrative": narrative_response,
                "choices": session.current_state.available_choices,
                "stats": session.stats.to_dict(),
                "inventory": session.inventory,
                "scene_id": session.current_state.scene_id,
                "flags": session.current_state.flags,
            }

            if choice_result:
                response["choice_result"] = choice_result

            logger.info(
                f"Processed turn for session {session_id}, turn {session.turn_count}"
            )
            return response

        except Exception as e:
            logger.error(f"Failed to process turn for session {session_id}: {e}")
            raise GameError(f"Failed to process turn: {str(e)}")

    async def _execute_choice(
        self, session: GameSession, choice_id: str
    ) -> Dict[str, Any]:
        """Execute a player choice and apply consequences"""

        # Find the choice
        available_choices = self.choice_manager.generate_choices(
            session.current_state.story_context,
            session.stats.to_dict(),
            session.inventory,
            session.current_state.flags,
        )

        choice = self.choice_manager.validate_choice(choice_id, available_choices)
        if not choice:
            raise InvalidChoiceError(f"Invalid choice: {choice_id}")

        # Check if choice can be made
        can_choose, reason = choice.can_choose(
            session.stats.to_dict(), session.inventory, session.current_state.flags
        )
        if not can_choose:
            raise InvalidChoiceError(f"Cannot make choice: {reason}")

        # Execute choice
        success, consequences = choice.execute(
            session.stats.to_dict(), session.stats.luck
        )

        # Apply consequences
        result = {"success": success, "consequences": {}}

        # Apply stat changes
        if "stats" in consequences:
            for stat, change in consequences["stats"].items():
                if hasattr(session.stats, stat):
                    current_value = getattr(session.stats, stat)
                    new_value = max(
                        0, current_value + change
                    )  # Prevent negative values
                    setattr(session.stats, stat, new_value)
                    result["consequences"][stat] = change

        # Apply flag changes
        if "flags" in consequences:
            session.current_state.flags.update(consequences["flags"])
            result["consequences"]["flags"] = consequences["flags"]

        # Apply inventory changes
        if "add_items" in consequences:
            session.inventory.extend(consequences["add_items"])
            result["consequences"]["add_items"] = consequences["add_items"]

        if "remove_items" in consequences:
            for item in consequences["remove_items"]:
                if item in session.inventory:
                    session.inventory.remove(item)
            result["consequences"]["remove_items"] = consequences["remove_items"]

        return result

    def list_sessions(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """List game sessions"""
        sessions = []
        for session in self.sessions.values():
            if active_only and not session.is_active:
                continue

            sessions.append(
                {
                    "session_id": session.session_id,
                    "player_name": session.player_name,
                    "persona_id": session.persona_id,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                    "turn_count": session.turn_count,
                    "is_active": session.is_active,
                }
            )

        return sorted(sessions, key=lambda x: x["updated_at"], reverse=True)

    def end_session(self, session_id: str):
        """End a game session"""
        session = self.get_session(session_id)
        session.is_active = False
        session.updated_at = datetime.now()
        self._save_session(session)
        logger.info(f"Ended session: {session_id}")

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get session summary"""
        session = self.get_session(session_id)

        return {
            "session_id": session.session_id,
            "player_name": session.player_name,
            "persona_name": (
                self.persona_manager.get_persona(session.persona_id).name
                if self.persona_manager.get_persona(session.persona_id)
                else "Unknown"
            ),
            "turn_count": session.turn_count,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "stats": session.stats.to_dict(),
            "inventory": session.inventory,
            "current_scene": session.current_state.scene_id,
            "is_active": session.is_active,
            "total_history": len(session.history),
        }

    def list_personas(self) -> List[Dict[str, Any]]:
        """List available personas"""
        personas = []
        for persona in self.persona_manager.list_personas():
            personas.append(
                {
                    "persona_id": persona.persona_id,
                    "name": persona.name,
                    "description": persona.description,
                    "personality_traits": persona.personality_traits,
                    "special_abilities": persona.special_abilities,
                }
            )
        return personas


# Global engine instance
_story_engine: Optional[StoryEngine] = None


def get_story_engine() -> StoryEngine:
    """Get global story engine instance"""
    global _story_engine
    if _story_engine is None:
        config = get_config()
        config_dir = Path(config.config_dir)
        _story_engine = StoryEngine(config_dir)
    return _story_engine
