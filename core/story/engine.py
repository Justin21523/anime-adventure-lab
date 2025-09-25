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
from typing import Dict, List, Optional, Any, Union, Tuple

from .game_state import GameSession, GameState, PlayerStats
from .persona import PersonaManager, GamePersona
from .narrative import NarrativeGenerator, StoryContext
from .choices import ChoiceManager, GameChoice
from .story_system import (
    GameCharacter,
    CharacterRole,
    CharacterState,
    SceneContext,
    SceneType,
    SceneMood,
    StoryContextMemory,
    ContextualChoice,
    EnhancedNarrativeGenerator,
)
from ..exceptions import GameError, SessionNotFoundError, InvalidChoiceError
from ..config import get_config
from ..shared_cache import get_shared_cache

logger = logging.getLogger(__name__)


class StoryEngine:
    """Main story engine coordinating all game systems"""

    def __init__(self, config_dir: Optional[Path] = None, enhanced_mode: bool = True):
        self.config = get_config()
        self.cache = get_shared_cache()
        self.enhanced_mode = enhanced_mode

        # Initialize component managers
        self.persona_manager = PersonaManager(
            config_dir / "game_persona.json" if config_dir else None
        )

        # Choose narrative generator based on mode
        if self.enhanced_mode:
            try:
                self.narrative_generator = EnhancedNarrativeGenerator()
                logger.info("Enhanced narrative generator initialized")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize enhanced generator, using classic: {e}"
                )
                self.narrative_generator = NarrativeGenerator()
                self.enhanced_mode = False
        else:
            self.narrative_generator = NarrativeGenerator()

        self.choice_manager = ChoiceManager()

        # Session storage
        self.sessions: Dict[str, GameSession] = {}
        self.sessions_path = Path(self.cache) / "game_sessions"  # type: ignore
        self.sessions_path.mkdir(exist_ok=True)

        # Enhanced features
        if self.enhanced_mode:
            self.context_memories: Dict[str, StoryContextMemory] = {}
            self.contextual_choices = self._load_contextual_choices()

        # Load existing sessions
        self._load_sessions()

        logger.info("Story engine initialized")

    def _load_contextual_choices(self) -> Dict[str, List[ContextualChoice]]:
        """Load contextual choice templates"""
        return {
            "dialogue": [
                ContextualChoice(
                    choice_id="friendly_greeting",
                    base_text="友善地打招呼",
                    choice_type="dialogue",
                    context_sensitive=True,
                    text_variations={
                        "peaceful": "溫和地向對方問好",
                        "tense": "謹慎地向對方致意",
                        "mysterious": "小心翼翼地接近",
                    },
                    relationship_changes={"any": 1},
                    success_chance=0.9,
                ),
                ContextualChoice(
                    choice_id="ask_about_location",
                    base_text="詢問關於這個地方的情況",
                    choice_type="dialogue",
                    required_characters=["any_npc"],
                    relationship_changes={"any": 1},
                    success_chance=0.8,
                ),
                ContextualChoice(
                    choice_id="express_concern",
                    base_text="表達關心和擔憂",
                    choice_type="dialogue",
                    required_characters=["any_npc"],
                    relationship_changes={"any": 2},
                    success_chance=0.85,
                ),
            ],
            "exploration": [
                ContextualChoice(
                    choice_id="examine_surroundings",
                    base_text="仔細觀察周圍環境",
                    choice_type="exploration",
                    context_sensitive=True,
                    text_variations={
                        "mysterious": "警覺地檢查四周",
                        "peaceful": "悠閒地欣賞景色",
                        "dangerous": "謹慎地掃視環境",
                    },
                    success_chance=0.95,
                ),
                ContextualChoice(
                    choice_id="search_for_clues",
                    base_text="尋找線索或有用的物品",
                    choice_type="exploration",
                    stat_changes={"intelligence": 1},
                    success_chance=0.7,
                ),
                ContextualChoice(
                    choice_id="investigate_details",
                    base_text="深入調查可疑的細節",
                    choice_type="exploration",
                    stat_changes={"intelligence": 1},
                    success_chance=0.6,
                ),
            ],
            "action": [
                ContextualChoice(
                    choice_id="help_character",
                    base_text="主動提供幫助",
                    choice_type="action",
                    required_characters=["any_npc"],
                    relationship_changes={"any": 2},
                    stat_changes={"charisma": 1},
                    success_chance=0.8,
                ),
                ContextualChoice(
                    choice_id="wait_and_observe",
                    base_text="等待並觀察情況發展",
                    choice_type="action",
                    success_chance=1.0,
                ),
                ContextualChoice(
                    choice_id="take_initiative",
                    base_text="主動採取行動",
                    choice_type="action",
                    stat_changes={"charisma": 1},
                    dramatic_weight=3,
                    success_chance=0.75,
                ),
            ],
        }

    def _serialize_context_memory(
        self, context_memory: StoryContextMemory
    ) -> Dict[str, Any]:
        """Serialize context memory to JSON-compatible dict"""
        return {
            "session_id": context_memory.session_id,
            "player_name": context_memory.player_name,
            "characters": {
                k: v.to_dict() for k, v in context_memory.characters.items()
            },
            "scenes": {k: v.to_dict() for k, v in context_memory.scenes.items()},
            "scene_sequence": context_memory.scene_sequence,
            "current_scene_id": context_memory.current_scene_id,
            "world_flags": context_memory.world_flags,
            "location_states": context_memory.location_states,
            "main_plot_points": context_memory.main_plot_points,
            "player_decisions": context_memory.player_decisions,
            "player_relationships": context_memory.player_relationships,
        }

    def _deserialize_context_memory(self, data: Dict[str, Any]) -> StoryContextMemory:
        """Deserialize context memory from JSON data"""
        context_memory = StoryContextMemory(session_id=data["session_id"])

        # Restore characters
        for char_id, char_data in data.get("characters", {}).items():
            character = GameCharacter(
                character_id=char_data["character_id"],
                name=char_data["name"],
                role=CharacterRole(char_data["role"]),
                personality_traits=char_data["personality_traits"],
                speaking_style=char_data["speaking_style"],
                background_story=char_data["background_story"],
                motivations=char_data["motivations"],
                relationships=char_data["relationships"],
                current_state=CharacterState(char_data["current_state"]),
                current_location=char_data["current_location"],
                health=char_data["health"],
                mood=char_data["mood"],
                dialogue_history=char_data.get("dialogue_history", []),
                interaction_count=char_data.get("interaction_count", 0),
                last_seen_turn=char_data.get("last_seen_turn", 0),
            )
            context_memory.add_character(character)

        # Restore scenes
        for scene_id, scene_data in data.get("scenes", {}).items():
            scene = SceneContext(
                scene_id=scene_data["scene_id"],
                scene_type=SceneType(scene_data["scene_type"]),
                title=scene_data["title"],
                description=scene_data["description"],
                location=scene_data["location"],
                time_of_day=scene_data["time_of_day"],
                weather=scene_data["weather"],
                atmosphere=SceneMood(scene_data["atmosphere"]),
                present_characters=scene_data["present_characters"],
                primary_npc=scene_data.get("primary_npc"),
                plot_points=scene_data.get("plot_points", []),
                active_conflicts=scene_data.get("active_conflicts", []),
                scene_objectives=scene_data.get("scene_objectives", []),
                available_actions=scene_data.get("available_actions", []),
                turn_number=scene_data.get("turn_number", 0),
            )
            context_memory.add_scene(scene)

        # Restore other data
        context_memory.scene_sequence = data.get("scene_sequence", [])
        context_memory.current_scene_id = data.get("current_scene_id", "")
        context_memory.world_flags = data.get("world_flags", {})
        context_memory.location_states = data.get("location_states", {})
        context_memory.main_plot_points = data.get("main_plot_points", [])
        context_memory.player_decisions = data.get("player_decisions", [])
        context_memory.player_relationships = data.get("player_relationships", {})

        return context_memory

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

        # Initialize enhanced context if enabled
        if self.enhanced_mode:
            self._initialize_enhanced_context(session, persona, setting)

        self._save_session(session)
        logger.info(f"Created new session: {session.session_id}")
        return session

    def _initialize_enhanced_context(
        self, session: GameSession, persona: GamePersona, setting: str
    ):
        """Initialize enhanced context memory for new session"""
        context_memory = StoryContextMemory(
            session.session_id,
        )

        # Create player character
        player_character = GameCharacter(
            character_id="player",
            name=session.player_name,
            role=CharacterRole.PLAYER,
            personality_traits=["勇敢", "好奇", "善良"],
            speaking_style="直接而友善",
            background_story=f"一個開始新冒險的{session.player_name}",
            relationships={},
            motivations=["探索世界", "幫助他人", "成長"],
            current_location="起點",
        )
        context_memory.add_character(player_character)

        # Create narrator character based on persona
        narrator_character = GameCharacter(
            character_id="narrator",
            name=persona.name,
            role=CharacterRole.NARRATOR,
            personality_traits=persona.personality_traits,
            speaking_style=persona.speaking_style,
            background_story=persona.backstory,
            relationships={},
            motivations=persona.goals,
            current_location="起點",
            persona_prompt=self.persona_manager.generate_persona_prompt(
                persona, session.current_state.story_context
            ),
        )
        context_memory.add_character(narrator_character)

        # Create opening scene
        opening_scene = SceneContext(
            scene_id="opening",
            scene_type=SceneType.OPENING,
            title="冒險的開始",
            description="故事即將開始的地方",
            location="起點",
            time_of_day="清晨",
            weather="晴朗",
            atmosphere=SceneMood.MYSTERIOUS,
            present_characters=["player", "narrator"],
            primary_npc="narrator",
            scene_objectives=["設定背景", "介紹世界", "提供初始選擇"],
            turn_number=0,
        )
        context_memory.add_scene(opening_scene)
        context_memory.current_scene_id = "opening"

        self.context_memories[session.session_id] = context_memory

    def get_session(self, session_id: str) -> GameSession:
        """Get game session by ID"""
        session = self.sessions.get(session_id)
        if not session:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        return session

    def _load_sessions(self):
        """Load existing game sessions from storage"""
        try:
            for session_file in self.sessions_path.glob("*.json"):
                with open(session_file, "r", encoding="utf-8") as f:
                    session_data = json.load(f)
                    session = GameSession.from_dict(session_data)
                    self.sessions[session.session_id] = session

                    # Load enhanced context if available
                    if self.enhanced_mode:
                        context_file = session_file.with_suffix(".context.json")
                        if context_file.exists():
                            try:
                                with open(context_file, "r", encoding="utf-8") as cf:
                                    context_data = json.load(cf)
                                    context_memory = self._deserialize_context_memory(
                                        context_data
                                    )
                                    self.context_memories[session.session_id] = (
                                        context_memory
                                    )
                            except Exception as e:
                                logger.warning(
                                    f"Failed to load context for {session.session_id}: {e}"
                                )

            logger.info(f"Loaded {len(self.sessions)} existing sessions")
        except Exception as e:
            logger.error(f"Failed to load sessions: {e}")

    def _save_session(self, session: GameSession):
        """Save session to storage"""
        try:
            session_file = self.sessions_path / f"{session.session_id}.json"
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)

            # Save enhanced context if available
            if self.enhanced_mode and session.session_id in self.context_memories:
                context_file = session_file.with_suffix(".context.json")
                context_data = self._serialize_context_memory(
                    self.context_memories[session.session_id]
                )
                with open(context_file, "w", encoding="utf-8") as f:
                    json.dump(context_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")

    def make_choice(self, session_id: str, choice_id: str) -> Dict[str, Any]:
        """Make a choice in the story (完整實作版本)"""

        if session_id not in self.sessions:
            raise SessionNotFoundError(f"Session not found: {session_id}")

        session = self.sessions[session_id]

        # Generate current available choices
        if self.enhanced_mode and session_id in self.context_memories:
            # Enhanced mode: use contextual choices
            context_memory = self.context_memories[session_id]
            available_choices = self._generate_contextual_choices(context_memory)
        else:
            # Classic mode: use regular choices
            available_choices = self.choice_manager.generate_choices(
                session.current_state.story_context,
                session.stats.to_dict(),
                session.inventory,
                session.current_state.flags,
            )

        # Find the requested choice
        selected_choice = None
        for choice in available_choices:
            if choice.choice_id == choice_id:
                selected_choice = choice
                break

        if not selected_choice:
            raise InvalidChoiceError(f"Choice not available: {choice_id}")

        # Check if choice can be made
        if hasattr(selected_choice, "can_execute"):
            # Enhanced contextual choice
            context_memory = self.context_memories[session_id]
            can_execute, reason = selected_choice.can_execute(  # type: ignore
                context_memory, session.stats.to_dict(), session.inventory
            )
        else:
            # Regular choice
            can_execute, reason = selected_choice.can_choose(  # type: ignore
                session.stats.to_dict(), session.inventory, session.current_state.flags
            )

        if not can_execute:
            raise InvalidChoiceError(f"Cannot make choice: {reason}")

        # Execute the choice
        if hasattr(selected_choice, "execute") and hasattr(
            selected_choice, "relationship_changes"
        ):
            # Enhanced contextual choice execution
            context_memory = self.context_memories[session_id]
            success, consequences = selected_choice.execute(  # type: ignore
                context_memory, session.stats.to_dict()  # type: ignore
            )
        else:
            # Regular choice execution
            success, consequences = selected_choice.execute(  # type: ignore
                session.stats.to_dict(), session.stats.luck
            )

        # Apply consequences to session
        result = {"success": success, "consequences": {}}

        # Apply stat changes
        if "stats" in consequences:
            for stat, change in consequences["stats"].items():
                if hasattr(session.stats, stat):
                    current_value = getattr(session.stats, stat)
                    new_value = max(0, current_value + change)
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

        # Apply relationship changes (enhanced mode)
        if (
            "relationships" in consequences
            and self.enhanced_mode
            and session_id in self.context_memories
        ):
            context_memory = self.context_memories[session_id]
            for char_id, change in consequences["relationships"].items():
                context_memory.update_character_relationship("player", char_id, change)
            result["consequences"]["relationships"] = consequences["relationships"]

        # Update session
        session.turn_count += 1
        session.updated_at = datetime.now()

        # Save session
        self._save_session(session)

        return result

    async def process_turn(
        self, session_id: str, player_input: str, choice_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a player turn with enhanced or classic mode"""

        if self.enhanced_mode and session_id in self.context_memories:
            return await self._process_enhanced_turn(
                session_id, player_input, choice_id
            )
        else:
            return await self._process_classic_turn(session_id, player_input, choice_id)

    async def _process_enhanced_turn(
        self, session_id: str, player_input: str, choice_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process turn with enhanced context management"""

        session = self.get_session(session_id)
        context_memory = self.context_memories[session_id]
        persona = self.persona_manager.get_persona(session.persona_id)

        if not persona:
            raise GameError(f"Persona not found: {session.persona_id}")

        try:
            # Execute choice if provided
            choice_result = None
            if choice_id:
                choice_result = await self._execute_contextual_choice(
                    context_memory, choice_id, player_input, session
                )

            # Generate narrative with enhanced context
            if isinstance(self.narrative_generator, EnhancedNarrativeGenerator):
                narrative_result = (
                    await self.narrative_generator.generate_contextual_narrative(
                        context_memory, player_input, choice_result
                    )
                )
            else:
                # Fallback to classic generation
                story_context = self._build_story_context_from_memory(context_memory)
                persona_prompt = self.persona_manager.generate_persona_prompt(
                    persona, session.current_state.story_context
                )
                narrative_result = {
                    "main_narrative": await self.narrative_generator.generate_scene(
                        story_context, persona_prompt, player_input
                    ),
                    "character_dialogues": [],
                    "scene_changes": {},
                    "narrative_focus": "general_progression",
                }

            # Update context memory
            await self._update_context_memory(
                context_memory, player_input, narrative_result, choice_result
            )

            # Generate new contextual choices
            new_choices = self._generate_contextual_choices(context_memory)

            # Update session state
            session.current_state.scene_description = narrative_result["main_narrative"]
            session.current_state.available_choices = [
                {
                    "choice_id": choice.choice_id,
                    "text": choice.get_display_text(context_memory),
                    "type": choice.choice_type,
                    "difficulty": choice.difficulty,
                }
                for choice in new_choices
            ]

            # Add to history
            session.add_to_history(
                player_input, narrative_result["main_narrative"], choice_id
            )

            # Save session
            self._save_session(session)

            return {
                "session_id": session.session_id,
                "turn_count": session.turn_count,
                "narrative": narrative_result["main_narrative"],
                "character_dialogues": narrative_result.get("character_dialogues", []),
                "choices": session.current_state.available_choices,
                "stats": session.stats.to_dict(),
                "inventory": session.inventory,
                "scene_info": {
                    "current_scene": (
                        context_memory.get_current_scene().to_dict()  # type: ignore
                        if context_memory.get_current_scene()
                        else {}
                    ),
                    "present_characters": [
                        char.name for char in context_memory.get_characters_in_scene()
                    ],
                    "mood_shift": narrative_result.get("mood_shift"),
                    "scene_changes": narrative_result.get("scene_changes", {}),
                },
                "context_summary": {
                    "total_scenes": len(context_memory.scenes),
                    "total_characters": len(context_memory.characters),
                    "active_plot_points": len(context_memory.main_plot_points),
                },
            }

        except Exception as e:
            logger.error(
                f"Failed to process enhanced turn for session {session_id}: {e}"
            )
            # Fallback to classic processing
            return await self._process_classic_turn(session_id, player_input, choice_id)

    async def _process_classic_turn(
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
                current_scene=session.current_state.scene_id,  # type: ignore
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
            narrative_response = await self.narrative_generator.generate_scene(  # type: ignore
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

    def _build_story_context_from_memory(
        self, context_memory: StoryContextMemory
    ) -> StoryContext:
        """Convert enhanced context memory to classic StoryContext"""
        current_scene = context_memory.get_current_scene()

        return StoryContext(
            current_scene=current_scene.scene_id if current_scene else "unknown",
            previous_scenes=[
                scene_id for scene_id in context_memory.scene_sequence[-3:]
            ],
            active_characters=[
                char.name for char in context_memory.get_characters_in_scene()
            ],
            current_location=current_scene.location if current_scene else "unknown",
            time_of_day=current_scene.time_of_day if current_scene else "unknown",
            weather=current_scene.weather if current_scene else "unknown",
            mood=current_scene.atmosphere if current_scene else "neutral",  # type: ignore
            plot_points=context_memory.main_plot_points[-3:],
            world_state=context_memory.world_flags,
        )

    async def _execute_contextual_choice(
        self,
        context_memory: StoryContextMemory,
        choice_id: str,
        player_input: str,
        session: GameSession,
    ) -> Dict[str, Any]:
        """Execute a contextual choice and apply consequences"""

        # Find the choice across all categories
        selected_choice = None
        for category_choices in self.contextual_choices.values():
            for choice in category_choices:
                if choice.choice_id == choice_id:
                    selected_choice = choice
                    break
            if selected_choice:
                break

        if not selected_choice:
            raise GameError(f"Choice not found: {choice_id}")

        # Check if choice can be executed
        can_execute, reason = selected_choice.can_execute(
            context_memory, session.stats.to_dict(), session.inventory
        )

        if not can_execute:
            return {"success": False, "reason": reason}

        # Apply consequences
        result = {"success": True, "consequences": {}}

        # Apply relationship changes
        for char_id, change in selected_choice.relationship_changes.items():
            if char_id == "any":
                # Apply to all present NPCs
                for char in context_memory.get_characters_in_scene():
                    if char.role != CharacterRole.PLAYER:
                        current_score = context_memory.player_relationships.get(
                            char.character_id, 0
                        )
                        context_memory.player_relationships[char.character_id] = max(
                            -10, min(10, current_score + change)
                        )
            else:
                current_score = context_memory.player_relationships.get(char_id, 0)
                context_memory.player_relationships[char_id] = max(
                    -10, min(10, current_score + change)
                )

        # Apply stat changes
        for stat, change in selected_choice.stat_changes.items():
            if hasattr(session.stats, stat):
                current_value = getattr(session.stats, stat)
                new_value = max(0, current_value + change)
                setattr(session.stats, stat, new_value)
                result["consequences"][stat] = change

        # Apply flag changes
        context_memory.world_flags.update(selected_choice.flag_changes)

        # Record the decision
        current_scene = context_memory.get_current_scene()
        turn_number = current_scene.turn_number if current_scene else 0

        context_memory.record_player_decision(
            player_input, choice_id, turn_number, result["consequences"]
        )

        result["choice_executed"] = selected_choice.base_text
        result["dramatic_weight"] = selected_choice.dramatic_weight

        return result

    async def _update_context_memory(
        self,
        context_memory: StoryContextMemory,
        player_input: str,
        narrative_result: Dict[str, Any],
        choice_result: Optional[Dict[str, Any]],
    ):
        """Update context memory with new information"""

        current_scene = context_memory.get_current_scene()
        if current_scene:
            current_scene.turn_number += 1

            # Update plot points if narrative suggests new developments
            narrative = narrative_result["main_narrative"].lower()

            # Simple keyword-based plot point detection
            plot_keywords = {
                "發現": "發現了重要線索",
                "秘密": "揭露了隱藏的秘密",
                "危險": "遇到了新的威脅",
                "盟友": "結識了潛在的盟友",
                "敵人": "識別出新的敵對勢力",
            }

            for keyword, plot_point in plot_keywords.items():
                if (
                    keyword in narrative
                    and plot_point not in context_memory.main_plot_points
                ):
                    context_memory.main_plot_points.append(plot_point)
                    current_scene.add_plot_point(plot_point)

        # Update character states based on interactions
        for dialogue in narrative_result.get("character_dialogues", []):
            char_id = dialogue["character_id"]
            character = context_memory.get_character(char_id)
            if character:
                # Update character emotional state based on interaction
                new_state = self._infer_character_state_from_dialogue(
                    dialogue["content"], dialogue.get("emotional_state", "neutral")
                )
                if new_state:
                    character.current_state = CharacterState(new_state)

    def validate_choice(
        self, choice_id: str, available_choices: List[GameChoice]
    ) -> Optional[GameChoice]:
        """Validate that a choice is available and can be executed"""
        for choice in available_choices:
            if choice.choice_id == choice_id:
                return choice
        return None

    def _infer_character_state_from_dialogue(
        self, dialogue: str, current_state: str
    ) -> Optional[str]:
        """Infer character emotional state from dialogue content"""

        dialogue_lower = dialogue.lower()

        state_indicators = {
            "happy": ["高興", "開心", "快樂", "滿意", "愉快", "歡喜", "喜悅"],
            "angry": ["生氣", "憤怒", "不滿", "惱火", "怒", "憤慨", "暴怒"],
            "fearful": ["害怕", "恐懼", "擔心", "緊張", "不安", "驚恐", "畏懼"],
            "excited": ["興奮", "激動", "期待", "熱情", "活力", "振奮", "狂熱"],
            "suspicious": ["懷疑", "不信", "質疑", "猜忌", "警惕", "疑惑", "不信任"],
            "sad": ["悲傷", "難過", "沮喪", "憂鬱", "失落", "哀傷", "痛苦"],
            "worried": ["擔憂", "焦慮", "煩惱", "憂慮", "不安", "顧慮", "掛心"],
            "confident": ["自信", "確信", "肯定", "堅定", "篤定", "信心", "把握"],
        }

        # Count matches for each emotion
        emotion_scores = {}
        for emotion, indicators in state_indicators.items():
            score = sum(1 for indicator in indicators if indicator in dialogue_lower)
            if score > 0:
                emotion_scores[emotion] = score

        # Return the emotion with highest score, or None if no clear emotion
        if emotion_scores:
            best_emotion = max(emotion_scores, key=emotion_scores.get)  # type: ignore
            # Only return if the score is significant compared to dialogue length
            if emotion_scores[best_emotion] >= 1:
                return best_emotion

        return None

    def _generate_contextual_choices(
        self, context_memory: StoryContextMemory
    ) -> List[ContextualChoice]:
        """Generate context-appropriate choices"""

        current_scene = context_memory.get_current_scene()
        present_characters = context_memory.get_characters_in_scene()

        available_choices = []

        # Always include basic exploration choices
        for choice in self.contextual_choices.get("exploration", []):
            available_choices.append(choice)

        # Add dialogue choices if NPCs are present
        if any(char.role != CharacterRole.PLAYER for char in present_characters):
            for choice in self.contextual_choices.get("dialogue", []):
                available_choices.append(choice)

        # Add action choices
        for choice in self.contextual_choices.get("action", []):
            available_choices.append(choice)

        # Filter choices based on context requirements
        valid_choices = []
        for choice in available_choices:
            can_execute, _ = choice.can_execute(context_memory, {}, [])
            if can_execute or len(valid_choices) < 2:  # Ensure minimum choices
                valid_choices.append(choice)

        return valid_choices[:4]  # Limit to 4 choices max

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

    def get_choice_preview(self, choice_id: str, session_id: str) -> Dict[str, Any]:
        """Get preview of choice consequences before execution"""

        session = self.get_session(session_id)

        # Generate current choices to find the requested one
        if self.enhanced_mode and session_id in self.context_memories:
            context_memory = self.context_memories[session_id]
            choices = self._generate_contextual_choices(context_memory)
        else:
            choices = self.choice_manager.generate_choices(
                session.current_state.story_context,
                session.stats.to_dict(),
                session.inventory,
                session.current_state.flags,
            )

        # Find the specific choice
        target_choice = None
        for choice in choices:
            if choice.choice_id == choice_id:
                target_choice = choice
                break

        if not target_choice:
            return {"error": "Choice not found"}

        # For enhanced choices, use enhanced preview
        if hasattr(target_choice, "get_display_text"):
            return {
                "choice_id": choice_id,
                "display_text": target_choice.get_display_text(
                    context_memory if self.enhanced_mode else {}  # type: ignore
                ),
                "type": target_choice.choice_type,
                "difficulty": target_choice.difficulty,
                "success_chance": target_choice.success_chance,
                "requirements": target_choice.requirements,  # type: ignore
                "consequences_preview": target_choice.consequences,  # type: ignore
                "description": getattr(target_choice, "description", None),
            }
        else:
            # Classic choice preview
            return self.choice_manager.get_choice_consequences_preview(
                target_choice, session.stats.to_dict()  # type: ignore
            )

    def get_character_relationship_info(
        self, session_id: str, character_id: str = None  # type: ignore
    ) -> Dict[str, Any]:
        """Get relationship information for characters in enhanced mode"""

        if not self.enhanced_mode or session_id not in self.context_memories:
            return {"error": "Enhanced mode not available or session not found"}

        context_memory = self.context_memories[session_id]

        if character_id:
            # Get info for specific character
            character = context_memory.get_character(character_id)
            if not character:
                return {"error": "Character not found"}

            return {
                "character_id": character_id,
                "name": character.name,
                "role": character.role.value,
                "current_state": character.current_state.value,
                "current_location": character.current_location,
                "relationship_score": context_memory.player_relationships.get(
                    character_id, 0
                ),
                "recent_interactions": (
                    character.get_recent_dialogue(3)
                    if hasattr(character, "get_recent_dialogue")
                    else []
                ),
                "interaction_count": getattr(character, "interaction_count", 0),
                "personality_traits": getattr(character, "personality_traits", []),
            }
        else:
            # Get info for all characters
            relationships = {}
            for char_id, character in context_memory.characters.items():
                if character.role != CharacterRole.PLAYER:
                    relationships[char_id] = {
                        "name": character.name,
                        "role": character.role.value,
                        "relationship_score": context_memory.player_relationships.get(
                            char_id, 0
                        ),
                        "current_state": character.current_state.value,
                        "interaction_count": getattr(character, "interaction_count", 0),
                    }

            return {
                "total_characters": len(relationships),
                "relationships": relationships,
                "average_relationship": (
                    sum(info["relationship_score"] for info in relationships.values())
                    / len(relationships)
                    if relationships
                    else 0
                ),
            }

    def export_session_data(
        self, session_id: str, include_context: bool = True
    ) -> Dict[str, Any]:
        """Export complete session data for backup or analysis"""

        if session_id not in self.sessions:
            return {"error": "Session not found"}

        session = self.sessions[session_id]
        export_data = {
            "session_data": session.to_dict(),
            "export_timestamp": datetime.now().isoformat(),
            "enhanced_mode": self.enhanced_mode,
        }

        # Include enhanced context if available
        if (
            self.enhanced_mode
            and session_id in self.context_memories
            and include_context
        ):
            context_memory = self.context_memories[session_id]
            export_data["enhanced_context"] = self._serialize_context_memory(
                context_memory
            )

        return export_data

    def import_session_data(self, session_data: Dict[str, Any]) -> bool:
        """Import session data from backup"""

        try:
            # Restore basic session
            session = GameSession.from_dict(session_data["session_data"])
            self.sessions[session.session_id] = session

            # Restore enhanced context if available
            if (
                session_data.get("enhanced_mode")
                and "enhanced_context" in session_data
                and self.enhanced_mode
            ):

                context_memory = self._deserialize_context_memory(
                    session_data["enhanced_context"]
                )
                self.context_memories[session.session_id] = context_memory

            # Save restored session
            self._save_session(session)

            logger.info(f"Imported session: {session.session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to import session data: {e}")
            return False

    def get_system_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""

        total_sessions = len(self.sessions)
        active_sessions = sum(
            1 for session in self.sessions.values() if session.is_active
        )
        total_turns = sum(session.turn_count for session in self.sessions.values())

        metrics = {
            "session_metrics": {
                "total_sessions": total_sessions,
                "active_sessions": active_sessions,
                "inactive_sessions": total_sessions - active_sessions,
                "total_turns_processed": total_turns,
                "average_turns_per_session": (
                    total_turns / total_sessions if total_sessions > 0 else 0
                ),
            },
            "system_metrics": {
                "enhanced_mode_enabled": self.enhanced_mode,
                "available_personas": len(self.persona_manager.list_personas()),
                "choice_templates": (
                    len(self.choice_manager.choice_templates)
                    if hasattr(self.choice_manager, "choice_templates")
                    else 0
                ),
            },
        }

        # Enhanced mode specific metrics
        if self.enhanced_mode:
            total_characters = sum(
                len(context.characters) for context in self.context_memories.values()
            )
            total_scenes = sum(
                len(context.scenes) for context in self.context_memories.values()
            )

            metrics["enhanced_metrics"] = {
                "context_memories_active": len(self.context_memories),
                "total_characters_tracked": total_characters,
                "total_scenes_created": total_scenes,
                "average_characters_per_session": (
                    total_characters / len(self.context_memories)
                    if self.context_memories
                    else 0
                ),
            }

        return metrics

    def list_sessions(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """List game sessions"""
        sessions = []
        for session in self.sessions.values():
            if active_only and not session.is_active:
                continue

            session_info = {
                "session_id": session.session_id,
                "player_name": session.player_name,
                "persona_id": session.persona_id,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "turn_count": session.turn_count,
                "is_active": session.is_active,
            }

            # Add enhanced info if available
            if self.enhanced_mode and session.session_id in self.context_memories:
                context_memory = self.context_memories[session.session_id]
                session_info.update(
                    {
                        "enhanced_mode": True,
                        "total_characters": len(context_memory.characters),
                        "total_scenes": len(context_memory.scenes),
                        "current_scene": context_memory.current_scene_id,
                        "plot_points": len(context_memory.main_plot_points),
                    }
                )
            else:
                session_info["enhanced_mode"] = False

            sessions.append(session_info)

        return sessions

    def get_session_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get enhanced context information for a session"""
        if not self.enhanced_mode or session_id not in self.context_memories:
            return None

        context_memory = self.context_memories[session_id]
        current_scene = context_memory.get_current_scene()

        return {
            "session_id": context_memory.session_id,
            "player_name": context_memory.player_name,
            "current_scene": current_scene.to_dict() if current_scene else None,
            "present_characters": [
                {
                    "character_id": char.character_id,
                    "name": char.name,
                    "role": char.role.value,
                    "current_state": char.current_state.value,
                    "relationship_score": context_memory.player_relationships.get(
                        char.character_id, 0
                    ),
                }
                for char in context_memory.get_characters_in_scene()
            ],
            "world_flags": context_memory.world_flags,
            "main_plot_points": context_memory.main_plot_points,
            "recent_decisions": context_memory.player_decisions[
                -5:
            ],  # Last 5 decisions
            "total_scenes": len(context_memory.scenes),
            "total_characters": len(context_memory.characters),
        }

    def update_character_state(
        self, session_id: str, character_id: str, updates: Dict[str, Any]
    ) -> bool:
        """Update character state in enhanced context"""
        if not self.enhanced_mode or session_id not in self.context_memories:
            return False

        context_memory = self.context_memories[session_id]
        character = context_memory.get_character(character_id)

        if not character:
            return False

        # Update allowed character properties
        if "current_state" in updates:
            try:
                character.current_state = CharacterState(updates["current_state"])
            except ValueError:
                logger.warning(f"Invalid character state: {updates['current_state']}")

        if "current_location" in updates:
            character.current_location = updates["current_location"]

        if "health" in updates:
            character.health = max(0, min(100, updates["health"]))

        if "mood" in updates:
            character.mood = updates["mood"]

        # Save the updated context
        session = self.get_session(session_id)
        self._save_session(session)

        return True

    def delete_session(self, session_id: str) -> bool:
        """Delete a game session"""
        if session_id not in self.sessions:
            return False

        try:
            # Remove from memory
            del self.sessions[session_id]

            if self.enhanced_mode and session_id in self.context_memories:
                del self.context_memories[session_id]

            # Remove files
            session_file = self.sessions_path / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()

            context_file = self.sessions_path / f"{session_id}.context.json"
            if context_file.exists():
                context_file.unlink()

            logger.info(f"Deleted session: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False

    def end_session(self, session_id: str) -> Dict[str, Any]:
        """End a game session and provide summary"""
        if session_id not in self.sessions:
            return {"error": "Session not found"}

        session = self.sessions[session_id]
        session.is_active = False
        session.updated_at = datetime.now()

        # Generate session summary
        summary = {
            "session_id": session_id,
            "player_name": session.player_name,
            "total_turns": session.turn_count,
            "duration": str(session.updated_at - session.created_at),
            "final_stats": session.stats.to_dict(),
            "inventory": session.inventory,
            "major_choices": len([h for h in session.history if h.get("choice_id")]),
        }

        # Enhanced summary if available
        if self.enhanced_mode and session_id in self.context_memories:
            context_memory = self.context_memories[session_id]
            summary.update(
                {
                    "scenes_visited": len(context_memory.scenes),
                    "characters_met": len(context_memory.characters),
                    "plot_points_completed": len(context_memory.main_plot_points),
                    "relationship_scores": context_memory.player_relationships,
                }
            )

        self._save_session(session)
        logger.info(f"Session {session_id} ended")

        return summary

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of a game session"""
        if session_id not in self.sessions:
            return {"error": "Session not found"}

        session = self.sessions[session_id]

        summary = {
            "session_id": session_id,
            "player_name": session.player_name,
            "persona_id": session.persona_id,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "turn_count": session.turn_count,
            "is_active": session.is_active,
            "current_scene": session.current_state.scene_id,
            "stats": session.stats.to_dict(),
            "inventory_count": len(session.inventory),
        }

        return summary

    def list_personas(self) -> List[Dict[str, Any]]:
        """List available personas"""
        personas = []
        for persona in self.persona_manager.list_personas():
            personas.append(
                {
                    "persona_id": persona.persona_id,
                    "name": persona.name,
                    "description": persona.description,
                    "type": (
                        persona.persona_type.value
                        if hasattr(persona.persona_type, "value")
                        else str(persona.persona_type)
                    ),
                    "personality_traits": persona.personality_traits[
                        :3
                    ],  # Show first 3 traits
                }
            )
        return personas


# =============================================================================
# Factory Functions
# =============================================================================


def get_story_engine(
    config_dir: Optional[Path] = None, enhanced_mode: bool = True
) -> StoryEngine:
    """Factory function to get story engine instance"""
    return StoryEngine(config_dir=config_dir, enhanced_mode=enhanced_mode)


def create_story_engine(
    config_dir: Optional[Path] = None, enhanced_mode: bool = True
) -> StoryEngine:
    """Create new story engine instance"""
    return StoryEngine(config_dir=config_dir, enhanced_mode=enhanced_mode)


# =============================================================================
# Global engine instance for singleton pattern
# =============================================================================

_story_engine: Optional[StoryEngine] = None


def get_global_story_engine() -> StoryEngine:
    """Get global story engine instance"""
    global _story_engine
    if _story_engine is None:
        config = get_config()
        config_dir = Path(config.config_dir)
        _story_engine = StoryEngine(config_dir=config_dir, enhanced_mode=True)
    return _story_engine


def reset_global_story_engine():
    """Reset global story engine instance"""
    global _story_engine
    _story_engine = None
