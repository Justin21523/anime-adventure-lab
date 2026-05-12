# =============================================================================
# core/story/engine.py
"""
Main Story Engine
Coordinates all story components and manages game flow
"""

import json
import logging
import asyncio
import random
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
from .memory_manager import get_memory_manager, StoryMemoryManager
from ..exceptions import GameError, SessionNotFoundError, InvalidChoiceError
from ..config import get_config
from ..shared_cache import get_shared_cache

logger = logging.getLogger(__name__)

# Agent integration (optional, controlled by config)
try:
    from ..agents.story_agent_layer import get_agent_layer
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False
    logger.warning("Agent layer not available")

# P1 subsystems (lazy import to avoid torch issues)
try:
    from .combat import CombatSystem, CombatSession, Combatant, CombatAction, CombatResult
    COMBAT_AVAILABLE = True
except ImportError:
    COMBAT_AVAILABLE = False
    CombatSystem = None

try:
    from .achievement import AchievementManager
    ACHIEVEMENT_AVAILABLE = True
except ImportError:
    ACHIEVEMENT_AVAILABLE = False
    AchievementManager = None

try:
    from .economy import EconomyManager
    ECONOMY_AVAILABLE = True
except ImportError:
    ECONOMY_AVAILABLE = False
    EconomyManager = None

try:
    from .relationship import RelationshipManager
    RELATIONSHIP_AVAILABLE = True
except ImportError:
    RELATIONSHIP_AVAILABLE = False
    RelationshipManager = None

try:
    from .map_exploration import MapSystem, Location
    MAP_AVAILABLE = True
except ImportError:
    MAP_AVAILABLE = False
    MapSystem = None


class StoryEngine:
    """Main story engine coordinating all game systems"""

    def __init__(self, config_dir: Optional[Path] = None, enhanced_mode: bool = True, agent_enabled: bool = True):
        self.config = get_config()
        self.cache = get_shared_cache()
        self.enhanced_mode = enhanced_mode
        self.agent_enabled = agent_enabled and AGENT_AVAILABLE

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

        # P1 subsystems
        self.combat_system: Optional[CombatSystem] = None
        self.achievement_manager: Optional[AchievementManager] = None
        self.economy_manager: Optional[EconomyManager] = None
        self.relationship_manager: Optional[RelationshipManager] = None
        self.map_system: Optional[MapSystem] = None

        if COMBAT_AVAILABLE:
            self.combat_system = CombatSystem()
            logger.info("Combat system initialized")
        if ACHIEVEMENT_AVAILABLE:
            self.achievement_manager = AchievementManager()
            self.achievement_manager.register_default_achievements()
            logger.info("Achievement system initialized")
        if ECONOMY_AVAILABLE:
            self.economy_manager = EconomyManager()
            logger.info("Economy system initialized")
        if RELATIONSHIP_AVAILABLE:
            self.relationship_manager = RelationshipManager()
            logger.info("Relationship system initialized")
        if MAP_AVAILABLE:
            self.map_system = MapSystem()
            logger.info("Map system initialized")

        # Session storage
        self.sessions: Dict[str, GameSession] = {}
        try:
            cache_root = self.cache.get_output_path("games")  # type: ignore[assignment]
            self.sessions_path = Path(cache_root) / "story_sessions"
            self.sessions_path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Falling back to local story session path: %s", exc)
            fallback_path = Path("outputs") / "story_sessions"
            fallback_path.mkdir(parents=True, exist_ok=True)
            self.sessions_path = fallback_path

        # Enhanced features
        if self.enhanced_mode:
            self.context_memories: Dict[str, StoryContextMemory] = {}
            self.contextual_choices = self._load_contextual_choices()
            # Context compression: summarize old turns to reduce token burn
            self.context_compressors: Dict[str, "ContextCompressor"] = {}

        # Load existing sessions
        self._load_sessions()

        logger.info("Story engine initialized")

    def _apply_runtime_preset_llm(self, session: GameSession) -> None:
        """
        Best-effort: configure the global LLM adapter using session.runtime_preset_id.

        Goal:
        - RTX 5080 16GB preset should actually drive Qwen 7B 4bit, dtype, device_map, etc.
        - Keep the global adapter object stable (in-place reconfigure) so other subsystems
          holding a reference don't go stale.
        """
        try:
            story_ctx = getattr(getattr(session, "current_state", None), "story_context", {}) or {}
            if not isinstance(story_ctx, dict):
                return
            preset_id = str(story_ctx.get("runtime_preset_id") or "").strip()
            if not preset_id:
                return

            from core.runtime.catalog import get_runtime_preset

            preset = get_runtime_preset(preset_id) or {}
            llm_cfg = preset.get("llm") if isinstance(preset.get("llm"), dict) else {}
            model_name = str(llm_cfg.get("model_name") or "").strip()
            if not model_name:
                return

            llm_kwargs: Dict[str, Any] = {}
            for key in ["device_map", "torch_dtype", "use_quantization", "quantization_bits"]:
                if key in llm_cfg and llm_cfg.get(key) is not None:
                    llm_kwargs[key] = llm_cfg.get(key)

            from core.llm.adapter import get_llm_adapter

            adapter = get_llm_adapter(model_name=model_name, **llm_kwargs)
            try:
                setattr(self.narrative_generator, "llm", adapter)
            except Exception:
                pass
        except Exception as exc:  # noqa: BLE001
            logger.debug("Runtime preset LLM apply skipped: %s", exc)

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
            "player_choices_history": getattr(context_memory, 'player_choices_history', []),
            "player_personality_profile": getattr(context_memory, 'player_personality_profile', {}),
            "character_relationships": getattr(context_memory, 'character_relationships', {}),
            "global_variables": getattr(context_memory, 'global_variables', {}),
            "timeline": getattr(context_memory, 'timeline', []),
            "subplot_threads": getattr(context_memory, 'subplot_threads', {}),
            "completed_objectives": getattr(context_memory, 'completed_objectives', []),
            "turn_history": getattr(context_memory, 'turn_history', []),
            "narrative_memory": getattr(context_memory, 'narrative_memory', []),
            "scene_history": [s.to_dict() for s in context_memory.scene_history] if context_memory.scene_history else [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

    def _deserialize_context_memory(self, data: Dict[str, Any]) -> StoryContextMemory:
        """Deserialize context memory from JSON data"""
        if "session_id" not in data:
            raise ValueError("Context memory must contain 'session_id'")

        context_memory = StoryContextMemory(session_id=data["session_id"])
        # Store created_at/updated_at for audit (optional, backward compat)
        _ca = data.get("created_at") or ""
        _ua = data.get("updated_at") or data.get("created_at") or ""
        try:
            if _ca:
                context_memory.created_at = datetime.fromisoformat(_ca)
            else:
                context_memory.created_at = datetime.now()
        except (ValueError, TypeError):
            context_memory.created_at = datetime.now()
        try:
            if _ua:
                context_memory.updated_at = datetime.fromisoformat(_ua)
            else:
                context_memory.updated_at = context_memory.created_at
        except (ValueError, TypeError):
            context_memory.updated_at = context_memory.created_at

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
        context_memory.player_name = data.get("player_name", "")
        context_memory.scene_sequence = data.get("scene_sequence", [])
        context_memory.current_scene_id = data.get("current_scene_id", "")
        context_memory.world_flags = data.get("world_flags", {})
        context_memory.location_states = data.get("location_states", {})
        context_memory.main_plot_points = data.get("main_plot_points", [])
        context_memory.player_decisions = data.get("player_decisions", [])
        context_memory.player_relationships = data.get("player_relationships", {})
        # Restore additional fields (may be missing in old context files — backward compat)
        context_memory.player_choices_history = data.get("player_choices_history", [])
        context_memory.player_personality_profile = data.get("player_personality_profile", {})
        context_memory.character_relationships = data.get("character_relationships", {})
        context_memory.global_variables = data.get("global_variables", {})
        context_memory.timeline = data.get("timeline", [])
        context_memory.subplot_threads = data.get("subplot_threads", {})
        context_memory.completed_objectives = data.get("completed_objectives", [])
        context_memory.turn_history = data.get("turn_history", [])
        context_memory.narrative_memory = data.get("narrative_memory", [])
        # Restore scene_history
        for sh_data in data.get("scene_history", []):
            try:
                sh = SceneContext(
                    scene_id=sh_data["scene_id"],
                    scene_type=SceneType(sh_data["scene_type"]),
                    title=sh_data["title"],
                    description=sh_data["description"],
                    location=sh_data["location"],
                    time_of_day=sh_data["time_of_day"],
                    weather=sh_data["weather"],
                    atmosphere=SceneMood(sh_data["atmosphere"]),
                    present_characters=sh_data["present_characters"],
                    primary_npc=sh_data.get("primary_npc"),
                    plot_points=sh_data.get("plot_points", []),
                    active_conflicts=sh_data.get("active_conflicts", []),
                    scene_objectives=sh_data.get("scene_objectives", []),
                    available_actions=sh_data.get("available_actions", []),
                    turn_number=sh_data.get("turn_number", 0),
                )
                context_memory.scene_history.append(sh)
            except Exception:
                pass
        # Restore current_scene from scene_history if available
        if context_memory.scene_history:
            context_memory.current_scene = context_memory.scene_history[-1]

        return context_memory

    def create_session(
        self,
        player_name: str,
        persona_id: str = "wise_sage",
        setting: str = "fantasy",
        difficulty: str = "medium",
        world_id: str = "default",
        player_template_id: Optional[str] = None,
        runtime_preset_id: Optional[str] = None,
    ) -> GameSession:
        """Create new game session"""

        # Validate persona
        persona = self.persona_manager.get_persona(persona_id)
        if not persona:
            raise GameError(f"Unknown persona: {persona_id}")

        # Resolve world_id best-effort (unknown world -> fallback to default)
        try:
            from core.worldpacks import get_worldpack_manager

            wpm = get_worldpack_manager()
            if wpm.get_worldpack(world_id) is None:
                world_id = "default"
        except Exception as exc:  # noqa: BLE001
            logger.debug("Worldpack resolution skipped: %s", exc)

        # Create initial game state
        effective_runtime_preset_id: Optional[str] = None
        try:
            cleaned = str(runtime_preset_id or "").strip()
            effective_runtime_preset_id = cleaned or None
        except Exception:
            effective_runtime_preset_id = None

        if effective_runtime_preset_id is None:
            try:
                from core.worldpacks import get_worldpack_manager

                wpm = get_worldpack_manager()
                worldpack = wpm.get_worldpack(world_id)
                candidate = getattr(worldpack, "runtime_preset_id", None) if worldpack else None
                candidate = str(candidate or "").strip()
                effective_runtime_preset_id = candidate or None
            except Exception:
                effective_runtime_preset_id = None

        if effective_runtime_preset_id is None:
            try:
                from core.runtime.catalog import load_runtime_preset_catalog

                catalog = load_runtime_preset_catalog()
                candidate = str(catalog.get("default_preset_id") or "").strip()
                effective_runtime_preset_id = candidate or None
            except Exception:
                effective_runtime_preset_id = None

        initial_state = GameState(
            scene_id="opening",
            scene_description="",
            available_choices=[],
            story_context={
                "world_id": world_id,
                "runtime_preset_id": effective_runtime_preset_id,
                "setting": setting,
                "difficulty": difficulty,
                "player_template_id": player_template_id,
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
            world_id=world_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            turn_count=0,
            stats=PlayerStats(),
            inventory=[],
            current_state=initial_state,
        )

        # Store session
        self.sessions[session.session_id] = session

        # Apply character class if specified (P2.5)
        if player_template_id:
            try:
                from core.story.character_classes import apply_class_to_session, get_class
                cls = apply_class_to_session(session, player_template_id)
                if cls:
                    logger.info(f"Applied class: {cls.name} to {session.session_id}")
                else:
                    logger.warning(f"Unknown class: {player_template_id}, skipping")
            except Exception as _exc:
                logger.debug(f"Class application failed: {_exc}")

        # Initialize enhanced context if enabled
        if self.enhanced_mode:
            self._initialize_enhanced_context(
                session,
                persona,
                setting,
                player_template_id=player_template_id,
            )

        self._save_session(session)
        logger.info(f"Created new session: {session.session_id}")
        return session

    def _initialize_enhanced_context(
        self,
        session: GameSession,
        persona: GamePersona,
        setting: str,
        player_template_id: Optional[str] = None,
    ):
        """Initialize enhanced context memory for new session"""
        context_memory = StoryContextMemory(
            session.session_id,
        )
        context_memory.player_name = session.player_name  # type: ignore[attr-defined]
        context_memory.scene_sequence = []  # type: ignore[attr-defined]
        context_memory.location_states = {}  # type: ignore[attr-defined]
        context_memory.main_plot_points = []  # type: ignore[attr-defined]
        context_memory.player_decisions = []  # type: ignore[attr-defined]
        context_memory.player_relationships = {}  # type: ignore[attr-defined]
        context_memory.world_flags = {}  # type: ignore[attr-defined]

        # Apply worldpack (best-effort)
        worldpack = None
        try:
            from core.worldpacks import get_worldpack_manager

            wpm = get_worldpack_manager()
            worldpack = wpm.get_worldpack(getattr(session, "world_id", "default"))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skip worldpack load: %s", exc)

        if worldpack:
            try:
                context_memory.world_flags = dict(worldpack.world_flags or {})  # type: ignore[attr-defined]
            except Exception:
                pass

            # Expose basic world info for prompt construction / UI

            # Apply built-in world setting on top of worldpack
            try:
                from core.story.world_settings import apply_world_to_session
                builtin_world = apply_world_to_session(session, getattr(session, "world_id", "default"))
                if builtin_world:
                    context_memory.world_setting_name = builtin_world.name
                    context_memory.world_setting_desc = builtin_world.description
                    logger.info(f"Applied built-in world: {builtin_world.name}")
            except Exception as _exc:
                logger.debug(f"Built-in world fallback skipped: {_exc}")
            try:
                session.current_state.story_context["world_name"] = worldpack.name
                session.current_state.story_context["world_description"] = worldpack.description
                session.current_state.story_context["world_visual"] = worldpack.visual.model_dump()
                session.current_state.story_context["worldpack_updated_at"] = worldpack.updated_at
                # World-level orchestrator config (agent_profile)
                session.current_state.story_context["agent_profile"] = (
                    worldpack.agent_profile.model_dump()
                    if getattr(worldpack, "agent_profile", None) is not None
                    else {}
                )
                if getattr(session, "world_id", "default") != "default":
                    session.current_state.story_context["setting"] = worldpack.setting
                    session.current_state.story_context["difficulty"] = worldpack.difficulty
            except Exception:
                pass

        # Resolve player template from worldpack
        player_template = None
        try:
            template_id = (
                player_template_id
                or session.current_state.story_context.get("player_template_id")
                or None
            )
            if worldpack and worldpack.player_templates:
                if template_id:
                    player_template = next(
                        (t for t in worldpack.player_templates if t.template_id == template_id),
                        None,
                    )
                if player_template is None:
                    player_template = worldpack.player_templates[0]
                    session.current_state.story_context["player_template_id"] = (
                        player_template.template_id
                    )
        except Exception:
            player_template = None

        # Create player character (use template if provided)
        player_character = GameCharacter(
            character_id="player",
            name=session.player_name,
            role=CharacterRole.PLAYER,
            personality_traits=(
                list(player_template.personality_traits)
                if player_template and player_template.personality_traits
                else ["勇敢", "好奇", "善良"]
            ),
            speaking_style=(
                player_template.speaking_style
                if player_template and player_template.speaking_style
                else "直接而友善"
            ),
            background_story=(
                player_template.background_story
                if player_template and player_template.background_story
                else f"一個開始新冒險的{session.player_name}"
            ),
            relationships={},
            motivations=(
                list(player_template.motivations)
                if player_template and player_template.motivations
                else ["探索世界", "幫助他人", "成長"]
            ),
            current_location="起點",
            persona_prompt=(
                player_template.persona_prompt if player_template else ""
            ),
        )
        context_memory.add_character(player_character)

        if player_template:
            try:
                context_memory.player_personality_profile = {  # type: ignore[attr-defined]
                    "template_id": player_template.template_id,
                    "name": player_template.name,
                    "description": player_template.description,
                    "personality_traits": player_template.personality_traits,
                    "speaking_style": player_template.speaking_style,
                    "background_story": player_template.background_story,
                    "motivations": player_template.motivations,
                    "persona_prompt": player_template.persona_prompt,
                }
            except Exception:
                pass

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

        # Add worldpack NPC/characters into context memory (best-effort)
        opening_world_characters = []
        if worldpack:
            from core.story.story_system import CharacterRole as _Role

            role_map = {
                "npc": _Role.NPC,
                "companion": _Role.COMPANION,
                "antagonist": _Role.ANTAGONIST,
            }
            for c in worldpack.characters:
                try:
                    character = GameCharacter(
                        character_id=c.character_id,
                        name=c.name,
                        role=role_map.get(c.role, _Role.NPC),
                        personality_traits=list(c.personality_traits),
                        speaking_style=c.speaking_style,
                        background_story=c.background_story,
                        motivations=list(c.motivations),
                        relationships=dict(c.relationships),
                        current_location="起點",
                        persona_prompt=c.persona_prompt,
                        content_restrictions=list(c.content_restrictions),
                    )
                    context_memory.add_character(character)
                    if getattr(c, "start_in_opening", False):
                        opening_world_characters.append(c.character_id)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Skip invalid world character %s: %s", getattr(c, "character_id", "?"), exc)
                    continue

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
            present_characters=["player", "narrator"] + opening_world_characters,
            primary_npc=opening_world_characters[0] if opening_world_characters else "narrator",
            scene_objectives=["設定背景", "介紹世界", "提供初始選擇"],
            turn_number=0,
        )
        context_memory.add_scene(opening_scene)
        context_memory.current_scene_id = "opening"

        self.context_memories[session.session_id] = context_memory

    def _rebuild_context_memory(
        self, session: GameSession
    ) -> StoryContextMemory:
        """
        Rebuild a minimal enhanced context memory for an existing session
        that lost its context file (e.g. API restart with stale data).
        Creates player + narrator characters and a current scene from session state.
        """
        context_memory = StoryContextMemory(session.session_id)
        context_memory.player_name = session.player_name
        context_memory.scene_sequence = []
        context_memory.location_states = {}
        context_memory.main_plot_points = []
        context_memory.player_decisions = []
        context_memory.player_relationships = {}
        context_memory.world_flags = {}

        # Rebuild player character from session data
        player_character = GameCharacter(
            character_id="player",
            name=session.player_name,
            role=CharacterRole.PLAYER,
            personality_traits=["勇敢", "好奇", "善良"],
            speaking_style="直接而友善",
            background_story=f"一個正在冒險中的{session.player_name}",
            relationships={},
            motivations=["探索世界", "幫助他人", "成長"],
            current_location=session.current_state.story_context.get(
                "current_location", "未知地點"
            ),
        )
        context_memory.add_character(player_character)

        # Rebuild narrator character
        try:
            persona = self.persona_manager.get_persona(session.persona_id)
            if persona:
                narrator_character = GameCharacter(
                    character_id="narrator",
                    name=persona.name,
                    role=CharacterRole.NARRATOR,
                    personality_traits=persona.personality_traits,
                    speaking_style=persona.speaking_style,
                    background_story=persona.backstory,
                    relationships={},
                    motivations=persona.goals,
                    current_location="隨處",
                )
            else:
                narrator_character = GameCharacter(
                    character_id="narrator",
                    name="旁白",
                    role=CharacterRole.NARRATOR,
                    personality_traits=["神秘", "智慧"],
                    speaking_style="敘事風格",
                    background_story="故事的講述者",
                    relationships={},
                    motivations=["推動劇情"],
                    current_location="隨處",
                )
        except Exception:
            narrator_character = GameCharacter(
                character_id="narrator",
                name="旁白",
                role=CharacterRole.NARRATOR,
                personality_traits=["神秘", "智慧"],
                speaking_style="敘事風格",
                background_story="故事的講述者",
                relationships={},
                motivations=["推動劇情"],
                current_location="隨處",
            )
        context_memory.add_character(narrator_character)

        # Rebuild current scene from session state
        current_scene_id = session.current_state.scene_id or "scene_continued"
        current_location = session.current_state.story_context.get(
            "current_location", "未知地點"
        )
        scene = SceneContext(
            scene_id=current_scene_id,
            scene_type=SceneType.EXPLORATION,
            title="冒險持續中",
            description=session.current_state.scene_description[:200]
            if session.current_state.scene_description
            else "故事正在繼續",
            location=current_location,
            time_of_day=session.current_state.story_context.get(
                "time_of_day", "白天"
            ),
            weather=session.current_state.story_context.get("weather", "晴朗"),
            atmosphere=SceneMood.MYSTERIOUS,
            present_characters=["player", "narrator"],
            primary_npc="narrator",
            scene_objectives=["繼續冒險"],
            turn_number=session.turn_count,
        )
        context_memory.add_scene(scene)
        context_memory.scene_sequence = [current_scene_id]
        context_memory.current_scene_id = current_scene_id

        logger.info(
            f"Rebuilt context memory for {session.session_id} "
            f"(scene={current_scene_id}, turn={session.turn_count})"
        )
        return context_memory

    def get_session(self, session_id: str) -> GameSession:
        """Get game session by ID (with Redis hot cache)."""
        session = self.sessions.get(session_id)
        if not session:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        # Push to Redis hot cache (background, non-blocking)
        try:
            from core.redis_cache import cache_session
            cache_session(session_id, session.to_dict())
        except Exception:
            pass
        return session

    def _load_sessions(self):
        """Load existing game sessions from storage"""
        loaded_contexts = 0
        rebuilt_contexts = 0
        skipped_sessions = 0

        for session_file in self.sessions_path.glob("*.json"):
            # Skip .context.json files — they are context memory dumps, not session data
            if session_file.name.endswith(".context.json"):
                continue

            session = None
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    session_data = json.load(f)
                # Guard: if the JSON doesn't look like a session dict, skip it
                if not isinstance(session_data, dict):
                    logger.warning("Session file %s is not a dict, skipping", session_file)
                    skipped_sessions += 1
                    continue
                session = GameSession.from_dict(session_data)
                self.sessions[session.session_id] = session
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skip invalid session file %s: %s", session_file, exc)
                skipped_sessions += 1
                continue

            if self.enhanced_mode and session:
                context_file = session_file.with_suffix(".context.json")
                if context_file.exists():
                    try:
                        with open(context_file, "r", encoding="utf-8") as cf:
                            context_data = json.load(cf)
                        context_memory = self._deserialize_context_memory(
                            context_data
                        )
                        self.context_memories[session.session_id] = context_memory
                        loaded_contexts += 1
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Failed to load context for %s: %s",
                            session.session_id,
                            exc,
                        )
                        # Auto-rebuild minimal context memory for sessions with corrupt context
                        try:
                            context_memory = self._rebuild_context_memory(session)
                            self.context_memories[session.session_id] = context_memory
                            rebuilt_contexts += 1
                        except Exception as rebuild_err:
                            logger.error(
                                "Failed to rebuild context for %s: %s",
                                session.session_id,
                                rebuild_err,
                            )
                else:
                    # Session has no context file at all — rebuild from session data
                    try:
                        context_memory = self._rebuild_context_memory(session)
                        self.context_memories[session.session_id] = context_memory
                        rebuilt_contexts += 1
                    except Exception as rebuild_err:
                        logger.error(
                            "Failed to rebuild context for %s: %s",
                            session.session_id,
                            rebuild_err,
                        )

        logger.info(
            f"Loaded {len(self.sessions)} sessions, "
            f"{loaded_contexts} enhanced contexts, "
            f"{rebuilt_contexts} rebuilt contexts, "
            f"{skipped_sessions} skipped"
        )

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

    def save_game_slot(self, session_id: str, slot_name: str) -> Dict[str, Any]:
        """Save a game session to a specific named slot."""
        if session_id not in self.sessions:
            raise SessionNotFoundError(f"Session not found: {session_id}")
            
        session = self.sessions[session_id]
        save_root = self.sessions_path.parent / "saves" / session_id
        save_root.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().isoformat()
        slot_id = f"slot_{int(datetime.now().timestamp())}"
        
        # Export full session data
        exported_data = self.export_session_data(session_id, include_context=True)
        
        # Create slot metadata
        slot_info = {
            "slot_id": slot_id,
            "name": slot_name or f"存檔 {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "timestamp": timestamp,
            "turn_count": session.turn_count,
            "scene_image_url": session.current_state.story_context.get("last_scene_image", {}).get("image_url", ""),
            "player_name": session.player_name,
            "stats": session.stats.to_dict() if hasattr(session.stats, "to_dict") else {}
        }
        
        # Save metadata and data
        save_file = save_root / f"{slot_id}.json"
        with open(save_file, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": slot_info,
                "data": exported_data
            }, f, ensure_ascii=False, indent=2)
            
        return slot_info

    def list_game_slots(self, session_id: str) -> List[Dict[str, Any]]:
        """List all save slots for a session."""
        save_root = self.sessions_path.parent / "saves" / session_id
        if not save_root.exists():
            return []
            
        slots = []
        for file in save_root.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    if "metadata" in content:
                        slots.append(content["metadata"])
            except Exception:
                continue
                
        # Sort by timestamp descending
        slots.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return slots

    def load_game_slot(self, session_id: str, slot_id: str) -> GameSession:
        """Load a game session from a specific slot."""
        save_file = self.sessions_path.parent / "saves" / session_id / f"{slot_id}.json"
        if not save_file.exists():
            raise GameError(f"Save slot {slot_id} not found for session {session_id}")
            
        with open(save_file, "r", encoding="utf-8") as f:
            content = json.load(f)
            
        # Import data (this might overwrite the current session in memory)
        success = self.import_session_data(content["data"])
        if not success:
            raise GameError("Failed to import session data from save slot")
            
        # Reload the session from path to be safe
        session = self._load_session_by_id(session_id)
        if not session:
            raise SessionNotFoundError(f"Session {session_id} could not be reloaded after import")
            
        self.sessions[session_id] = session
        return session

    def get_narrative_flowchart(self, session_id: str) -> Dict[str, Any]:
        """Generate a scene-based flowchart with Scene → Choice → NPC nodes and proper connections.

        Returns:
        - nodes: list of {id, type, label, scene_id, turn, ...}
        - edges: list of {id, source, target, label} — all directed correctly (A → B)
        """
        if session_id not in self.sessions:
            raise SessionNotFoundError(f"Session not found: {session_id}")

        session = self.sessions[session_id]
        history = getattr(session, "history", [])

        nodes = []
        edges = []
        edge_counter = [0]

        def _add_edge(source: str, target: str, label: str = ""):
            edge_counter[0] += 1
            edges.append({
                "id": f"e_{edge_counter[0]}",
                "source": source,
                "target": target,
                "label": label,
            })

        # Collect unique scenes from history
        scene_map: Dict[str, int] = {}  # scene_id -> first turn
        turn_to_scene: Dict[int, str] = {}  # turn -> scene_id

        for entry in history:
            turn_num = entry.get("turn", 0)
            scene_id = entry.get("scene_id") or entry.get("scene") or f"scene_{turn_num}"
            turn_to_scene[turn_num] = scene_id
            if scene_id not in scene_map:
                scene_map[scene_id] = turn_num

        # Ordered scene list (by first appearance)
        ordered_scenes = sorted(scene_map.keys(), key=lambda s: scene_map[s])

        # ── Start node ──
        nodes.append({
            "id": "start",
            "type": "start",
            "label": "故事開始",
            "scene_id": None,
            "turn": 0,
        })
        if ordered_scenes:
            _add_edge("start", f"scene:{ordered_scenes[0]}", "開始")

        # ── Scene nodes ──
        for idx, scene_id in enumerate(ordered_scenes):
            snode_id = f"scene:{scene_id}"
            first_turn = scene_map[scene_id]
            scene_turns = [t for t, s in turn_to_scene.items() if s == scene_id]
            # Build a short summary from the first turn in this scene
            first_entry = next((e for e in history if e.get("turn") == first_turn), {})
            summary = (first_entry.get("action") or first_entry.get("result") or "")[:80]

            nodes.append({
                "id": snode_id,
                "type": "scene",
                "label": f"場景 {idx + 1}",
                "scene_id": scene_id,
                "turn": first_turn,
                "turns": sorted(scene_turns),
                "summary": summary,
                "image_url": first_entry.get("scene_image", {}).get("image_url", ""),
            })

            # Edge: Scene N → Scene N+1
            if idx < len(ordered_scenes) - 1:
                next_scene = ordered_scenes[idx + 1]
                _add_edge(snode_id, f"scene:{next_scene}", f"前往場景 {idx + 2}")

            # ── Choice nodes for each turn in this scene ──
            for t in scene_turns:
                entry = next((e for e in history if e.get("turn") == t), {})
                choices = entry.get("choices", [])
                player_input = entry.get("action") or entry.get("player_input", "")
                made_choice = entry.get("choice_made", "")

                cnode_id = f"choice:turn_{t}"
                nodes.append({
                    "id": cnode_id,
                    "type": "choice",
                    "label": f"選擇 (T{t})",
                    "scene_id": scene_id,
                    "turn": t,
                    "player_input": player_input,
                    "made_choice": made_choice,
                    "options": [c.get("label") or c.get("text") or c.get("choice_id", "?") for c in choices[:4]] if isinstance(choices, list) else [],
                })

                # Scene → Choice (branch out)
                _add_edge(snode_id, cnode_id, player_input[:30] if player_input else f"回合 {t}")

            # ── NPC nodes in this scene ──
            npcs_in_scene: Dict[str, Dict] = {}
            for t in scene_turns:
                entry = next((e for e in history if e.get("turn") == t), {})
                for npc_key in ("npcs", "characters", "encounters"):
                    npc_list = entry.get(npc_key, [])
                    for npc in (npc_list if isinstance(npc_list, list) else []):
                        npc_name = npc.get("name") or npc.get("character_id", "") if isinstance(npc, dict) else str(npc)
                        if npc_name:
                            npcs_in_scene[npc_name] = npc

            for npc_name, npc_data in npcs_in_scene.items():
                nnpc_id = f"npc:{npc_name}"
                role = npc_data.get("role", "") if isinstance(npc_data, dict) else ""
                nodes.append({
                    "id": nnpc_id,
                    "type": "npc",
                    "label": npc_name,
                    "scene_id": scene_id,
                    "turn": first_turn,
                    "role": role,
                })
                # Scene → NPC (NPC belongs to this scene)
                _add_edge(snode_id, nnpc_id, role or "登場")

        return {"nodes": nodes, "edges": edges, "scenes": len(ordered_scenes), "total_turns": len(history)}

    def save_session(self, session: GameSession) -> None:
        """Persist a session after mutating in-memory fields."""
        self._save_session(session)

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

        # Update session timestamp (turn_count is incremented by add_to_history, not here)
        session.updated_at = datetime.now()

        # Save session
        self._save_session(session)

        return result

    async def process_turn(
        self, session_id: str, player_input: str, choice_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Process a player turn with enhanced or classic mode.

        Args:
            session_id: Active session id.
            player_input: Player action text.
            choice_id: Optional choice id to execute.
            timeout: Maximum seconds allowed for this turn (default 90s).
                     On timeout the engine returns a partial result with a notice.
        """
        effective_timeout = timeout if timeout is not None else 90.0

        async def _do_turn() -> Dict[str, Any]:
            if self.enhanced_mode and session_id in self.context_memories:
                return await self._process_enhanced_turn(
                    session_id, player_input, choice_id
                )
            else:
                return await self._process_classic_turn(session_id, player_input, choice_id)

        try:
            result = await asyncio.wait_for(_do_turn(), timeout=effective_timeout)
            return result
        except asyncio.TimeoutError:
            logger.warning(
                "Turn processing timed out after %.1fs for session %s", effective_timeout, session_id
            )
            # Return a partial / graceful result
            try:
                session = self.get_session(session_id)
            except Exception:
                session = None

            fallback_narrative = (
                "故事仍在繼續，但回應來得較慢。請再試一次你的行動，或選擇其他選項。"
            )
            return {
                "narrative": fallback_narrative,
                "choices": [],
                "stats": session.stats.to_dict() if session and hasattr(session.stats, "to_dict") else {},
                "inventory": list(getattr(session, "inventory", []) or []),
                "flags": dict(getattr(getattr(session, "current_state", None), "flags", {}) or {}),
                "scene_id": getattr(getattr(session, "current_state", None), "scene_id", None),
                "error": "turn_timeout",
                "error_message": f"Turn processing timed out after {effective_timeout}s",
                "partial": True,
            }

    async def process_turn_stream(
        self,
        session_id: str,
        player_input: str,
        choice_id: Optional[str] = None,
    ):
        """Process a player turn with SSE streaming for narrative tokens.

        Yields JSON lines:
          - {"type": "token", "content": "..."}  — narrative text tokens
          - {"type": "done",  "result": {...}}     — final complete result dict
        """
        if not (self.enhanced_mode and session_id in self.context_memories):
            # Fallback: run normal turn, emit as single chunk
            result = await self.process_turn(session_id, player_input, choice_id)
            yield {"type": "token", "content": result.get("narrative", "")}
            yield {"type": "done", "result": result}
            return

        # Enhanced streaming turn — replicate _process_enhanced_turn logic
        # but collect narrative tokens via stream_callback and yield them
        try:
            session = self.get_session(session_id)
            self._apply_runtime_preset_llm(session)
            context_memory = self.context_memories[session_id]
            persona = self.persona_manager.get_persona(session.persona_id)
            if not persona:
                raise GameError(f"Persona not found: {session.persona_id}")

            # Snapshots for state delta tracking
            pre_flags: Dict[str, Any] = {}
            pre_stats: Dict[str, Any] = {}
            pre_inventory: List[str] = []
            pre_relationships: Dict[str, Any] = {}
            try:
                import copy
                pre_flags = copy.deepcopy(getattr(session.current_state, "flags", {}) or {})
                pre_stats = copy.deepcopy(session.stats.to_dict() if hasattr(session.stats, "to_dict") else {})
                pre_inventory = list(getattr(session, "inventory", []) or [])
                pre_relationships = copy.deepcopy(getattr(context_memory, "player_relationships", {}) or {})
            except Exception:
                pass

            # Execute choice if provided
            choice_result = None
            if choice_id:
                choice_result = await self._execute_contextual_choice(
                    context_memory, choice_id, player_input, session
                )

            # Narrative generation with streaming callback
            # Use a plain list (not async queue) to avoid blocking the SSE reader
            import asyncio
            token_list: List[str] = []

            def _stream_wrapper(token: str):
                """Sync callback — append to list without blocking."""
                token_list.append(token)

            # Generate narrative with streaming
            if hasattr(self.narrative_generator, 'generate_contextual_narrative'):
                narrative_result = await self.narrative_generator.generate_contextual_narrative(
                    context_memory,
                    player_input,
                    choice_result,
                    inventory=session.inventory,
                    stats=session.stats.to_dict(),
                    stream_callback=_stream_wrapper,
                )
            else:
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

            # Drain collected tokens and yield
            logger.info(f"[STREAM] Collected {len(token_list)} tokens from narrative stream")
            for token in token_list:
                yield {"type": "token", "content": token}

            # ── Rest of _process_enhanced_turn logic (identical) ──

            # Apply state delta from AI
            state_delta = narrative_result.get("state_delta", {})
            _empty = {"inventory_add": [], "inventory_remove": [], "stat_changes": {}, "flags_triggered": []}
            if state_delta != _empty:
                logger.info(f"State delta from AI: {state_delta}")
            if state_delta:
                for item in state_delta.get("inventory_add", []):
                    if item not in session.inventory:
                        session.inventory.append(item)
                for item in state_delta.get("inventory_remove", []):
                    if item in session.inventory:
                        try:
                            session.inventory.remove(item)
                        except Exception:
                            pass
                stat_changes = state_delta.get("stat_changes", {})
                for stat, change in stat_changes.items():
                    if hasattr(session.stats, stat):
                        try:
                            current = getattr(session.stats, stat)
                            setattr(session.stats, stat, current + int(change))
                        except Exception:
                            pass
                for flag in state_delta.get("flags_triggered", []):
                    session.current_state.flags[flag] = True

            # P1 subsystems
            combat_result = None
            achievements_unlocked = []
            economy_result = None
            relationship_updates = []
            map_updates = {}
            narrative_text = narrative_result["main_narrative"]

            combat_result = await self._handle_combat(
                session_id, session, narrative_text, context_memory, choice_id
            )
            if combat_result:
                for stat, change in combat_result.get("stat_changes", {}).items():
                    if hasattr(session.stats, stat):
                        try:
                            current = getattr(session.stats, stat)
                            setattr(session.stats, stat, max(0, current + int(change)))
                        except Exception:
                            pass
                for xp in combat_result.get("xp_gained", []):
                    if hasattr(session.stats, "experience"):
                        session.stats.experience += xp

            achievements_unlocked = await self._check_achievements(
                session_id, session, context_memory, narrative_text
            )
            economy_result = await self._handle_economy(
                session_id, session, narrative_text, combat_result
            )
            relationship_updates = await self._handle_relationships(
                session_id, session, context_memory, narrative_text, player_input
            )
            map_updates = await self._handle_map_exploration(
                session_id, session, context_memory, narrative_text
            )

            # Update context memory
            try:
                await self._update_context_memory(
                    context_memory, player_input, narrative_result, choice_result
                )
                self._save_session(session)
                try:
                    session.current_state.scene_id = str(getattr(context_memory, "current_scene_id", "") or session.current_state.scene_id)
                except Exception:
                    pass
            except Exception as e:
                logger.error(f"Failed to update context memory: {e}", exc_info=True)
                raise

            # Generate new choices
            try:
                llm_choices = await asyncio.to_thread(
                    self._generate_llm_choices_sync,
                    narrative_result["main_narrative"],
                    context_memory,
                    session,
                )
                new_choices = llm_choices
            except Exception as e:
                logger.warning(f"LLM choice generation failed ({e}), falling back to templates")
                try:
                    new_choices = self._generate_contextual_choices(context_memory)
                except Exception:
                    new_choices = []

            # GUARD: Ensure choices are NEVER empty — use hard-coded fallback
            if not new_choices:
                logger.warning("All choice generators returned empty — using emergency fallback choices")
                new_choices = [
                    {"choice_id": "continue_forward", "text": "繼續前進", "type": "action", "difficulty": "easy"},
                    {"choice_id": "look_around", "text": "仔細觀察周圍環境", "type": "exploration", "difficulty": "easy"},
                    {"choice_id": "check_status", "text": "檢查自身狀態和裝備", "type": "action", "difficulty": "easy"},
                    {"choice_id": "wait_observe", "text": "等待並觀察情況發展", "type": "action", "difficulty": "easy"},
                ]

            # Update session state
            session.current_state.scene_description = narrative_result["main_narrative"]
            formatted_choices = []
            for choice in new_choices:
                if isinstance(choice, dict):
                    formatted_choices.append({
                        "choice_id": choice.get("choice_id", f"choice_{len(formatted_choices)}"),
                        "text": choice.get("text", "繼續行動"),
                        "type": choice.get("type", "action"),
                        "difficulty": choice.get("difficulty", "medium"),
                    })
                else:
                    try:
                        formatted_choices.append({
                            "choice_id": choice.choice_id,
                            "text": choice.get_display_text(context_memory),
                            "type": choice.choice_type,
                            "difficulty": choice.difficulty,
                        })
                    except Exception:
                        formatted_choices.append({
                            "choice_id": getattr(choice, 'choice_id', f"choice_{len(formatted_choices)}"),
                            "text": getattr(choice, 'base_text', '繼續行動'),
                            "type": getattr(choice, 'choice_type', 'action'),
                            "difficulty": getattr(choice, 'difficulty', 'medium'),
                        })
            session.current_state.available_choices = formatted_choices

            # Add to history
            session.add_to_history(
                player_input, narrative_result["main_narrative"], choice_id
            )

            # Mirror turn into context_memory
            try:
                context_memory.turn_history.append({
                    "turn": session.turn_count - 1,
                    "player_input": player_input,
                    "ai_response": narrative_result["main_narrative"],
                    "choice_id": choice_id,
                    "scene_id": getattr(session.current_state, "scene_id", None),
                    "timestamp": datetime.now().isoformat(),
                })
                if len(context_memory.turn_history) > 20:
                    context_memory.turn_history = context_memory.turn_history[-20:]
            except Exception:
                pass

            # Agent decision layer
            agent_actions = None
            if self.agent_enabled:
                try:
                    from core.agents.story_agent_layer import get_agent_layer
                    agent_layer = get_agent_layer()
                    should_intervene, intervention_reason = await agent_layer.should_agent_intervene(
                        session_id, player_input, narrative_result["main_narrative"], context_memory
                    )
                    if should_intervene:
                        decision = await agent_layer.make_decision(
                            session_id, player_input, narrative_result["main_narrative"], context_memory, session.stats.to_dict()
                        )
                        if decision:
                            agent_actions = await agent_layer.execute_decision(session_id, decision)
                except Exception as e:
                    logger.warning(f"Agent decision failed: {e}")

            # Persist per-turn artifacts
            try:
                if getattr(session, "history", None) and isinstance(session.history[-1], dict):
                    session.history[-1]["agent_used"] = bool(agent_actions)
                    if agent_actions:
                        session.history[-1]["agent_actions"] = agent_actions
            except Exception:
                pass

            try:
                import copy
                from collections import Counter
                post_flags = copy.deepcopy(getattr(session.current_state, "flags", {}) or {})
                post_stats = copy.deepcopy(session.stats.to_dict() if hasattr(session.stats, "to_dict") else {})
                post_inventory = list(getattr(session, "inventory", []) or [])
                post_relationships = copy.deepcopy(getattr(context_memory, "player_relationships", {}) or {})

                flag_changes: List[Dict[str, Any]] = []
                for k in sorted(set(pre_flags.keys()) | set(post_flags.keys())):
                    old = pre_flags.get(k)
                    new = post_flags.get(k)
                    if old != new:
                        flag_changes.append({"key": k, "old": old, "new": new})

                stat_changes: List[Dict[str, Any]] = []
                for k in sorted(set(pre_stats.keys()) | set(post_stats.keys())):
                    old = pre_stats.get(k)
                    new = post_stats.get(k)
                    if old == new:
                        continue
                    change = None
                    try:
                        if isinstance(old, (int, float)) and isinstance(new, (int, float)):
                            change = new - old
                    except Exception:
                        change = None
                    stat_changes.append({"key": k, "old": old, "new": new, "change": change})

                pre_inv = Counter(pre_inventory)
                post_inv = Counter(post_inventory)
                inv_added: List[Dict[str, Any]] = []
                inv_removed: List[Dict[str, Any]] = []
                for item in sorted(set(pre_inv.keys()) | set(post_inv.keys())):
                    delta = int(post_inv.get(item, 0)) - int(pre_inv.get(item, 0))
                    if delta > 0:
                        inv_added.append({"item": item, "count": delta})
                    elif delta < 0:
                        inv_removed.append({"item": item, "count": abs(delta)})

                rel_changes: List[Dict[str, Any]] = []
                for cid in sorted(set(pre_relationships.keys()) | set(post_relationships.keys())):
                    try:
                        old = int(pre_relationships.get(cid, 0) or 0)
                        new = int(post_relationships.get(cid, 0) or 0)
                    except Exception:
                        continue
                    if old != new:
                        rel_changes.append({"character_id": str(cid), "old": old, "new": new, "change": new - old})

                state_delta_record = {
                    "flags": flag_changes[:80],
                    "stats": stat_changes[:40],
                    "inventory": {"added": inv_added[:30], "removed": inv_removed[:30]},
                    "relationships": rel_changes[:40],
                }
                if getattr(session, "history", None) and isinstance(session.history[-1], dict):
                    session.history[-1]["state_delta"] = state_delta_record
            except Exception:
                pass

            # Record turn in memory manager
            try:
                from core.story.memory_manager import get_memory_manager
                memory_manager = get_memory_manager(session_id)
                await memory_manager.record_turn(
                    turn_number=session.turn_count,
                    player_input=player_input,
                    narrative_response=narrative_result["main_narrative"],
                    scene_id=context_memory.current_scene.scene_id if context_memory.current_scene else None,
                    choices_made=[choice_id] if choice_id else [],
                    flags_changed={},
                    stats_changed={},
                )
            except Exception as e:
                logger.warning(f"Failed to record turn in memory: {e}")

            # Save session
            self._save_session(session)

            result = {
                "session_id": session.session_id,
                "turn_count": session.turn_count,
                "narrative": narrative_result["main_narrative"],
                "character_dialogues": narrative_result.get("character_dialogues", []),
                "choices": session.current_state.available_choices,
                "stats": session.stats.to_dict(),
                "inventory": session.inventory,
                "scene_info": {
                    "current_scene": (
                        context_memory.get_current_scene().to_dict()
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
            if agent_actions:
                result["agent_actions"] = agent_actions

            yield {"type": "done", "result": result}

        except Exception as e:
            logger.error(f"Failed to process streaming turn for {session_id}: {e}")
            # Fallback
            try:
                result = await self._process_classic_turn(session_id, player_input, choice_id)
                yield {"type": "token", "content": result.get("narrative", "")}
                yield {"type": "done", "result": result}
            except Exception:
                yield {"type": "done", "result": {"error": str(e), "narrative": "敘事生成失敗", "choices": []}}

    async def _process_enhanced_turn(
        self, session_id: str, player_input: str, choice_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process turn with enhanced context management"""

        try:
            logger.debug(f"Processing enhanced turn for session {session_id}")
            session = self.get_session(session_id)
            self._apply_runtime_preset_llm(session)
            context_memory = self.context_memories[session_id]
            persona = self.persona_manager.get_persona(session.persona_id)

            if not persona:
                raise GameError(f"Persona not found: {session.persona_id}")
        except Exception as e:
            logger.error(f"Failed to get session/persona: {e}", exc_info=True)
            raise

        # Snapshot state for per-turn diffs (Turn Inspector)
        pre_flags: Dict[str, Any] = {}
        pre_stats: Dict[str, Any] = {}
        pre_inventory: List[str] = []
        pre_relationships: Dict[str, Any] = {}
        try:
            import copy

            pre_flags = copy.deepcopy(getattr(session.current_state, "flags", {}) or {})
            pre_stats = copy.deepcopy(session.stats.to_dict() if hasattr(session.stats, "to_dict") else {})
            pre_inventory = list(getattr(session, "inventory", []) or [])
            pre_relationships = copy.deepcopy(getattr(context_memory, "player_relationships", {}) or {})
        except Exception:
            pre_flags = {}
            pre_stats = {}
            pre_inventory = []
            pre_relationships = {}

        try:
            # Execute choice if provided
            choice_result = None
            if choice_id:
                choice_result = await self._execute_contextual_choice(
                    context_memory, choice_id, player_input, session
                )

            # Generate narrative with enhanced context
            # Use hasattr to check for generate_contextual_narrative method
            # (avoids isinstance class mismatch when EnhancedNarrativeGenerator
            #  is imported from different module paths)
            if hasattr(self.narrative_generator, 'generate_contextual_narrative'):
                narrative_result = await self.narrative_generator.generate_contextual_narrative(
                    context_memory,
                    player_input,
                    choice_result,
                    inventory=session.inventory,
                    stats=session.stats.to_dict(),
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

            # Apply state delta from AI (Consistency Enforcement)
            state_delta = narrative_result.get("state_delta", {})
            _empty = {"inventory_add": [], "inventory_remove": [], "stat_changes": {}, "flags_triggered": []}
            if state_delta != _empty:
                logger.info(f"State delta from AI: {state_delta}")
            if state_delta:
                logger.info(f"Applying state delta from AI: {state_delta}")
                # Add items
                for item in state_delta.get("inventory_add", []):
                    if item not in session.inventory:
                        session.inventory.append(item)
                # Remove items
                for item in state_delta.get("inventory_remove", []):
                    if item in session.inventory:
                        try:
                            session.inventory.remove(item)
                        except Exception:
                            pass
                # Update stats
                stat_changes = state_delta.get("stat_changes", {})
                for stat, change in stat_changes.items():
                    if hasattr(session.stats, stat):
                        try:
                            current = getattr(session.stats, stat)
                            setattr(session.stats, stat, current + int(change))
                        except Exception:
                            pass
                # Update flags
                for flag in state_delta.get("flags_triggered", []):
                    session.current_state.flags[flag] = True

            # ── P1: Subsystem hooks ──
            combat_result = None
            achievements_unlocked = []
            economy_result = None
            relationship_updates = []
            map_updates = {}
            narrative_text = narrative_result["main_narrative"]

            # 1. Combat detection
            combat_result = await self._handle_combat(
                session_id, session, narrative_text, context_memory, choice_id
            )
            if combat_result:
                logger.info("Combat resolved: %s", combat_result.get("result"))
                # Merge combat stat changes into session
                for stat, change in combat_result.get("stat_changes", {}).items():
                    if hasattr(session.stats, stat):
                        try:
                            current = getattr(session.stats, stat)
                            setattr(session.stats, stat, max(0, current + int(change)))
                        except Exception:
                            pass
                # Merge combat XP
                for xp in combat_result.get("xp_gained", []):
                    if hasattr(session.stats, "experience"):
                        session.stats.experience += xp

            # 2. Achievement check
            achievements_unlocked = await self._check_achievements(
                session_id, session, context_memory, narrative_text
            )
            if achievements_unlocked:
                logger.info("Achievements unlocked: %s", [a.get("name") for a in achievements_unlocked])

            # 3. Economy (gold from combat/exploration)
            economy_result = await self._handle_economy(
                session_id, session, narrative_text, combat_result
            )

            # 4. Relationship updates
            relationship_updates = await self._handle_relationships(
                session_id, session, context_memory, narrative_text, player_input
            )

            # 5. Map exploration
            map_updates = await self._handle_map_exploration(
                session_id, session, context_memory, narrative_text
            )

            # Update context memory
            try:
                logger.debug("Updating context memory...")
                await self._update_context_memory(
                    context_memory, player_input, narrative_result, choice_result
                )
                # Persist context memory changes immediately
                self._save_session(session)
                # Keep session scene_id aligned with enhanced context for UI/history
                try:
                    session.current_state.scene_id = str(getattr(context_memory, "current_scene_id", "") or session.current_state.scene_id)
                except Exception:
                    pass
            except Exception as e:
                logger.error(f"Failed to update context memory: {e}", exc_info=True)
                raise

            # Generate new choices — LLM-driven for variety
            try:
                logger.debug("Generating LLM contextual choices...")
                # Run in a separate thread to avoid event loop conflicts with httpx
                llm_choices = await asyncio.to_thread(
                    self._generate_llm_choices_sync,
                    narrative_result["main_narrative"],
                    context_memory,
                    session,
                )
                new_choices = llm_choices
                logger.debug(f"Generated {len(new_choices)} LLM choices")
            except Exception as e:
                logger.warning(f"LLM choice generation failed ({e}), falling back to templates", exc_info=True)
                try:
                    new_choices = self._generate_contextual_choices(context_memory)
                except Exception:
                    new_choices = []

            # GUARD: Ensure choices are NEVER empty — use hard-coded fallback
            if not new_choices:
                logger.warning("All choice generators returned empty — using emergency fallback choices")
                new_choices = [
                    {"choice_id": "continue_forward", "text": "繼續前進", "type": "action", "difficulty": "easy"},
                    {"choice_id": "look_around", "text": "仔細觀察周圍環境", "type": "exploration", "difficulty": "easy"},
                    {"choice_id": "check_status", "text": "檢查自身狀態和裝備", "type": "action", "difficulty": "easy"},
                    {"choice_id": "wait_observe", "text": "等待並觀察情況發展", "type": "action", "difficulty": "easy"},
                ]

            # Update session state
            try:
                logger.debug("Updating session state...")
                session.current_state.scene_description = narrative_result["main_narrative"]

                # Handle both dict (LLM) and object (template) choices
                formatted_choices = []
                for choice in new_choices:
                    if isinstance(choice, dict):
                        formatted_choices.append({
                            "choice_id": choice.get("choice_id", f"choice_{len(formatted_choices)}"),
                            "text": choice.get("text", "繼續行動"),
                            "type": choice.get("type", "action"),
                            "difficulty": choice.get("difficulty", "medium"),
                        })
                    else:
                        try:
                            choice_text = choice.get_display_text(context_memory)
                            formatted_choices.append({
                                "choice_id": choice.choice_id,
                                "text": choice_text,
                                "type": choice.choice_type,
                                "difficulty": choice.difficulty,
                            })
                        except Exception:
                            formatted_choices.append({
                                "choice_id": getattr(choice, 'choice_id', f"choice_{len(formatted_choices)}"),
                                "text": getattr(choice, 'base_text', '繼續行動'),
                                "type": getattr(choice, 'choice_type', 'action'),
                                "difficulty": getattr(choice, 'difficulty', 'medium'),
                            })

                session.current_state.available_choices = formatted_choices
                logger.debug(f"Updated session with {len(session.current_state.available_choices)} choices")
            except Exception as e:
                logger.error(f"Failed to update session state: {e}", exc_info=True)
                raise

            # Add to history (add_to_history increments turn_count internally)
            session.add_to_history(
                player_input, narrative_result["main_narrative"], choice_id
            )

            # Mirror turn into context_memory for narrative generator
            # so it can see recent I/O and avoid regenerating the same content.
            try:
                context_memory.turn_history.append({
                    "turn": session.turn_count - 1,
                    "player_input": player_input,
                    "ai_response": narrative_result["main_narrative"],
                    "choice_id": choice_id,
                    "scene_id": getattr(session.current_state, "scene_id", None),
                    "timestamp": datetime.now().isoformat(),
                })
                # Keep bounded
                if len(context_memory.turn_history) > 20:
                    context_memory.turn_history = context_memory.turn_history[-20:]
            except Exception:
                pass

            # Agent decision layer (optional)
            agent_actions = None
            if self.agent_enabled:
                try:
                    agent_layer = get_agent_layer()

                    # Check if Agent should intervene
                    should_intervene, intervention_reason = await agent_layer.should_agent_intervene(
                        session_id,
                        player_input,
                        narrative_result["main_narrative"],
                        context_memory
                    )

                    if should_intervene:
                        logger.info(f"Agent intervening: {intervention_reason}")

                        # Let Agent make decision
                        decision = await agent_layer.make_decision(
                            session_id,
                            player_input,
                            narrative_result["main_narrative"],
                            context_memory,
                            session.stats.to_dict()
                        )

                        if decision:
                            # Execute Agent decision
                            agent_actions = await agent_layer.execute_decision(
                                session_id,
                                decision
                            )

                            logger.info(
                                f"Agent executed {len(decision.tool_calls)} actions: "
                                f"{agent_actions['overall_success']}"
                            )
                except Exception as e:
                    logger.warning(f"Agent decision failed: {e}")

            # Persist per-turn agent actions into session history for UI timeline
            try:
                if getattr(session, "history", None) and isinstance(session.history[-1], dict):
                    session.history[-1]["agent_used"] = bool(agent_actions)
                    if agent_actions:
                        session.history[-1]["agent_actions"] = agent_actions
                    # Normalized artifacts (best-effort; router may enrich further)
                    try:
                        entry = session.history[-1]
                        artifacts = entry.get("artifacts")
                        if not isinstance(artifacts, dict):
                            artifacts = {}
                        agents_bucket = artifacts.get("agents")
                        if not isinstance(agents_bucket, dict):
                            agents_bucket = {}
                        agents_bucket["used"] = bool(entry.get("agent_used")) or bool(agent_actions)
                        if agent_actions:
                            agents_bucket["actions"] = agent_actions
                        artifacts["agents"] = agents_bucket
                        entry["artifacts"] = artifacts
                    except Exception:
                        pass
            except Exception:
                pass

            # Persist per-turn state delta into session history for UI timeline
            try:
                import copy
                from collections import Counter

                post_flags = copy.deepcopy(getattr(session.current_state, "flags", {}) or {})
                post_stats = copy.deepcopy(session.stats.to_dict() if hasattr(session.stats, "to_dict") else {})
                post_inventory = list(getattr(session, "inventory", []) or [])
                post_relationships = copy.deepcopy(getattr(context_memory, "player_relationships", {}) or {})

                flag_changes: List[Dict[str, Any]] = []
                for k in sorted(set(pre_flags.keys()) | set(post_flags.keys())):
                    old = pre_flags.get(k)
                    new = post_flags.get(k)
                    if old != new:
                        flag_changes.append({"key": k, "old": old, "new": new})

                stat_changes: List[Dict[str, Any]] = []
                for k in sorted(set(pre_stats.keys()) | set(post_stats.keys())):
                    old = pre_stats.get(k)
                    new = post_stats.get(k)
                    if old == new:
                        continue
                    change = None
                    try:
                        if isinstance(old, (int, float)) and isinstance(new, (int, float)):
                            change = new - old
                    except Exception:
                        change = None
                    stat_changes.append({"key": k, "old": old, "new": new, "change": change})

                pre_inv = Counter(pre_inventory)
                post_inv = Counter(post_inventory)
                inv_added: List[Dict[str, Any]] = []
                inv_removed: List[Dict[str, Any]] = []
                for item in sorted(set(pre_inv.keys()) | set(post_inv.keys())):
                    delta = int(post_inv.get(item, 0)) - int(pre_inv.get(item, 0))
                    if delta > 0:
                        inv_added.append({"item": item, "count": delta})
                    elif delta < 0:
                        inv_removed.append({"item": item, "count": abs(delta)})

                rel_changes: List[Dict[str, Any]] = []
                for cid in sorted(set(pre_relationships.keys()) | set(post_relationships.keys())):
                    try:
                        old = int(pre_relationships.get(cid, 0) or 0)
                        new = int(post_relationships.get(cid, 0) or 0)
                    except Exception:
                        continue
                    if old != new:
                        rel_changes.append(
                            {"character_id": str(cid), "old": old, "new": new, "change": new - old}
                        )

                state_delta = {
                    "flags": flag_changes[:80],
                    "stats": stat_changes[:40],
                    "inventory": {"added": inv_added[:30], "removed": inv_removed[:30]},
                    "relationships": rel_changes[:40],
                }

                if getattr(session, "history", None) and isinstance(session.history[-1], dict):
                    session.history[-1]["state_delta"] = state_delta
                    # Normalized artifacts (best-effort)
                    try:
                        entry = session.history[-1]
                        artifacts = entry.get("artifacts")
                        if not isinstance(artifacts, dict):
                            artifacts = {}
                        artifacts["diff"] = state_delta
                        entry["artifacts"] = artifacts
                    except Exception:
                        pass
            except Exception:
                pass

            # Record turn in memory manager
            try:
                memory_manager = get_memory_manager(session_id)

                # Track changes from Agent actions
                flags_changed = {}
                stats_changed = {}
                if agent_actions and agent_actions.get("overall_success"):
                    for tool_result in agent_actions.get("tool_results", []):
                        if tool_result.get("success") and tool_result.get("tool") == "modify_world_state":
                            result_data = tool_result.get("result", {})
                            if "modified_flags" in result_data:
                                flags_changed.update(result_data["modified_flags"])
                        elif tool_result.get("success") and tool_result.get("tool") == "update_character_state":
                            result_data = tool_result.get("result", {})
                            if "modified_stats" in result_data:
                                stats_changed.update(result_data["modified_stats"])

                await memory_manager.record_turn(
                    turn_number=session.turn_count,
                    player_input=player_input,
                    narrative_response=narrative_result["main_narrative"],
                    scene_id=context_memory.current_scene.scene_id if context_memory.current_scene else None,
                    choices_made=[choice_id] if choice_id else [],
                    flags_changed=flags_changed,
                    stats_changed=stats_changed
                )
            except Exception as e:
                logger.warning(f"Failed to record turn in memory: {e}")

            # Save session
            self._save_session(session)

            result = {
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

            # Add Agent actions if available
            if agent_actions:
                result["agent_actions"] = agent_actions

            return result

        except Exception as e:
            logger.error(
                f"Failed to process enhanced turn for session {session_id}: {e}"
            )
            # Fallback to classic processing
            return await self._process_classic_turn(session_id, player_input, choice_id)

   # =========================================================================
    # P1 Subsystem Handlers
    # =========================================================================

    async def _handle_combat(
        self,
        session_id: str,
        session,
        narrative: str,
        context_memory,
        choice_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Detect combat in narrative and resolve it using CombatSystem."""
        if not COMBAT_AVAILABLE or not self.combat_system:
            return None

        # Keywords that signal a combat encounter
        combat_keywords = [
            "戰鬥", "battle", "combat", "attack", "攻擊", "enemy", "敵人",
            "monster", "怪物", "boss", "fight", "格鬥", "對戰", "vs",
            "warrior", "war", "warfare", "dungeon", "demon", "devil",
            "dragon", "goblin", "orc", "zombie", "skeleton", "undead",
            "wolf", "bear", "slime", "moblin", "bandit", "thief",
        ]
        narrative_lower = narrative.lower()
        has_combat = any(kw in narrative_lower for kw in combat_keywords)

        # Also check flags
        flags = getattr(session.current_state, "flags", {}) or {}
        if flags.get("in_combat") or flags.get("combat_active"):
            has_combat = True

        if not has_combat:
            return None

        logger.info("Combat detected in narrative, initializing combat session...")

        import uuid
        from .stats import CharacterStats

        # Build player combatant
        player_stats_dict = session.stats.to_dict() if hasattr(session.stats, "to_dict") else {}
        player_stats = CharacterStats.from_dict(player_stats_dict)

        # Extract enemy info from narrative (simple heuristic)
        enemy_name = "未知敵人"
        enemy_hp = 50
        enemy_attack = 8
        enemy_defense = 5
        # Try to parse from narrative
        import re
        # Look for patterns like "敵人 HP: XX" or "Monster (HP: XX)"
        hp_match = re.search(r'(?:HP|hp|生命)[：:]?\s*(\d+)', narrative)
        if hp_match:
            enemy_hp = int(hp_match.group(1))
        atk_match = re.search(r'(?:ATK|atk|攻擊)[：:]?\s*(\d+)', narrative)
        if atk_match:
            enemy_attack = int(atk_match.group(1))

        # Generate a meaningful enemy name from context
        scene_id = getattr(context_memory, "current_scene_id", "unknown") or "unknown"
        enemy_name = f"野獸_{scene_id}" if scene_id != "unknown" else "野生怪物"

        enemy_stats = CharacterStats(
            combat__hp=enemy_hp,
            combat__max_hp=enemy_hp,
            combat__attack=enemy_attack,
            combat__defense=enemy_defense,
            combat__speed=5,
        )

        # Use fallback simple stats
        from .stats import CombatStats
        enemy_combat = CombatStats(
            attack=enemy_attack, defense=enemy_defense, speed=5,
            hit_rate=0.80, crit_rate=0.05, evasion=0.10,
            hp=enemy_hp, max_hp=enemy_hp, mp=20, max_mp=20,
        )
        enemy_stats = CharacterStats(combat=enemy_combat, is_alive=True)

        player_combat = player_stats.combat if player_stats.combat else CombatStats()

        player_combatant = Combatant(
            character_id="player",
            name=session.player_name or "冒險者",
            stats=CharacterStats(combat=player_combat, is_alive=True),
        )
        enemy_combatant = Combatant(
            character_id="enemy_1",
            name=enemy_name,
            stats=enemy_stats,
        )

        session_obj = CombatSession(
            session_id=f"combat_{uuid.uuid4().hex[:8]}",
            player=player_combatant,
            enemies=[enemy_combatant],
        )

        # Roll initiative
        order = self.combat_system.roll_initiative(session_obj)

        # Auto-resolve: 1-3 rounds
        rounds = min(3, 1 + (enemy_hp // 50))
        combat_log = []
        total_xp = 0
        for r in range(rounds):
            if not session_obj.is_active():
                break
            # Player action: attack
            player_actions = [(CombatAction.ATTACK, None)]
            results = self.combat_system.resolve_combat(session_obj, player_actions)
            for res in results:
                combat_log.append(res.message)
                total_xp += res.xp_gained

        result = session_obj.result
        victory = (result == CombatResult.VICTORY)
        defeat = (result == CombatResult.DEFEAT)

        # Calculate stat changes
        player_hp_change = player_combatant.stats.combat.hp - player_combat.max_hp
        stat_changes = {}
        if player_hp_change != 0:
            stat_changes["hp"] = player_hp_change

        return {
            "type": "combat",
            "session_id": session_obj.session_id,
            "result": result.value if result else "unknown",
            "enemy_name": enemy_name,
            "enemy_hp_initial": enemy_hp,
            "player_hp_after": player_combatant.stats.combat.hp,
            "player_hp_before": player_combat.max_hp,
            "stat_changes": stat_changes,
            "xp_gained": [total_xp] if total_xp > 0 else [],
            "initiative_order": order,
            "turns": session_obj.turn_number,
            "log": combat_log,
            "victory": victory,
            "defeat": defeat,
        }

    async def _check_achievements(
        self,
        session_id: str,
        session,
        context_memory,
        narrative: str,
    ) -> List[Dict[str, Any]]:
        """Check and trigger achievements based on current game state."""
        if not ACHIEVEMENT_AVAILABLE or not self.achievement_manager:
            return []

        unlocked = []

        # Track stats
        turn_count = session.turn_count
        self.achievement_manager.track("turns", turn_count)
        self.achievement_manager.track("scenes_visited", len(context_memory.scene_sequence))

        # Track flags
        flags = getattr(session.current_state, "flags", {}) or {}
        for flag_name, flag_val in flags.items():
            if flag_val:
                self.achievement_manager.track(f"flag_{flag_name}")

        # Track inventory items
        if hasattr(session, "inventory") and session.inventory:
            self.achievement_manager.track("items_collected", len(session.inventory))

        # Check for combat-related achievements
        narrative_lower = narrative.lower()
        if any(kw in narrative_lower for kw in ["戰鬥", "battle", "combat", "attack", "攻擊"]):
            self.achievement_manager.track("combats_fought")

        if any(kw in narrative_lower for kw in ["victory", "勝利", "win", "勝"]):
            self.achievement_manager.track("victories")

        # Check all achievements
        for achievement in self.achievement_manager.check_all():
            unlocked.append({
                "id": achievement.achievement_id,
                "name": achievement.name,
                "description": achievement.description,
                "condition_type": achievement.condition_type,
                "unlocked_at": session.turn_count,
            })

        return unlocked

    async def _handle_economy(
        self,
        session_id: str,
        session,
        narrative: str,
        combat_result: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Handle economy events: gold from combat, shop visits, quests."""
        if not ECONOMY_AVAILABLE or not self.economy_manager:
            return None

        # Initialize gold if not present
        if not hasattr(session, "gold"):
            session.gold = 100  # Starting gold

        economy_log = []

        # Gold from combat victory
        if combat_result and combat_result.get("victory"):
            gold_reward = 10 + random.randint(5, 20)
            old_gold = session.gold
            session.gold, msg = self.economy_manager.earn_gold(
                gold_reward, old_gold, "Combat victory"
            )
            economy_log.append({
                "event": "combat_reward",
                "gold_earned": gold_reward,
                "new_total": session.gold,
                "message": msg,
            })

        # Detect shop visits
        narrative_lower = narrative.lower()
        shop_keywords = ["shop", "store", "商店", "market", "市集", "buy", "購買", "sell", "出售"]
        if any(kw in narrative_lower for kw in shop_keywords):
            economy_log.append({
                "event": "shop_visit",
                "available_gold": session.gold,
                "message": "偵測到商店場景，玩家可以購買物品",
            })

        # Detect quest completion
        quest_keywords = ["quest", "任務", "mission", "完成", "complete", "reward", "獎勵", "報酬"]
        if any(kw in narrative_lower for kw in quest_keywords):
            quest_gold = 20 + random.randint(10, 30)
            old_gold = session.gold
            session.gold, msg = self.economy_manager.earn_gold(
                quest_gold, old_gold, "Quest completion"
            )
            economy_log.append({
                "event": "quest_reward",
                "gold_earned": quest_gold,
                "new_total": session.gold,
                "message": msg,
            })

        if not economy_log:
            return None

        return {
            "gold": session.gold,
            "events": economy_log,
        }

    async def _handle_relationships(
        self,
        session_id: str,
        session,
        context_memory,
        narrative: str,
        player_input: str,
    ) -> List[Dict[str, Any]]:
        """Update NPC relationships based on player actions and dialogue."""
        if not RELATIONSHIP_AVAILABLE or not self.relationship_manager:
            return []

        updates = []

        # Extract NPC mentions from narrative
        import re
        # Look for quoted dialogue or NPC names
        npc_names = re.findall(r'["「](.+?)["」]', narrative)

        # Check player input for social cues
        positive_cues = ["help", "幫助", "kind", "善良", "thank", "感謝", "friend", "朋友",
                         "trust", "信任", "love", "愛", "support", "支援", "encourage", "鼓勵"]
        negative_cues = ["attack", "攻擊", "hate", "恨", "lie", "騙", "threat", "威脅",
                         "ignore", "無視", "betray", "背叛", "insult", "侮辱"]

        for npc_name in npc_names[:3]:  # Limit to top 3 mentioned NPCs
            if not npc_name or len(npc_name) < 1:
                continue

            # Ensure NPC exists in relationship manager
            if not self.relationship_manager.get(npc_name):
                self.relationship_manager.add_npc(
                    character_id=npc_name,
                    name=npc_name,
                    initial_affection=50,
                )

            # Determine affection change
            affection_delta = 0
            if any(cue in player_input.lower() for cue in positive_cues):
                affection_delta = random.randint(2, 5)
            elif any(cue in player_input.lower() for cue in negative_cues):
                affection_delta = -random.randint(2, 5)

            if affection_delta != 0:
                old_rel = self.relationship_manager.get(npc_name)
                new_affection = self.relationship_manager.modify_affection(
                    npc_name, affection_delta, "player_choice"
                )
                updates.append({
                    "npc": npc_name,
                    "affection_change": affection_delta,
                    "old_affection": old_rel.affection if old_rel else 0,
                    "new_affection": new_affection,
                    "tier": self.relationship_manager.get_tier(npc_name).value
                           if self.relationship_manager.get_tier(npc_name)
                           else "stranger",
                })

        return updates

    async def _handle_map_exploration(
        self,
        session_id: str,
        session,
        context_memory,
        narrative: str,
    ) -> Dict[str, Any]:
        """Handle map exploration events: new locations, discoveries."""
        if not MAP_AVAILABLE or not self.map_system:
            return {}

        result = {
            "current_location": None,
            "new_discoveries": [],
            "available_locations": [],
        }

        # Get or create current location from scene_id
        scene_id = getattr(context_memory, "current_scene_id", None)
        if scene_id:
            location = self.map_system.get_location(scene_id)
            if not location:
                # Create new location
                scene_info = context_memory.get_current_scene()
                scene_name = scene_info.scene_name if scene_info else scene_id
                new_loc = Location(
                    location_id=scene_id,
                    name=scene_name,
                    description=narrative[:100] if narrative else "",
                    explored=False,
                )
                self.map_system.add_location(new_loc)
                self.map_system.set_current_location(scene_id)

                result["new_discoveries"].append({
                    "location_id": scene_id,
                    "name": scene_name,
                    "type": "new",
                    "description": f"發現了新地點：{scene_name}",
                })
            else:
                # Mark as explored
                if not location.explored:
                    location.explored = True
                    result["new_discoveries"].append({
                        "location_id": scene_id,
                        "name": location.name,
                        "type": "explored",
                        "description": f"探索了：{location.name}",
                    })

            result["current_location"] = {
                "id": scene_id,
                "name": location.name if location else scene_id,
                "explored": location.explored if location else True,
            }

            # List nearby unexplored locations
            all_locs = self.map_system.get_available_locations()
            result["available_locations"] = [
                {"id": l.location_id, "name": l.name, "explored": l.explored}
                for l in all_locs[:10]  # Limit to 10
            ]

        # Detect travel keywords
        narrative_lower = narrative.lower()
        travel_keywords = ["travel", "traveling", "移動", "前往", "move", "go", "walk",
                           "run", "explore", "探索", "venture", "discover", "發現"]
        if any(kw in narrative_lower for kw in travel_keywords):
            self.map_system.set_current_location(scene_id) if scene_id else None

        return result

    async def _process_classic_turn(
        self, session_id: str, player_input: str, choice_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a player turn and generate response"""
        session = self.get_session(session_id)
        self._apply_runtime_preset_llm(session)
        persona = self.persona_manager.get_persona(session.persona_id)

        if not persona:
            raise GameError(f"Persona not found: {session.persona_id}")

        try:
            # Handle choice execution if provided
            choice_result = None
            if choice_id:
                try:
                    choice_result = await self._execute_choice(session, choice_id)
                except InvalidChoiceError:
                    logger.debug(f"Classic mode: dynamic choice accepted: {choice_id}")
                    choice_result = {
                        "success": True,
                        "choice_id": choice_id,
                        "player_input": player_input,
                        "consequences": {},
                    }

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

            # Record turn in memory manager
            try:
                memory_manager = get_memory_manager(session_id)
                await memory_manager.record_turn(
                    turn_number=session.turn_count,
                    player_input=player_input,
                    narrative_response=narrative_response,
                    scene_id=session.current_state.scene_id,
                    choices_made=[choice_id] if choice_id else [],
                    flags_changed={},  # TODO: Track flag changes
                    stats_changed={}   # TODO: Track stat changes
                )
            except Exception as e:
                logger.warning(f"Failed to record turn in memory: {e}")

            # Save session
            self._save_session(session)

            # P1 subsystems (not active in classic mode)
            combat_result = None
            achievements_unlocked = []
            economy_result = None
            relationship_updates = []
            map_updates = {}

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
                # P1 subsystems
                "combat_result": combat_result,
                "achievements_unlocked": achievements_unlocked,
                "economy": economy_result,
                "relationship_updates": relationship_updates,
                "map_updates": map_updates,
            }

            if choice_result:
                response["choice_result"] = choice_result

            logger.info(
                f"Processed turn for session {session_id}, turn {session.turn_count}"
            )

            # ── Context compression (P2.3) ──
            try:
                from core.story.context_compressor import ContextCompressor
                if session_id not in self.context_compressors:
                    # Wire the LLM callback for summaries
                    async def _llm_summarize(prompt):
                        msgs = [{"role": "user", "content": prompt}]
                        inner = getattr(self.narrative_generator.llm, '_llm', self.narrative_generator.llm)
                        if inner:
                            r = inner.chat([type('M',(),{"role":"user","content":prompt})()])
                            return r.content if hasattr(r, 'content') else str(r)
                        return prompt[:200]
                    self.context_compressors[session_id] = ContextCompressor(
                        recent_turns=5,
                        compress_interval=10,
                        llm_callback=_llm_summarize,
                    )
                compressor = self.context_compressors[session_id]
                compressor.add_turn({
                    "turn_number": session.turn_count,
                    "player_input": player_input,
                    "narrative": narrative_response,
                    "combat_result": combat_result,
                    "inventory_add": [],
                    "flags_triggered": [],
                    "achievements_unlocked": achievements_unlocked,
                })
                if compressor.should_compress():
                    await compressor.compress_old_turns()
            except Exception as _exc:
                logger.debug(f"Context compression skipped: {_exc}")
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
            # Accept dynamic choices (e.g. left_path, right_path) as valid
            # when they're not in the static template list
            logger.debug(f"Dynamic choice accepted: {choice_id}")
            return {
                "success": True,
                "choice_id": choice_id,
                "player_input": player_input,
                "consequences": {},
            }

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
        
        # Add to narrative memory - CRITICAL for context continuity
        context_memory.add_narrative_memory({
            "type": "story_turn",
            "player_input": player_input,
            "narrative": narrative_result["main_narrative"],
            "choice_made": choice_result.get("choice_executed") if choice_result else None
        })

        # Update scene_sequence so the narrative generator knows we've progressed
        scene_changes = narrative_result.get("scene_changes", {})
        new_scene_id = scene_changes.get("next_scene_id")
        if not new_scene_id and current_scene:
            new_scene_id = current_scene.scene_id
        if new_scene_id:
            if context_memory.scene_sequence and context_memory.scene_sequence[-1] == new_scene_id:
                # Scene hasn't changed — still append to show turn progression
                pass
            context_memory.scene_sequence.append(new_scene_id)
            # Keep bounded
            if len(context_memory.scene_sequence) > 50:
                context_memory.scene_sequence = context_memory.scene_sequence[-50:]
            # Update current_scene_id reference
            try:
                context_memory.current_scene_id = new_scene_id
            except Exception:
                pass

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

    def _generate_llm_choices_sync(
        self,
        narrative_text: str,
        context_memory: StoryContextMemory,
        session: GameSession,
        max_retries: int = 2,
    ) -> List[Dict[str, str]]:
        """
        Use LLM to generate 4 contextually-appropriate choices.
        SYNC method — must be called from a thread (not async context) to avoid event loop conflicts.
        Falls back to empty list on failure after max_retries attempts.
        """
        import json
        import re

        for attempt in range(1, max_retries + 1):
            try:
                from core.llm.base import ChatMessage

                # Use the narrative_generator's LLM (already configured with llama.cpp)
                ng = self.narrative_generator
                if not hasattr(ng, 'llm') or ng.llm is None:
                    raise RuntimeError("Narrative generator has no LLM")
                # ng.llm is LLMAdapter; _llm is the actual LlamaCppServerLLM.
                # DO NOT use get_llm() — that creates a Transformers QwenLLM instead.
                inner_llm = getattr(ng.llm, '_llm', ng.llm)
                if not inner_llm.is_available():
                    inner_llm.load_model()

                # Build turn history summary for context
                recent_turns = getattr(context_memory, "turn_history", []) or []
                history_summary = ""
                for t in recent_turns[-3:]:
                    ai_resp = t.get("ai_response", "")
                    if len(ai_resp) > 150:
                        ai_resp = ai_resp[:150] + "..."
                    history_summary += f"Turn {t.get('turn', '?')}: {ai_resp}\n"

                # Build previously used choice texts to avoid repetition
                prev_choices = session.current_state.available_choices or []
                prev_texts = ""
                for pc in prev_choices:
                    t = pc.get("text", "") if isinstance(pc, dict) else str(pc)
                    if t:
                        prev_texts += f"- {t}\n"

                scene_info = ""
                try:
                    scene = context_memory.get_current_scene()
                    if scene:
                        scene_info = f"場景: {scene.location}"
                except Exception:
                    scene_info = "場景: 當前位置"

                prompt = f"""你是一個互動小說遊戲的劇情引擎。根據以下劇情，生成 4 個玩家下一步的行動選擇。

## 最近劇情摘要
{history_summary if history_summary else "（這是遊戲的開頭）"}

## 當前場景
{scene_info}

## 最新敘事（請仔細閱讀，這是當前的劇情狀態）
{narrative_text}

## 玩家資訊
- 名字: {session.player_name}
- 等級: {session.stats.level}
- 生命: {session.stats.health}
- 道具: {', '.join(session.inventory[:5]) if session.inventory else '無'}

## 上一回合的選擇（不要重複這些）
{prev_texts if prev_texts else "（沒有上一回合的選擇）"}

## 輸出格式
請嚴格以 JSON 陣列輸出，不要有其他文字：
[
  {{"choice_id": "短英文id", "text": "繁體中文描述", "type": "action/exploration/dialogue/combat", "difficulty": "easy/medium/hard"}},
  ...
]

## 嚴格要求
1. 每個 choice_id 用简短的英文（如: enter_cave, talk_merchant）
2. 每個 text 30字以內，繁體中文，描述具體行動
3. 4 個選擇必須明顯不同：一個推進主線、一個探索、一個對話/互動、一個冒險/高風險
4. 選擇必須與【最新敘事】的結尾緊密銜接，不要重複上一回合做過的事
5. 不要使用「繼續前進」、「仔細觀察」、「等待觀察」等通用選項，要具體化"""

                messages = [
                    ChatMessage(role="system", content="你是互動小說的選擇生成器。只輸出 JSON 陣列，不要任何其他文字。嚴格遵守格式。"),
                    ChatMessage(role="user", content=prompt),
                ]

                # Call sync chat() — this method runs in a thread (via asyncio.to_thread) so
                # LlamaCppServerLLM.chat() can create its own event loop without conflicts.
                resp = inner_llm.chat(messages, max_tokens=1024, temperature=0.9)

                content = resp.content if hasattr(resp, 'content') else str(resp)

                # Extract JSON array from response
                json_match = re.search(r'\[[\s\S]*\]', content)
                if json_match:
                    choices = json.loads(json_match.group())
                    if isinstance(choices, list) and len(choices) > 0:
                        seen_ids = set()
                        normalized = []
                        for i, c in enumerate(choices[:4]):
                            cid = c.get("choice_id", f"choice_{i}")
                            if cid in seen_ids:
                                cid = f"{cid}_{i}"
                            seen_ids.add(cid)
                            normalized.append({
                                "choice_id": cid,
                                "text": c.get("text", "繼續行動"),
                                "type": c.get("type", "action"),
                                "difficulty": c.get("difficulty", "medium"),
                            })
                        if normalized:
                            logger.info(f"LLM choice generation succeeded on attempt {attempt}")
                            return normalized

                # If we get here, parsing failed — retry
                logger.warning(
                    f"LLM choice generation returned invalid JSON on attempt {attempt}/{max_retries}"
                )

            except Exception as e:
                logger.warning(
                    f"LLM choice generation error on attempt {attempt}/{max_retries}: {e}"
                )

        logger.error(
            f"All {max_retries} LLM choice generation attempts failed — returning empty for fallback"
        )
        return []

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

        # Find the choice — prefer the persisted available_choices from the
        # current game state so the player picks from the same set the UI showed.
        choice = None
        try:
            raw = getattr(session.current_state, "available_choices", []) or []
            for c in raw:
                if isinstance(c, dict) and c.get("choice_id") == choice_id:
                    # Rebuild a GameChoice from the dict for execute() compat
                    from .choices import GameChoice, ChoiceType, Difficulty

                    choice = GameChoice(
                        choice_id=c["choice_id"],
                        text=c.get("text", ""),
                        choice_type=(
                            ChoiceType(c["type"]) if "type" in c else ChoiceType.ACTION
                        ),
                        difficulty=(
                            Difficulty(c["difficulty"])
                            if "difficulty" in c
                            else Difficulty.NORMAL
                        ),
                        requirements={},
                        consequences={},
                    )
                    break
        except Exception:
            choice = None

        # Fallback: regenerate if not found in session state
        if not choice:
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
                "world_id": getattr(session, "world_id", "default"),
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

    def sync_worldpack_into_session(
        self,
        session_id: str,
        *,
        mode: str = "add_only",
    ) -> Dict[str, Any]:
        """
        Sync the latest WorldPack into an existing session (enhanced mode only).

        This is used by 「世界工作室」在 Story 工作台內即時套用世界變更：
        - 更新 story_context 的 world 基本資訊 / visual
        - 將 WorldPack.characters 合併到 context_memory.characters（可選擇 add_only/merge）
        - 將 WorldPack.world_flags（排除常見動態前綴）合併到 session.flags 與 context_memory.world_flags
        """
        if not self.enhanced_mode or session_id not in getattr(self, "context_memories", {}):
            raise GameError("Enhanced mode context not available for this session")

        if mode not in {"add_only", "merge"}:
            raise GameError("Invalid sync mode (expected: add_only | merge)")

        session = self.get_session(session_id)
        world_id = str(getattr(session, "world_id", "default") or "default").strip() or "default"

        try:
            from core.worldpacks import get_worldpack_manager

            wpm = get_worldpack_manager()
            worldpack = wpm.get_worldpack(world_id)
        except Exception as exc:  # noqa: BLE001
            raise GameError(f"Failed to load worldpack: {world_id}") from exc

        if worldpack is None:
            raise GameError(f"Worldpack not found: {world_id}")

        context_memory = self.context_memories[session_id]

        # Update UI-friendly world context snapshot
        try:
            session.current_state.story_context["world_name"] = worldpack.name
            session.current_state.story_context["world_description"] = worldpack.description
            session.current_state.story_context["world_visual"] = worldpack.visual.model_dump()
            session.current_state.story_context["agent_profile"] = (
                worldpack.agent_profile.model_dump()
                if getattr(worldpack, "agent_profile", None) is not None
                else {}
            )
            session.current_state.story_context["setting"] = worldpack.setting
            session.current_state.story_context["difficulty"] = worldpack.difficulty
            try:
                preset_id = str(getattr(worldpack, "runtime_preset_id", "") or "").strip()
            except Exception:
                preset_id = ""
            if preset_id:
                session.current_state.story_context["runtime_preset_id"] = preset_id
            else:
                # Keep existing session-level preset; if missing, fall back to catalog default.
                current = session.current_state.story_context.get("runtime_preset_id")
                if not current:
                    try:
                        from core.runtime.catalog import load_runtime_preset_catalog

                        catalog = load_runtime_preset_catalog()
                        fallback = str(catalog.get("default_preset_id") or "").strip()
                        if fallback:
                            session.current_state.story_context["runtime_preset_id"] = fallback
                    except Exception:
                        pass
            session.current_state.story_context["worldpack_updated_at"] = worldpack.updated_at
        except Exception:  # noqa: BLE001
            pass

        # Sync world flags (avoid clobbering common dynamic flags)
        dynamic_prefixes = (
            "quest_",
            "npc_met_",
            "location_discovered_",
            "item_acquired_",
            "event_",
            "achievement_",
        )
        flags_added: List[str] = []
        flags_updated: List[str] = []

        for key, value in (worldpack.world_flags or {}).items():
            flag_key = str(key or "").strip()
            if not flag_key:
                continue
            if any(flag_key.startswith(p) for p in dynamic_prefixes):
                continue

            in_session = flag_key in (session.current_state.flags or {})
            in_context = flag_key in (getattr(context_memory, "world_flags", {}) or {})

            if mode == "add_only" and (in_session or in_context):
                continue

            # Apply to both session.flags and context.world_flags for consistency in UI/prompts
            try:
                session.current_state.flags[flag_key] = bool(value)
            except Exception:  # noqa: BLE001
                pass
            try:
                context_memory.world_flags[flag_key] = bool(value)
            except Exception:  # noqa: BLE001
                pass

            if in_session or in_context:
                flags_updated.append(flag_key)
            else:
                flags_added.append(flag_key)

        # Sync world characters
        added_character_ids: List[str] = []
        updated_character_ids: List[str] = []

        role_map = {
            "npc": CharacterRole.NPC,
            "companion": CharacterRole.COMPANION,
            "antagonist": CharacterRole.ANTAGONIST,
        }

        for c in worldpack.characters or []:
            try:
                character_id = str(getattr(c, "character_id", "") or "").strip()
                if not character_id or character_id in {"player", "narrator"}:
                    continue

                existing = context_memory.get_character(character_id)
                if existing is None:
                    character = GameCharacter(
                        character_id=character_id,
                        name=str(getattr(c, "name", "") or character_id),
                        role=role_map.get(getattr(c, "role", "npc"), CharacterRole.NPC),
                        personality_traits=list(getattr(c, "personality_traits", []) or []),
                        speaking_style=str(getattr(c, "speaking_style", "") or ""),
                        background_story=str(getattr(c, "background_story", "") or ""),
                        motivations=list(getattr(c, "motivations", []) or []),
                        relationships=dict(getattr(c, "relationships", {}) or {}),
                        current_location=getattr(existing, "current_location", "unknown") if existing else "unknown",
                        persona_prompt=str(getattr(c, "persona_prompt", "") or ""),
                        content_restrictions=list(getattr(c, "content_restrictions", []) or []),
                    )
                    context_memory.add_character(character)
                    added_character_ids.append(character_id)
                else:
                    if mode != "merge":
                        continue
                    # Merge persona fields only; keep runtime state (health/mood/dialogue/history)
                    existing.name = str(getattr(c, "name", existing.name) or existing.name)
                    existing.role = role_map.get(getattr(c, "role", "npc"), existing.role)
                    existing.personality_traits = list(getattr(c, "personality_traits", existing.personality_traits) or [])
                    existing.speaking_style = str(getattr(c, "speaking_style", existing.speaking_style) or existing.speaking_style)
                    existing.background_story = str(getattr(c, "background_story", existing.background_story) or existing.background_story)
                    existing.motivations = list(getattr(c, "motivations", existing.motivations) or [])
                    existing.relationships = dict(getattr(c, "relationships", existing.relationships) or {})
                    existing.persona_prompt = str(getattr(c, "persona_prompt", existing.persona_prompt) or existing.persona_prompt)
                    existing.content_restrictions = list(getattr(c, "content_restrictions", existing.content_restrictions) or [])
                    updated_character_ids.append(character_id)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skip world character sync for %s: %s", getattr(c, "character_id", "?"), exc)
                continue

        # Persist
        self.save_session(session)

        return {
            "session_id": session_id,
            "world_id": world_id,
            "mode": mode,
            "flags_added": flags_added,
            "flags_updated": flags_updated,
            "characters_added": added_character_ids,
            "characters_updated": updated_character_ids,
            "worldpack_updated_at": getattr(worldpack, "updated_at", None),
        }

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
            "world_id": getattr(session, "world_id", "default"),
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
