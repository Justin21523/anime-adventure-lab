"""Interactive Story domain with lazy public exports.

Importing a single Story module must not initialize training, image-generation,
or local-model stacks. This keeps the API composition root lightweight while
preserving the legacy package-level API.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

__version__ = "1.0.0"
__author__ = "Multi-Modal Lab Team"

_EXPORTS = {
    "StoryEngine": ("core.story.engine", "StoryEngine"),
    "get_story_engine": ("core.story.engine", "get_story_engine"),
    "create_story_engine": ("core.story.engine", "create_story_engine"),
    "GameSession": ("core.story.game_state", "GameSession"),
    "GameState": ("core.story.game_state", "GameState"),
    "PlayerStats": ("core.story.game_state", "PlayerStats"),
    "NarrativeGenerator": ("core.story.narrative", "NarrativeGenerator"),
    "StoryContext": ("core.story.narrative", "StoryContext"),
    "EnhancedNarrativeGenerator": ("core.story.narrative", "EnhancedNarrativeGenerator"),
    "ChoiceManager": ("core.story.choices", "ChoiceManager"),
    "GameChoice": ("core.story.choices", "GameChoice"),
    "ChoiceType": ("core.story.choices", "ChoiceType"),
    "ChoiceDifficulty": ("core.story.choices", "ChoiceDifficulty"),
    "PersonaManager": ("core.story.persona", "PersonaManager"),
    "GamePersona": ("core.story.persona", "GamePersona"),
    "PersonaType": ("core.story.persona", "PersonaType"),
    "EmotionalState": ("core.story.persona", "EmotionalState"),
    "GameCharacter": ("core.story.story_system", "GameCharacter"),
    "CharacterRole": ("core.story.story_system", "CharacterRole"),
    "CharacterState": ("core.story.story_system", "CharacterState"),
    "SceneContext": ("core.story.story_system", "SceneContext"),
    "SceneType": ("core.story.story_system", "SceneType"),
    "SceneMood": ("core.story.story_system", "SceneMood"),
    "StoryContextMemory": ("core.story.story_system", "StoryContextMemory"),
    "ContextualChoice": ("core.story.story_system", "ContextualChoice"),
    "SystemEnhancedNarrativeGenerator": (
        "core.story.story_system",
        "EnhancedNarrativeGenerator",
    ),
    "EnhancedStoryEngine": ("core.story.story_system", "EnhancedStoryEngine"),
    "create_enhanced_story_engine": ("core.story.factory", "create_enhanced_story_engine"),
    "create_production_story_engine": ("core.story.factory", "create_production_story_engine"),
    "initialize_complete_story_system": (
        "core.story.initialization",
        "initialize_complete_story_system",
    ),
    "validate_complete_story_system": (
        "core.story.initialization",
        "validate_complete_story_system",
    ),
    "StorySystemDiagnostics": ("core.story.diagnostics", "StorySystemDiagnostics"),
    "StorySystemMonitor": ("core.story.monitoring", "StorySystemMonitor"),
    "StorySystemLogger": ("core.story.logging", "StorySystemLogger"),
    "get_story_system_status": ("core.story.utils", "get_story_system_status"),
    "safe_get_enum_value": ("core.story.utils", "safe_get_enum_value"),
    "create_default_scene": ("core.story.utils", "create_default_scene"),
    "validate_story_context": ("core.story.utils", "validate_story_context"),
}

__all__ = [*_EXPORTS, "initialize_story_system", "get_story_system_info"]


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def initialize_story_system(
    config_dir: Path | None = None,
    enhanced_mode: bool = True,
    enable_monitoring: bool = False,
    validate_on_init: bool = True,
):
    if enhanced_mode:
        factory = __getattr__("create_production_story_engine")
        engine, _status = factory(
            config_dir=config_dir,
            enhanced_mode=True,
            validate_system=validate_on_init,
        )
        if enable_monitoring:
            __getattr__("StorySystemMonitor")(engine).start_monitoring()
        return engine
    return __getattr__("create_story_engine")(config_dir)


def get_story_system_info() -> dict[str, Any]:
    return {
        "version": __version__,
        "author": __author__,
        "status": __getattr__("get_story_system_status")(),
        "architecture": "story-first",
    }
