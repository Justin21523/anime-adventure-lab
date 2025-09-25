# core/story/__init__.py - 修正導入錯誤和缺失的匯出項目

"""
Story Module - Enhanced Text Adventure Game System
Complete story engine with character management, scene transitions, and narrative generation
"""


from typing import List, Dict, Any, Optional
from pathlib import Path

# Core story components
from .engine import StoryEngine, get_story_engine, create_story_engine
from .game_state import GameSession, GameState, PlayerStats
from .narrative import NarrativeGenerator, StoryContext, EnhancedNarrativeGenerator
from .choices import ChoiceManager, GameChoice, ChoiceType, ChoiceDifficulty
from .persona import PersonaManager, GamePersona, PersonaType, EmotionalState

# Enhanced story system components
from .story_system import (
    GameCharacter,
    CharacterRole,
    CharacterState,
    SceneContext,
    SceneType,
    SceneMood,
    StoryContextMemory,
    ContextualChoice,
    EnhancedNarrativeGenerator as SystemEnhancedNarrativeGenerator,
    EnhancedStoryEngine,
)

# New system components
from .factory import create_enhanced_story_engine, create_production_story_engine
from .initialization import (
    initialize_complete_story_system,
    validate_complete_story_system,
)
from .diagnostics import StorySystemDiagnostics
from .monitoring import StorySystemMonitor
from .logging import StorySystemLogger

from .utils import (
    get_story_system_status,
    safe_get_enum_value,
    create_default_scene,
    validate_story_context,
)

import logging

logger = logging.getLogger(__name__)

# Version info
__version__ = "1.0.0"
__author__ = "Multi-Modal Lab Team"

# 完整的模組導出列表
__all__ = [
    # Core engine classes
    "StoryEngine",
    "EnhancedStoryEngine",
    "get_story_engine",
    "create_story_engine",
    "create_enhanced_story_engine",
    # Game state management
    "GameSession",
    "GameState",
    "PlayerStats",
    # Narrative generation
    "NarrativeGenerator",
    "EnhancedNarrativeGenerator",
    "SystemEnhancedNarrativeGenerator",
    "StoryContext",
    # Choice and interaction system
    "ChoiceManager",
    "GameChoice",
    "ChoiceType",
    "ChoiceDifficulty",
    "ContextualChoice",
    # Character and persona system
    "PersonaManager",
    "GamePersona",
    "PersonaType",
    "EmotionalState",
    "GameCharacter",
    "CharacterRole",
    "CharacterState",
    # Scene management
    "SceneContext",
    "SceneType",
    "SceneMood",
    "StoryContextMemory",
    # Production and factory functions
    "create_production_story_engine",
    "initialize_complete_story_system",
    "validate_complete_story_system",
    # System monitoring and diagnostics
    "StorySystemDiagnostics",
    "StorySystemMonitor",
    "StorySystemLogger",
    # Utility functions
    "get_story_system_status",
    "safe_get_enum_value",
    "create_default_scene",
    "validate_story_context",
    # System information
    "initialize_story_system",
    "get_story_system_info",
]


def initialize_story_system(
    config_dir: Optional[Path] = None,
    enhanced_mode: bool = True,
    enable_monitoring: bool = False,
    validate_on_init: bool = True,
) -> StoryEngine:
    """
    Initialize the complete story system with all components
    這是主要的模組初始化函數，提供簡化的接口

    Args:
        config_dir: Configuration directory path
        enhanced_mode: Whether to use enhanced features
        enable_monitoring: Whether to enable system monitoring
        validate_on_init: Whether to validate system on initialization

    Returns:
        Initialized StoryEngine instance
    """
    logger.info(f"Initializing story system (Enhanced: {enhanced_mode})")

    try:
        if enhanced_mode:
            # Use production engine creation for robustness
            engine, status_info = create_production_story_engine(
                config_dir=config_dir,
                enhanced_mode=enhanced_mode,
                validate_system=validate_on_init,
            )

            # Start monitoring if requested
            if enable_monitoring:
                monitor = StorySystemMonitor(engine)
                monitor.start_monitoring()
                logger.info("System monitoring enabled")

            if status_info["status"] in ["healthy", "operational_with_warnings"]:
                logger.info("Enhanced story system initialized successfully")
                return engine  # type: ignore
            else:
                logger.warning(
                    f"Story system initialized with status: {status_info['status']}"
                )
                return engine  # type: ignore

        else:
            # Use basic story engine
            engine = create_story_engine(config_dir)
            logger.info("Basic story system initialized")
            return engine

    except Exception as e:
        logger.error(f"Story system initialization failed: {e}")
        # Final fallback
        engine = create_story_engine(config_dir)
        logger.warning("Using fallback basic story engine")
        return engine


def get_story_system_info() -> Dict[str, Any]:
    """
    Get comprehensive information about the story system
    提供系統信息的統一接口

    Returns:
        Dictionary containing system information
    """
    return {
        "version": __version__,
        "author": __author__,
        "components": {
            "core_engine": "StoryEngine with classic and enhanced modes",
            "narrative_generation": "Context-aware story generation with LLM integration",
            "character_system": "Dynamic persona management with emotional states",
            "choice_system": "Context-sensitive player choice generation",
            "scene_management": "Enhanced scene transitions and environment tracking",
            "memory_system": "Long-term story context and relationship tracking",
            "monitoring_system": "Real-time diagnostics and performance monitoring",
            "production_tools": "Enterprise-grade deployment and management tools",
        },
        "features": {
            "enhanced_mode": "Advanced character interactions and context memory",
            "classic_mode": "Traditional story engine for compatibility",
            "dynamic_choices": "Context-aware choice generation",
            "emotional_ai": "Character emotional state tracking",
            "relationship_system": "Player-NPC relationship management",
            "scene_transitions": "Intelligent scene and location management",
            "narrative_coherence": "Long-term story continuity",
            "system_monitoring": "Real-time health and performance monitoring",
            "error_recovery": "Automatic fallback and error handling",
            "production_ready": "Enterprise deployment capabilities",
        },
        "system_status": get_story_system_status(),
    }
