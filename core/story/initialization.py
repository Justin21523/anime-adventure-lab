# 檔案：core/story/initialization.py (新建檔案)
"""
系統初始化和驗證功能
"""
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime

from .story_system import EnhancedStoryEngine, GameCharacter, CharacterRole
from .engine import StoryEngine
from .logging import StorySystemLogger
from core.exceptions import StorySystemError

logger = logging.getLogger(__name__)


def initialize_complete_story_system(
    config_dir: Optional[Path] = None,
    enhanced_mode: bool = True,
    validate_on_init: bool = True,
) -> Tuple[EnhancedStoryEngine, List[str]]:
    """
    Initialize complete story system with comprehensive validation

    Returns:
        Tuple of (engine, validation_issues)
    """
    issues = []

    try:
        # Initialize the enhanced story engine
        engine = EnhancedStoryEngine(config_dir)

        if validate_on_init:
            # Run comprehensive system validation
            validation_results = validate_complete_story_system(engine)
            issues.extend(validation_results)

        logger.info(f"Story system initialized (Enhanced: {enhanced_mode})")
        if issues:
            logger.warning(f"Initialization completed with {len(issues)} issues")

        return engine, issues

    except Exception as e:
        logger.error(f"Failed to initialize story system: {e}")
        issues.append(f"Initialization failed: {str(e)}")

        # Return a minimal fallback system
        fallback_engine = StoryEngine(config_dir, enhanced_mode=False)

        # Wrap in enhanced interface
        enhanced_wrapper = EnhancedStoryEngine.__new__(EnhancedStoryEngine)
        enhanced_wrapper.base_engine = fallback_engine
        enhanced_wrapper.context_memories = {}
        enhanced_wrapper.character_managers = {}
        enhanced_wrapper.scene_generators = {}

        return enhanced_wrapper, issues


def validate_complete_story_system(engine: "EnhancedStoryEngine") -> List[str]:
    """Validate complete story system functionality"""
    issues = []

    try:
        # 測試基本引擎功能
        if not hasattr(engine, "base_engine") or not engine.base_engine:
            issues.append("Missing or invalid base engine")
            return issues  # 沒有基本引擎就無法繼續測試

        # 測試上下文記憶
        if not hasattr(engine, "context_memories"):
            issues.append("Missing context memories system")

        # 測試角色管理
        if not hasattr(engine, "character_managers"):
            issues.append("Missing character managers system")

        # 測試會話創建
        try:
            test_session_id = engine.create_session("測試玩家", "default")

            # 驗證會話是否正確創建
            if test_session_id not in engine.base_engine.sessions:
                issues.append("Session creation failed - session not stored")

            # 測試上下文記憶創建
            if (
                hasattr(engine, "context_memories")
                and test_session_id not in engine.context_memories
            ):
                issues.append("Context memory not created for new session")

            # 測試選擇生成
            if test_session_id in engine.context_memories:
                try:
                    context_memory = engine.context_memories[test_session_id]
                    choices = engine._generate_contextual_choices(context_memory)

                    if not choices:
                        issues.append("No contextual choices generated")
                    elif not all(hasattr(choice, "choice_id") for choice in choices):
                        issues.append("Generated choices missing required attributes")

                except Exception as e:
                    issues.append(f"Contextual choice generation failed: {str(e)}")

            # 清理測試會話
            if test_session_id in engine.base_engine.sessions:
                del engine.base_engine.sessions[test_session_id]
            if (
                hasattr(engine, "context_memories")
                and test_session_id in engine.context_memories
            ):
                del engine.context_memories[test_session_id]
            if (
                hasattr(engine, "character_managers")
                and test_session_id in engine.character_managers
            ):
                del engine.character_managers[test_session_id]

        except Exception as e:
            issues.append(f"Session management test failed: {str(e)}")

        # 測試敘事生成
        if hasattr(engine, "base_engine") and hasattr(
            engine.base_engine, "narrative_generator"
        ):
            try:
                test_context = {
                    "player_input": "測試行動",
                    "current_location": "測試地點",
                }
                narrative = engine.base_engine.narrative_generator.generate_narrative(
                    test_context
                )

                if not narrative or not isinstance(narrative, str):
                    issues.append("Narrative generation returned invalid result")

            except Exception as e:
                issues.append(f"Narrative generation test failed: {str(e)}")

    except Exception as e:
        issues.append(f"System validation failed: {str(e)}")

    return issues
