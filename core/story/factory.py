# 檔案：core/story/factory.py (新建檔案)
"""
工廠函數和生產環境引擎創建
"""
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datetime import datetime

from .story_system import EnhancedStoryEngine
from .initialization import initialize_complete_story_system
from .logging import StorySystemLogger

logger = logging.getLogger(__name__)


def create_production_story_engine(
    config_dir: Optional[Path] = None,
    cache_root: Optional[str] = None,
    enhanced_mode: bool = True,
    validate_system: bool = True,
    fallback_on_error: bool = True,
) -> Tuple["EnhancedStoryEngine", Dict[str, Any]]:
    """
    Create production-ready story engine with comprehensive error handling

    Returns:
        Tuple of (engine, status_info)
    """
    status_info = {
        "initialization_time": datetime.now().isoformat(),
        "enhanced_mode": enhanced_mode,
        "validation_enabled": validate_system,
        "issues": [],
        "warnings": [],
        "status": "unknown",
    }

    try:
        # Initialize system
        engine, issues = initialize_complete_story_system(
            config_dir, enhanced_mode, validate_system
        )

        status_info["issues"] = issues

        # Determine system status
        if not issues:
            status_info["status"] = "healthy"
        elif len(issues) <= 3:
            status_info["status"] = "operational_with_warnings"
            status_info["warnings"] = issues
        else:
            status_info["status"] = "degraded"

        # Add system capabilities info
        status_info["capabilities"] = {
            "enhanced_narrative": hasattr(engine.base_engine, "narrative_generator"),
            "contextual_choices": hasattr(engine, "context_memories"),
            "character_management": hasattr(engine, "character_managers"),
            "scene_transitions": True,
            "relationship_tracking": enhanced_mode,
            "persistent_sessions": True,
        }

        logger.info(
            f"Production story engine created - Status: {status_info['status']}"
        )
        return engine, status_info

    except Exception as e:
        status_info["status"] = "failed"
        status_info["issues"].append(f"Engine creation failed: {str(e)}")

        if fallback_on_error:
            # Create minimal fallback
            from .engine import StoryEngine

            try:
                fallback_engine = StoryEngine(config_dir, enhanced_mode=False)

                # Create enhanced wrapper
                enhanced_wrapper = EnhancedStoryEngine.__new__(EnhancedStoryEngine)
                enhanced_wrapper.base_engine = fallback_engine
                enhanced_wrapper.context_memories = {}
                enhanced_wrapper.character_managers = {}
                enhanced_wrapper.scene_generators = {}

                status_info["status"] = "fallback_active"
                status_info["warnings"].append(
                    "Using fallback engine due to initialization failure"
                )

                logger.warning(
                    "Using fallback story engine due to initialization failure"
                )
                return enhanced_wrapper, status_info

            except Exception as fallback_error:
                status_info["issues"].append(
                    f"Fallback creation failed: {str(fallback_error)}"
                )
                logger.error(f"Both primary and fallback engine creation failed")
                raise
        else:
            raise


def create_enhanced_story_engine(
    config_dir: Optional[Path] = None, cache_root: Optional[str] = None
) -> EnhancedStoryEngine:
    """
    Factory function to create enhanced story engine
    簡化版本的工廠函數，專注於創建增強引擎
    """
    try:
        logger.info("Creating enhanced story engine...")
        engine = EnhancedStoryEngine(config_dir)
        logger.info("Enhanced story engine created successfully")
        return engine
    except Exception as e:
        logger.error(f"Failed to create enhanced story engine: {e}")
        # Fallback logic
        raise
