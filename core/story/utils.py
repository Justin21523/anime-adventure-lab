# 檔案：core/story/utils.py (新建檔案)
"""
實用工具函數和系統狀態檢查
"""
from typing import List, Dict, Any
from datetime import datetime

from .factory import create_production_story_engine
from .diagnostics import StorySystemDiagnostics


def get_story_system_status() -> Dict[str, Any]:
    """Get comprehensive story system status"""
    try:
        # Try to create a test engine
        engine, status_info = create_production_story_engine(
            enhanced_mode=True, validate_system=True, fallback_on_error=False
        )

        # Run diagnostics
        diagnostics = StorySystemDiagnostics(engine)
        full_diagnostics = diagnostics.run_full_diagnostics()

        return {
            "system_available": True,
            "initialization_status": status_info,
            "diagnostics": full_diagnostics,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {
            "system_available": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "fallback_available": True,  # Basic engine should always be available
        }


def safe_get_enum_value(enum_obj: Any, default: str = "unknown") -> str:
    """Safely get enum value, handling both enum and string cases"""
    if hasattr(enum_obj, "value"):
        return enum_obj.value
    elif isinstance(enum_obj, str):
        return enum_obj
    else:
        return default


def create_default_scene(location: str = "起始點", scene_id: str = "scene_001"):
    """Create a default scene for initialization"""
    from .story_system import SceneContext, SceneType, SceneMood

    return SceneContext(
        scene_id=scene_id,
        scene_type=SceneType.EXPLORATION,
        title="冒險的開始",
        description=f"你發現自己在{location}，準備開始一段新的冒險旅程。",
        location=location,
        time_of_day="黃昏",
        weather="晴朗",
        atmosphere=SceneMood.MYSTERIOUS,
    )


def validate_story_context(context_memory) -> List[str]:
    """Validate story context and return any issues found"""
    issues = []

    if not context_memory.current_scene:
        issues.append("缺少當前場景")

    if not context_memory.characters:
        issues.append("缺少角色定義")

    if "player" not in context_memory.characters:
        issues.append("缺少玩家角色")

    return issues
