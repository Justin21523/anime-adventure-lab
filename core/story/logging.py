# core/story/logging.py (新建檔案)
"""
統一的日誌記錄系統，提供故事系統專用的日誌功能
"""
import logging
from typing import Optional, Dict, Any
from datetime import datetime


class StorySystemLogger:
    """Centralized logging for story system"""

    def __init__(self, name: str = "story_system"):
        self.logger = logging.getLogger(name)

    def log_choice_execution(
        self, session_id: str, choice_id: str, result: Dict[str, Any]
    ):
        """Log choice execution"""
        success = result.get("success", False)
        self.logger.info(
            f"Choice executed - Session: {session_id}, Choice: {choice_id}, Success: {success}"
        )

    def log_scene_transition(self, session_id: str, from_scene: str, to_scene: str):
        """Log scene transition"""
        self.logger.info(
            f"Scene transition - Session: {session_id}, From: {from_scene}, To: {to_scene}"
        )

    def log_character_interaction(
        self, session_id: str, character_id: str, interaction_type: str
    ):
        """Log character interaction"""
        self.logger.debug(
            f"Character interaction - Session: {session_id}, Character: {character_id}, Type: {interaction_type}"
        )

    def log_error(
        self, operation: str, error: Exception, session_id: Optional[str] = None
    ):
        """Log system errors"""
        session_info = f" (Session: {session_id})" if session_id else ""
        self.logger.error(f"Error in {operation}{session_info}: {error}", exc_info=True)
