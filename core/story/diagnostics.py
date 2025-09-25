# 檔案：core/story/diagnostics.py
"""
全面的系統監控和診斷工具
"""
import logging
from typing import Dict, Any, List
from datetime import datetime

from .story_system import EnhancedStoryEngine
from .logging import StorySystemLogger


class StorySystemDiagnostics:
    """Comprehensive diagnostics for the story system"""

    def __init__(self, engine: EnhancedStoryEngine):
        self.engine = engine
        self.logger = StorySystemLogger("story_diagnostics")

    def run_full_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive system diagnostics"""
        diagnostics = {
            "timestamp": datetime.now().isoformat(),
            "system_health": {},
            "performance_metrics": {},
            "resource_usage": {},
            "active_sessions": {},
            "recommendations": [],
        }

        # System health checks
        diagnostics["system_health"] = self._check_system_health()

        # Performance metrics
        diagnostics["performance_metrics"] = self._gather_performance_metrics()

        # Resource usage
        diagnostics["resource_usage"] = self._check_resource_usage()

        # Active sessions analysis
        diagnostics["active_sessions"] = self._analyze_active_sessions()

        # Generate recommendations
        diagnostics["recommendations"] = self._generate_recommendations(diagnostics)

        return diagnostics

    def _check_system_health(self) -> Dict[str, Any]:
        """Check overall system health"""
        health = {
            "overall_status": "healthy",
            "component_status": {},
            "error_count": 0,
            "warning_count": 0,
        }

        # Check base engine
        if hasattr(self.engine, "base_engine") and self.engine.base_engine:
            health["component_status"]["base_engine"] = "operational"
        else:
            health["component_status"]["base_engine"] = "failed"
            health["error_count"] += 1

        # Check narrative generator
        if hasattr(self.engine, "base_engine") and hasattr(
            self.engine.base_engine, "narrative_generator"
        ):
            health["component_status"]["narrative_generator"] = "operational"
        else:
            health["component_status"]["narrative_generator"] = "degraded"
            health["warning_count"] += 1

        # Check choice manager
        if hasattr(self.engine, "base_engine") and hasattr(
            self.engine.base_engine, "choice_manager"
        ):
            health["component_status"]["choice_manager"] = "operational"
        else:
            health["component_status"]["choice_manager"] = "failed"
            health["error_count"] += 1

        # Check context memories
        if hasattr(self.engine, "context_memories"):
            health["component_status"]["context_system"] = "operational"
        else:
            health["component_status"]["context_system"] = "failed"
            health["error_count"] += 1

        # Determine overall status
        if health["error_count"] > 0:
            health["overall_status"] = (
                "degraded" if health["error_count"] <= 2 else "critical"
            )
        elif health["warning_count"] > 0:
            health["overall_status"] = "operational_with_warnings"

        return health

    def _gather_performance_metrics(self) -> Dict[str, Any]:
        """Gather system performance metrics"""
        metrics = {
            "total_sessions": 0,
            "active_sessions": 0,
            "total_contexts": 0,
            "average_session_age": 0,
            "memory_efficiency": "unknown",
        }

        try:
            # Count sessions
            if hasattr(self.engine, "base_engine") and hasattr(
                self.engine.base_engine, "sessions"
            ):
                metrics["total_sessions"] = len(self.engine.base_engine.sessions)

                active_sessions = [
                    s for s in self.engine.base_engine.sessions.values() if s.is_active
                ]
                metrics["active_sessions"] = len(active_sessions)

                if active_sessions:
                    total_age = sum(
                        (datetime.now() - s.created_at).total_seconds()
                        for s in active_sessions
                    )
                    metrics["average_session_age"] = total_age / len(active_sessions)

            # Count contexts
            if hasattr(self.engine, "context_memories"):
                metrics["total_contexts"] = len(self.engine.context_memories)

            # Estimate memory efficiency
            context_count = metrics["total_contexts"]
            session_count = metrics["total_sessions"]

            if session_count > 0:
                efficiency_ratio = context_count / session_count
                if efficiency_ratio > 0.8:
                    metrics["memory_efficiency"] = "good"
                elif efficiency_ratio > 0.5:
                    metrics["memory_efficiency"] = "fair"
                else:
                    metrics["memory_efficiency"] = "poor"

        except Exception as e:
            self.logger.log_error("performance_metrics_gathering", e)
            metrics["error"] = str(e)

        return metrics

    def _check_resource_usage(self) -> Dict[str, Any]:
        """Check system resource usage"""
        usage = {
            "memory_contexts": 0,
            "stored_sessions": 0,
            "character_instances": 0,
            "scene_instances": 0,
            "narrative_cache_size": 0,
        }

        try:
            # Count contexts and their contents
            if hasattr(self.engine, "context_memories"):
                usage["memory_contexts"] = len(self.engine.context_memories)

                total_characters = 0
                total_scenes = 0

                for context in self.engine.context_memories.values():
                    if hasattr(context, "characters"):
                        total_characters += len(context.characters)
                    if hasattr(context, "scenes"):
                        total_scenes += len(context.scenes)

                usage["character_instances"] = total_characters
                usage["scene_instances"] = total_scenes

            # Count stored sessions
            if hasattr(self.engine, "base_engine") and hasattr(
                self.engine.base_engine, "sessions"
            ):
                usage["stored_sessions"] = len(self.engine.base_engine.sessions)

            # Check narrative cache
            if (
                hasattr(self.engine, "base_engine")
                and hasattr(self.engine.base_engine, "narrative_generator")
                and hasattr(
                    self.engine.base_engine.narrative_generator, "character_voice_cache"
                )
            ):
                usage["narrative_cache_size"] = len(
                    self.engine.base_engine.narrative_generator.character_voice_cache  # type: ignore
                )

        except Exception as e:
            self.logger.log_error("resource_usage_check", e)
            usage["error"] = str(e)  # type: ignore

        return usage

    def _analyze_active_sessions(self) -> Dict[str, Any]:
        """Analyze active sessions"""
        analysis = {
            "session_details": [],
            "average_turns": 0,
            "most_active_persona": "unknown",
            "session_health": "good",
        }

        try:
            if hasattr(self.engine, "base_engine") and hasattr(
                self.engine.base_engine, "sessions"
            ):
                active_sessions = [
                    s for s in self.engine.base_engine.sessions.values() if s.is_active
                ]

                persona_counts = {}
                total_turns = 0

                for session in active_sessions:
                    session_info = {
                        "session_id": session.session_id,
                        "player_name": session.player_name,
                        "persona_id": session.persona_id,
                        "turn_count": session.turn_count,
                        "age_minutes": (
                            datetime.now() - session.created_at
                        ).total_seconds()
                        / 60,
                    }
                    analysis["session_details"].append(session_info)

                    # Count persona usage
                    persona_counts[session.persona_id] = (
                        persona_counts.get(session.persona_id, 0) + 1
                    )
                    total_turns += session.turn_count

                # Calculate averages
                if active_sessions:
                    analysis["average_turns"] = total_turns / len(active_sessions)

                # Find most popular persona
                if persona_counts:
                    analysis["most_active_persona"] = max(
                        persona_counts.items(), key=lambda x: x[1]
                    )[0]

                # Assess session health
                long_running_sessions = [
                    s
                    for s in active_sessions
                    if (datetime.now() - s.created_at).total_seconds() > 3600
                ]
                if len(long_running_sessions) / max(len(active_sessions), 1) > 0.5:
                    analysis["session_health"] = "many_long_running"

        except Exception as e:
            self.logger.log_error("session_analysis", e)
            analysis["error"] = str(e)

        return analysis

    def _generate_recommendations(self, diagnostics: Dict[str, Any]) -> List[str]:
        """Generate system recommendations based on diagnostics"""
        recommendations = []

        # Health-based recommendations
        system_health = diagnostics.get("system_health", {})
        if system_health.get("overall_status") == "critical":
            recommendations.append("系統狀態嚴重，建議立即重啟或降級到基本模式")
        elif system_health.get("overall_status") == "degraded":
            recommendations.append("系統功能受限，建議檢查失效組件")

        # Performance-based recommendations
        performance = diagnostics.get("performance_metrics", {})
        if performance.get("memory_efficiency") == "poor":
            recommendations.append("記憶體使用效率低，建議清理非活躍的上下文記憶")

        average_age = performance.get("average_session_age", 0)
        if average_age > 7200:  # 2 hours
            recommendations.append("存在長時間運行的會話，建議實施會話清理策略")

        # Resource-based recommendations
        resources = diagnostics.get("resource_usage", {})
        if resources.get("memory_contexts", 0) > 100:
            recommendations.append("上下文記憶數量過多，建議實施清理機制")

        if resources.get("character_instances", 0) > 1000:
            recommendations.append("角色實例數量龐大，建議優化角色管理")

        # Session-based recommendations
        sessions = diagnostics.get("active_sessions", {})
        if sessions.get("session_health") == "many_long_running":
            recommendations.append("過多長時間運行的會話，建議設置會話超時機制")

        if not recommendations:
            recommendations.append("系統運行良好，無需特殊維護")

        return recommendations
