"""
Agent Audit Logger

Logs all Agent actions for security audit and debugging.
Provides queryable action history.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class AgentAction:
    """Single Agent action record"""
    timestamp: str
    session_id: str
    tool_name: str
    params: Dict[str, Any]
    result: Any
    success: bool
    error: Optional[str] = None
    execution_time_ms: float = 0


class AgentAuditLogger:
    """
    Audit logger for Agent actions

    Logs all tool executions with:
    - Timestamp
    - Session ID
    - Tool name and parameters
    - Result and success status
    - Error messages if failed

    Provides:
    - Append-only log file
    - In-memory recent actions
    - Query by session/time/tool
    """

    def __init__(self, log_dir: Optional[Path] = None, max_memory: int = 1000):
        """
        Initialize audit logger

        Args:
            log_dir: Directory for log files (default: outputs/agent_audit_logs)
            max_memory: Maximum actions to keep in memory
        """
        self.log_dir = log_dir or Path("outputs/agent_audit_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.max_memory = max_memory
        self._memory_buffer: List[AgentAction] = []

        # Current log file (daily rotation)
        self._current_log_file: Optional[Path] = None
        self._ensure_log_file()

        logger.info(f"Agent audit logger initialized: {self.log_dir}")

    def _ensure_log_file(self) -> None:
        """Ensure current log file exists (daily rotation)"""
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = self.log_dir / f"agent_audit_{today}.jsonl"

        if log_file != self._current_log_file:
            self._current_log_file = log_file
            if not log_file.exists():
                log_file.touch()
                logger.info(f"Created new audit log file: {log_file}")

    async def log_action(
        self,
        session_id: str,
        tool_name: str,
        params: Dict[str, Any],
        result: Any,
        success: bool,
        error: Optional[str] = None,
        execution_time_ms: float = 0
    ) -> None:
        """
        Log an Agent action

        Args:
            session_id: Story session ID
            tool_name: Tool that was executed
            params: Parameters passed to tool
            result: Result from tool (if successful)
            success: Whether execution succeeded
            error: Error message (if failed)
            execution_time_ms: Execution time in milliseconds
        """
        action = AgentAction(
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            tool_name=tool_name,
            params=self._sanitize_params(params),
            result=self._sanitize_result(result),
            success=success,
            error=error,
            execution_time_ms=execution_time_ms
        )

        # Add to memory buffer
        self._memory_buffer.append(action)
        if len(self._memory_buffer) > self.max_memory:
            self._memory_buffer.pop(0)  # Remove oldest

        # Write to log file
        await self._write_to_file(action)

        # Log to console
        log_level = logging.INFO if success else logging.WARNING
        logger.log(
            log_level,
            f"Agent action: {tool_name} | Session: {session_id} | "
            f"Success: {success} | Time: {execution_time_ms:.1f}ms"
        )

    def _sanitize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize params for logging (remove sensitive data)"""
        # Deep copy and sanitize
        sanitized = {}
        for key, value in params.items():
            # Skip large objects
            if isinstance(value, (bytes, bytearray)):
                sanitized[key] = f"<bytes: {len(value)} bytes>"
            elif isinstance(value, str) and len(value) > 500:
                sanitized[key] = value[:500] + "... (truncated)"
            else:
                sanitized[key] = value
        return sanitized

    def _sanitize_result(self, result: Any) -> Any:
        """Sanitize result for logging"""
        # Convert complex objects to strings
        if result is None:
            return None
        elif isinstance(result, (str, int, float, bool)):
            return result
        elif isinstance(result, dict):
            return {k: str(v)[:200] for k, v in result.items()}
        elif isinstance(result, list):
            return [str(item)[:200] for item in result[:10]]  # First 10 items
        else:
            return str(result)[:200]

    async def _write_to_file(self, action: AgentAction) -> None:
        """Write action to log file (async)"""
        try:
            self._ensure_log_file()

            # Convert to JSON
            action_dict = asdict(action)
            json_line = json.dumps(action_dict, ensure_ascii=False)

            # Append to file
            with open(self._current_log_file, 'a', encoding='utf-8') as f:
                f.write(json_line + '\n')

        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

    def get_recent_actions(self, limit: int = 100) -> List[AgentAction]:
        """Get recent actions from memory buffer"""
        return self._memory_buffer[-limit:]

    def get_actions_by_session(self, session_id: str, limit: int = 100) -> List[AgentAction]:
        """Get actions for specific session from memory"""
        filtered = [
            action for action in self._memory_buffer
            if action.session_id == session_id
        ]
        return filtered[-limit:]

    def get_failed_actions(self, limit: int = 100) -> List[AgentAction]:
        """Get recent failed actions"""
        failed = [
            action for action in self._memory_buffer
            if not action.success
        ]
        return failed[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get audit statistics"""
        if not self._memory_buffer:
            return {
                "total_actions": 0,
                "success_count": 0,
                "failure_count": 0,
                "success_rate": 0.0,
                "tools_used": {},
                "sessions_tracked": 0
            }

        total = len(self._memory_buffer)
        success = sum(1 for a in self._memory_buffer if a.success)
        failed = total - success

        # Count tools
        tools_used = {}
        for action in self._memory_buffer:
            tools_used[action.tool_name] = tools_used.get(action.tool_name, 0) + 1

        # Count unique sessions
        sessions = set(a.session_id for a in self._memory_buffer)

        return {
            "total_actions": total,
            "success_count": success,
            "failure_count": failed,
            "success_rate": (success / total * 100) if total > 0 else 0.0,
            "tools_used": tools_used,
            "sessions_tracked": len(sessions),
            "memory_buffer_size": len(self._memory_buffer),
            "max_memory": self.max_memory
        }

    async def query_log_file(
        self,
        session_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        success: Optional[bool] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000
    ) -> List[AgentAction]:
        """
        Query log files for actions

        Args:
            session_id: Filter by session ID
            tool_name: Filter by tool name
            success: Filter by success status
            start_date: Filter by start date (YYYY-MM-DD)
            end_date: Filter by end date (YYYY-MM-DD)
            limit: Maximum results

        Returns:
            List of matching actions
        """
        results = []

        # Determine which log files to search
        log_files = []
        if start_date or end_date:
            # Search date range
            for log_file in sorted(self.log_dir.glob("agent_audit_*.jsonl")):
                log_files.append(log_file)
        else:
            # Search current file only
            if self._current_log_file and self._current_log_file.exists():
                log_files = [self._current_log_file]

        # Search files
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if len(results) >= limit:
                            break

                        try:
                            action_dict = json.loads(line)
                            action = AgentAction(**action_dict)

                            # Apply filters
                            if session_id and action.session_id != session_id:
                                continue
                            if tool_name and action.tool_name != tool_name:
                                continue
                            if success is not None and action.success != success:
                                continue

                            results.append(action)

                        except json.JSONDecodeError:
                            continue

            except Exception as e:
                logger.warning(f"Failed to read log file {log_file}: {e}")

        return results[-limit:]  # Return most recent


# Singleton instance
_audit_logger: Optional[AgentAuditLogger] = None


def get_audit_logger(log_dir: Optional[Path] = None) -> AgentAuditLogger:
    """Get or create singleton audit logger"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AgentAuditLogger(log_dir)
    return _audit_logger
