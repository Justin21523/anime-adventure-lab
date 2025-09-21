# core/llm/chat_manager.py (Fixed Version)
"""
Chat Manager
Handles multi-turn conversations, session management, and conversation history
"""

import json
import uuid
import logging
import tiktoken
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path

from .base import ChatMessage, LLMResponse
from ..config import get_config
from ..shared_cache import get_shared_cache
from ..exceptions import (
    ValidationError,
    SessionNotFoundError,
    ContextLengthExceededError,
)

logger = logging.getLogger(__name__)


@dataclass
class ChatSession:
    """Chat session data structure"""

    session_id: str
    created_at: datetime
    last_updated: datetime
    messages: List[ChatMessage]
    metadata: Dict[str, Any]
    max_history: int = 50
    system_prompt: Optional[str] = None

    def __init__(
        self,
        session_id: str,
        system_prompt: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.session_id = session_id
        self.system_prompt = system_prompt
        self.metadata = metadata or {}
        self.messages: List[ChatMessage] = []
        self.created_at = datetime.now()
        self.last_updated = datetime.now()

        # Add system message if provided
        if system_prompt:
            self.messages.append(ChatMessage(role="system", content=system_prompt))

    def add_message(
        self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a message to the session"""
        if role not in ["user", "assistant", "system"]:
            raise ValidationError(
                "role", role, "Must be one of: user, assistant, system"
            )

        if not content.strip():
            raise ValidationError("content", content, "Message content cannot be empty")

        message = ChatMessage(role=role, content=content, metadata=metadata)
        self.messages.append(message)
        self.last_updated = datetime.now()

        logger.debug(f"Added {role} message to session {self.session_id}")

        # Trim history if needed
        if len(self.messages) > self.max_history:
            # Keep system message + last N-1 messages
            system_messages = [msg for msg in self.messages if msg.role == "system"]
            other_messages = [msg for msg in self.messages if msg.role != "system"]

            # Keep most recent messages
            recent_messages = other_messages[
                -(self.max_history - len(system_messages)) :
            ]
            self.messages = system_messages + recent_messages

    def get_messages(
        self, include_system: bool = True, last_n: Optional[int] = None
    ) -> List[ChatMessage]:
        """Get messages from the session"""
        messages = self.messages.copy()

        if not include_system:
            messages = [msg for msg in messages if msg.role != "system"]

        if last_n is not None:
            messages = messages[-last_n:]

        return messages

    def get_message_count(self) -> int:
        """Get total message count"""
        return len(self.messages)

    def get_conversation_length(self) -> int:
        """Get total character length of conversation"""
        return sum(len(msg.content) for msg in self.messages)

    def clear_messages(self, keep_system: bool = True) -> None:
        """Clear all messages, optionally keeping system message"""
        if keep_system and self.messages and self.messages[0].role == "system":
            self.messages = [self.messages[0]]
        else:
            self.messages = []

        self.updated_at = datetime.utcnow()
        logger.info(f"Cleared messages for session {self.session_id}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "metadata": self.metadata,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "metadata": msg.metadata,
                }
                for msg in self.messages
            ],
            "created_at": self.created_at.isoformat(),
            "max_history": self.max_history,
            "system_prompt": self.system_prompt,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatSession":
        """Create session from dictionary"""
        session = cls(
            session_id=data["session_id"],
            system_prompt=data.get("system_prompt", ""),
            metadata=data.get("metadata", {}),
        )

        # Restore timestamps
        if "created_at" in data:
            session.created_at = datetime.fromisoformat(data["created_at"])
        if "last_updated" in data:
            session.last_updated = datetime.fromisoformat(data["last_updated"])

        # Restore messages (skip system message if it was auto-added)
        session.messages = []
        for msg_data in data.get("messages", []):
            message = ChatMessage(
                role=msg_data["role"],
                content=msg_data["content"],
                metadata=msg_data.get("metadata"),
            )
            session.messages.append(message)

        return session


class ChatManager:
    """Manages chat sessions and conversation history"""

    def __init__(self, persist_sessions: bool = True):
        self.config = get_config()
        self.cache = get_shared_cache()
        self._sessions: Dict[str, ChatSession] = {}
        self.persist_sessions = persist_sessions
        self._session_cleanup_interval = timedelta(hours=24)

        # Load existing sessions
        self._load_sessions()

        # Setup persistence directory
        if persist_sessions:
            self.cache = get_shared_cache()
            self.sessions_dir = Path(self.cache.cache_root) / "chat_sessions"
            self.sessions_dir.mkdir(parents=True, exist_ok=True)

            # Load existing sessions
            self._load_persisted_sessions()

        logger.info(f"ChatManager initialized (persist={persist_sessions})")

    def create_session(
        self,
        system_prompt: Optional[str] = None,
        max_history: int = 50,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Create new chat session

        Args:
            system_prompt: System message for the session
            max_history: Maximum number of messages to keep
            metadata: Additional session metadata

        Returns:
            Session ID
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        if session_id in self._sessions:
            raise ValidationError("session_id", session_id, "Session ID already exists")

        session = ChatSession(
            session_id=session_id,
            system_prompt=system_prompt,  # type: ignore
            metadata=metadata,
        )

        self._sessions[session_id] = session

        # Persist if enabled
        if self.persist_sessions:
            self._save_session(session_id)

        logger.info(f"Created chat session: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> ChatSession:
        """Get chat session by ID"""
        if session_id not in self._sessions:
            # Try to load from disk
            if self.persist_sessions and self._load_session(session_id):
                return self._sessions[session_id]

            raise SessionNotFoundError(session_id)

        return self._sessions[session_id]

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add message to session"""
        session = self.get_session(session_id)

        # Validate message
        self._validate_message(role, content)

        session.add_message(role, content, metadata)

        # Persist if enabled
        if self.persist_sessions:
            self._save_session(session_id)

        logger.debug(f"Added message to session {session_id}: {role}")

    def add_response(
        self,
        session_id: str,
        response: LLMResponse,
    ) -> None:
        """Add LLM response to session"""
        self.add_message(
            session_id=session_id,
            role="assistant",
            content=response.content,
            metadata={
                "model_name": response.model_name,
                "usage": response.usage,
                "timestamp": datetime.now().isoformat(),
                **(response.metadata or {}),
            },
        )

    def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
        include_system: bool = True,
    ) -> List[ChatMessage]:
        """Get messages from session"""
        session = self.get_session(session_id)
        messages = session.messages.copy()

        if not include_system:
            messages = [msg for msg in messages if msg.role != "system"]

        if limit:
            messages = messages[-limit:]

        return messages

    def get_conversation_context(
        self,
        session_id: str,
        max_tokens: Optional[int] = None,
    ) -> List[ChatMessage]:
        """
        Get conversation context for LLM, respecting token limits

        Args:
            session_id: Session ID
            max_tokens: Maximum token count (approximate)

        Returns:
            List of messages that fit in context
        """
        session = self.get_session(session_id)
        messages = session.messages.copy()

        if not max_tokens:
            return messages

        # Rough token estimation (1 token ≈ 4 characters)
        total_chars = 0
        max_chars = max_tokens * 4

        # Always include system messages
        context_messages = [msg for msg in messages if msg.role == "system"]
        system_chars = sum(len(msg.content) for msg in context_messages)
        total_chars += system_chars

        # Add other messages from most recent
        other_messages = [msg for msg in messages if msg.role != "system"]
        for message in reversed(other_messages):
            message_chars = len(message.content)
            if total_chars + message_chars > max_chars:
                break
            context_messages.insert(
                -len([m for m in context_messages if m.role == "system"]), message
            )
            total_chars += message_chars

        # Maintain chronological order
        context_messages.sort(key=lambda x: messages.index(x))

        if len(context_messages) < len(messages):
            logger.info(
                f"Context truncated: {len(context_messages)}/{len(messages)} messages, "
                f"{total_chars}/{max_chars} chars"
            )

        return context_messages

    def update_session_metadata(
        self,
        session_id: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Update session metadata"""
        session = self.get_session(session_id)
        session.metadata.update(metadata)
        session.last_updated = datetime.now()
        self._save_session(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete chat session"""
        if session_id in self._sessions:
            del self._sessions[session_id]

            # Remove from cache
            session_file = self._get_session_file(session_id)
            if session_file.exists():
                session_file.unlink()

            logger.info(f"Deleted chat session: {session_id}")
            return True

        return False

    def list_sessions(
        self,
        limit: Optional[int] = None,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """List all sessions"""
        sessions = []

        for session in self._sessions.values():
            session_info = {
                "session_id": session.session_id,
                "created_at": session.created_at.isoformat(),
                "last_updated": session.last_updated.isoformat(),
                "message_count": session.get_message_count(),
                "conversation_length": session.get_conversation_length(),
            }

            if include_metadata:
                session_info["metadata"] = session.metadata
                session_info["system_prompt"] = session.system_prompt

            sessions.append(session_info)

        # Sort by last updated (most recent first)
        sessions.sort(key=lambda x: x["last_updated"], reverse=True)

        if limit:
            sessions = sessions[:limit]

        return sessions

    def cleanup_old_sessions(self, max_age: Optional[timedelta] = None) -> int:
        """Clean up old sessions"""
        if max_age is None:
            max_age = self._session_cleanup_interval

        cutoff_time = datetime.now() - max_age
        old_sessions = [
            sid
            for sid, session in self._sessions.items()
            if session.last_updated < cutoff_time
        ]

        for session_id in old_sessions:
            self.delete_session(session_id)

        logger.info(f"Cleaned up {len(old_sessions)} old sessions")
        return len(old_sessions)

    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        if not self._sessions:
            return {"total_sessions": 0}

        total_messages = sum(s.get_message_count() for s in self._sessions.values())
        total_chars = sum(s.get_conversation_length() for s in self._sessions.values())

        oldest_session = min(self._sessions.values(), key=lambda s: s.created_at)
        newest_session = max(self._sessions.values(), key=lambda s: s.created_at)

        return {
            "total_sessions": len(self._sessions),
            "total_messages": total_messages,
            "total_conversation_chars": total_chars,
            "avg_messages_per_session": (
                total_messages / len(self._sessions) if self._sessions else 0
            ),
            "oldest_session_age_hours": (
                datetime.now() - oldest_session.created_at
            ).total_seconds()
            / 3600,
            "newest_session_age_hours": (
                datetime.now() - newest_session.created_at
            ).total_seconds()
            / 3600,
        }

    def _validate_message(self, role: str, content: str) -> None:
        """Validate message content"""
        if role not in ["system", "user", "assistant"]:
            raise ValidationError("role", role, "Invalid message role")

        if not content or not content.strip():
            raise ValidationError("content", content, "Message content cannot be empty")

        max_length = self.config.get("chat.max_message_length", 10000)
        if len(content) > max_length:
            raise ValidationError(
                "content",
                len(content),
                f"Message too long: {len(content)} > {max_length}",
            )

    def _get_session_file(self, session_id: str) -> Path:
        """Get session file path"""
        sessions_dir = Path(self.cache.cache_root) / "chat_sessions"
        sessions_dir.mkdir(exist_ok=True)
        return sessions_dir / f"{session_id}.json"

    def _save_session(self, session_id: str) -> None:
        """Save session to disk"""
        try:
            session = self._sessions[session_id]
            session_file = self._get_session_file(session_id)

            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")

    def _load_session(self, session_id: str) -> bool:
        """Load session from disk"""
        try:
            session_file = self._get_session_file(session_id)
            if not session_file.exists():
                return False

            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            session = ChatSession.from_dict(data)
            self._sessions[session_id] = session
            return True

        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return False

    def _load_sessions(self) -> None:
        """Load all sessions from disk at startup"""
        try:
            sessions_dir = Path(self.cache.cache_root) / "chat_sessions"
            if not sessions_dir.exists():
                return

            session_files = list(sessions_dir.glob("*.json"))
            for session_file in session_files:
                session_id = session_file.stem
                self._load_session(session_id)

            logger.info(f"Loaded {len(session_files)} chat sessions from cache")

        except Exception as e:
            logger.error(f"Failed to load sessions: {e}")

    def _load_persisted_sessions(self) -> None:
        """Load all persisted sessions from disk"""
        if not self.sessions_dir.exists():
            return

        loaded_count = 0
        for session_file in self.sessions_dir.glob("*.json"):
            session_id = session_file.stem
            if self._load_session(session_id):
                loaded_count += 1

        logger.info(f"Loaded {loaded_count} persisted chat sessions")

    def cleanup(self) -> None:
        """Clean up resources"""
        logger.info(f"ChatManager cleanup: {len(self._sessions)} sessions")
        self._sessions.clear()


# Global chat manager instance
_chat_manager: Optional[ChatManager] = None


def get_chat_manager(persist_sessions: bool = True) -> ChatManager:
    """Get global chat manager instance"""
    global _chat_manager
    if _chat_manager is None:
        _chat_manager = ChatManager(persist_sessions=persist_sessions)
    return _chat_manager


def reset_chat_manager():
    """Reset the global chat manager (for testing)"""
    global _chat_manager

    if _chat_manager:
        _chat_manager.cleanup()

    _chat_manager = None
    logger.info("Chat manager reset")
