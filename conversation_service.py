"""
Conversation State Management Service for RevOn AI Assistant.

Provides thread-safe in-memory storage for multi-turn conversations,
supporting session persistence, message pruning, and automatic TTL expiration.
"""

import time
import uuid
import threading
from typing import Any
from langsmith_config import trace_chat

DEFAULT_SESSION_TTL_SECONDS = 7200  # 2 hours
DEFAULT_MAX_TURNS_PER_SESSION = 20  # Last 20 messages (10 user + 10 assistant)


class ConversationSession:
    """Represents a single active conversation session."""

    def __init__(self, conversation_id: str):
        self.conversation_id: str = conversation_id
        self.created_at: float = time.monotonic()
        self.updated_at: float = time.monotonic()
        self.messages: list[dict[str, Any]] = []

    def touch(self) -> None:
        self.updated_at = time.monotonic()

    def is_expired(self, ttl_seconds: float) -> bool:
        return (time.monotonic() - self.updated_at) > ttl_seconds

    def add_message(self, role: str, content: str, sources: list[dict[str, Any]] | None = None) -> None:
        self.touch()
        entry: dict[str, Any] = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
        }
        if sources is not None:
            entry["sources"] = sources
        self.messages.append(entry)

        # Keep messages bounded
        if len(self.messages) > DEFAULT_MAX_TURNS_PER_SESSION:
            self.messages = self.messages[-DEFAULT_MAX_TURNS_PER_SESSION:]
            
    @trace_chat
    def get_history_for_prompt(self, max_messages: int = 10) -> list[dict[str, str]]:
        """Return clean role/content pairs formatted for LLM message history."""
        trimmed = self.messages[-max_messages:]
        return [
            {"role": m["role"], "content": m["content"]}
            for m in trimmed
            if m.get("role") in ("user", "assistant")
        ]


class ConversationManager:
    """Thread-safe manager for conversation sessions."""

    def __init__(self, ttl_seconds: float = DEFAULT_SESSION_TTL_SECONDS):
        self._sessions: dict[str, ConversationSession] = {}
        self._lock = threading.Lock()
        self._ttl_seconds: float = ttl_seconds

    def _purge_expired_locked(self) -> None:
        """Purge expired sessions; caller must hold self._lock."""
        expired_keys = [
            cid for cid, session in self._sessions.items()
            if session.is_expired(self._ttl_seconds)
        ]
        for cid in expired_keys:
            self._sessions.pop(cid, None)

    def get_or_create(self, conversation_id: str | None = None) -> tuple[str, ConversationSession]:
        """Get an existing session by ID or create a new one with a fresh UUID."""
        with self._lock:
            self._purge_expired_locked()

            if conversation_id and conversation_id.strip():
                clean_id = conversation_id.strip()
                session = self._sessions.get(clean_id)
                if session is not None and not session.is_expired(self._ttl_seconds):
                    session.touch()
                    return clean_id, session

            new_id = str(uuid.uuid4())
            new_session = ConversationSession(new_id)
            self._sessions[new_id] = new_session
            return new_id, new_session

    def get(self, conversation_id: str) -> ConversationSession | None:
        """Retrieve an active session if it exists and has not expired."""
        with self._lock:
            session = self._sessions.get(conversation_id)
            if session is None or session.is_expired(self._ttl_seconds):
                return None
            session.touch()
            return session
        
    @trace_chat
    def record_turn(
        self,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
        sources: list[dict[str, Any]] | None = None,
    ) -> None:
        """Record a completed question-and-answer turn."""
        with self._lock:
            session = self._sessions.get(conversation_id)
            if session is None:
                session = ConversationSession(conversation_id)
                self._sessions[conversation_id] = session

            session.add_message(role="user", content=user_message)
            session.add_message(role="assistant", content=assistant_message, sources=sources)

    def clear(self, conversation_id: str) -> bool:
        """Clear a conversation session."""
        with self._lock:
            return self._sessions.pop(conversation_id, None) is not None

    def active_sessions_count(self) -> int:
        """Return the count of unexpired active sessions."""
        with self._lock:
            self._purge_expired_locked()
            return len(self._sessions)


# Global singleton instance for the FastAPI application
conversation_manager = ConversationManager()
