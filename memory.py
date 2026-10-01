"""
memory.py — SQLite-backed conversational memory.

Stores conversation turns separately from document/chunk retrieval data and
returns recent history in the format expected by QueryRewriter.
"""
from __future__ import annotations

import time
import uuid
from typing import Optional

from db import Database


class ConversationMemory:
    """Persist and retrieve short-term conversation history using SQLite."""

    def __init__(self, db: Database, max_history_messages: int = 12) -> None:
        self.db = db
        self.max_history_messages = max(1, int(max_history_messages))

    def create_conversation(self) -> str:
        conversation_id = str(uuid.uuid4())
        now = time.time()
        with self.db.connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, created_at, updated_at) VALUES (?, ?, ?)",
                (conversation_id, now, now),
            )
        return conversation_id

    def ensure_conversation(self, conversation_id: str) -> None:
        now = time.time()
        with self.db.connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO conversations (id, created_at, updated_at) VALUES (?, ?, ?)",
                (conversation_id, now, now),
            )

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported message role: {role}")
        content = (content or "").strip()
        if not content:
            return
        self.ensure_conversation(conversation_id)
        now = time.time()
        with self.db.connect() as conn:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (conversation_id, role, content, now),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )

    def history(self, conversation_id: str, limit: Optional[int] = None) -> list[dict[str, str]]:
        message_limit = max(1, int(limit or self.max_history_messages))
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (conversation_id, message_limit),
            ).fetchall()
        return [
            {"role": row["role"], "content": row["content"]}
            for row in reversed(rows)
        ]

    def clear(self, conversation_id: str) -> None:
        with self.db.connect() as conn:
            conn.execute(
                "DELETE FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (time.time(), conversation_id),
            )

    def delete_conversation(self, conversation_id: str) -> None:
        with self.db.connect() as conn:
            conn.execute(
                "DELETE FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            )
            conn.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,),
            )
