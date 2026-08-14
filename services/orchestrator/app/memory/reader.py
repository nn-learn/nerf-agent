import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

from app.memory.models import MemoryMessage, MessageRole

_MESSAGE_EVENT_TYPES = ("transcript.final", "assistant.response.ready")


class EventMessageReader:
    """Keyset-paginated reader over the append-only event log."""

    def __init__(self, database_path: Path, *, batch_size: int = 500) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.database_path = database_path
        self.batch_size = batch_size

    def iter_batches(
        self,
        session_id: str,
        *,
        after_sequence: int = 0,
    ) -> Iterator[list[MemoryMessage]]:
        cursor = after_sequence
        while True:
            with closing(sqlite3.connect(self.database_path)) as connection:
                rows = connection.execute(
                    """
                    SELECT event_id, turn_id, seq, type, timestamp_ms, payload_json
                    FROM events
                    WHERE session_id = ? AND seq > ? AND type IN (?, ?)
                    ORDER BY seq ASC
                    LIMIT ?
                    """,
                    (
                        session_id,
                        cursor,
                        _MESSAGE_EVENT_TYPES[0],
                        _MESSAGE_EVENT_TYPES[1],
                        self.batch_size,
                    ),
                ).fetchall()
            if not rows:
                return
            batch: list[MemoryMessage] = []
            for row in rows:
                cursor = int(row[2])
                payload = json.loads(str(row[5]))
                event_type = str(row[3])
                text = self._message_text(event_type, payload)
                if not text:
                    continue
                batch.append(
                    MemoryMessage(
                        message_id=str(row[0]),
                        turn_id=str(row[1]),
                        role=(
                            MessageRole.USER
                            if event_type == "transcript.final"
                            else MessageRole.ASSISTANT
                        ),
                        text=text,
                        sequence=cursor,
                        timestamp_ms=int(row[4]),
                    )
                )
            if batch:
                yield batch

    def read_session(
        self,
        session_id: str,
        *,
        after_sequence: int = 0,
    ) -> list[MemoryMessage]:
        return [
            message
            for batch in self.iter_batches(
                session_id,
                after_sequence=after_sequence,
            )
            for message in batch
        ]

    @staticmethod
    def _message_text(event_type: str, payload: object) -> str:
        if not isinstance(payload, dict):
            return ""
        key = "text" if event_type == "transcript.final" else "display_text"
        value = payload.get(key)
        return value.strip() if isinstance(value, str) else ""
