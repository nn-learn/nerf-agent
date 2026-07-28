import asyncio
import json
import sqlite3
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.contracts.events import EventEnvelope


class MediaPersistenceError(ValueError):
    pass


class EventSequenceError(ValueError):
    pass


def reject_binary(value: object, path: str = "payload") -> None:
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise MediaPersistenceError(f"binary media forbidden at {path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            reject_binary(child, f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, str):
        for index, child in enumerate(value):
            reject_binary(child, f"{path}[{index}]")


class EventStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
        async with self._lock:
            with sqlite3.connect(self.database_path) as connection:
                connection.executescript(schema)

    async def append(self, event: EventEnvelope[Any]) -> bool:
        reject_binary(event.payload)
        async with self._lock:
            with sqlite3.connect(self.database_path) as connection:
                return self._append_locked(connection, event)

    async def append_payload(
        self,
        *,
        session_id: str,
        event_type: str,
        payload: dict[str, object],
        turn_id: str = "turn_system",
        trace_id: str = "trace_system",
        cancel_token: str = "ct_system",
        timestamp_ms: int | None = None,
    ) -> EventEnvelope[dict[str, object]]:
        reject_binary(payload)
        async with self._lock:
            with sqlite3.connect(self.database_path) as connection:
                self._ensure_session(connection, session_id)
                seq = self._max_seq(connection, session_id) + 1
                event = EventEnvelope[dict[str, object]](
                    event_id=f"evt_{uuid4().hex}",
                    session_id=session_id,
                    turn_id=turn_id,
                    trace_id=trace_id,
                    seq=seq,
                    type=event_type,
                    timestamp_ms=timestamp_ms or int(time.time() * 1000),
                    cancel_token=cancel_token,
                    payload=payload,
                )
                self._append_locked(connection, event)
                return event

    async def list_session(
        self, session_id: str
    ) -> list[EventEnvelope[dict[str, object]]]:
        async with self._lock:
            with sqlite3.connect(self.database_path) as connection:
                rows = connection.execute(
                    """
                    SELECT event_id, session_id, turn_id, trace_id, seq, type,
                           timestamp_ms, cancel_token, payload_json
                    FROM events
                    WHERE session_id = ?
                    ORDER BY seq ASC
                    """,
                    (session_id,),
                ).fetchall()
        return [
            EventEnvelope[dict[str, object]](
                event_id=row[0],
                session_id=row[1],
                turn_id=row[2],
                trace_id=row[3],
                seq=row[4],
                type=row[5],
                timestamp_ms=row[6],
                cancel_token=row[7],
                payload=json.loads(row[8]),
            )
            for row in rows
        ]

    async def set_consent(self, session_id: str, kind: str, granted: bool) -> None:
        async with self._lock:
            with sqlite3.connect(self.database_path) as connection:
                self._ensure_session(connection, session_id)
                connection.execute(
                    """
                    INSERT INTO consents(session_id, kind, granted, updated_at_ms)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(session_id, kind) DO UPDATE SET
                        granted = excluded.granted,
                        updated_at_ms = excluded.updated_at_ms
                    """,
                    (session_id, kind, int(granted), int(time.time() * 1000)),
                )

    async def is_consent_granted(self, session_id: str, kind: str) -> bool:
        async with self._lock:
            with sqlite3.connect(self.database_path) as connection:
                row = connection.execute(
                    "SELECT granted FROM consents WHERE session_id = ? AND kind = ?",
                    (session_id, kind),
                ).fetchone()
        return bool(row[0]) if row is not None else False

    def _append_locked(
        self, connection: sqlite3.Connection, event: EventEnvelope[Any]
    ) -> bool:
        self._ensure_session(connection, event.session_id)
        duplicate = connection.execute(
            "SELECT 1 FROM events WHERE event_id = ?",
            (event.event_id,),
        ).fetchone()
        if duplicate is not None:
            return False

        max_seq = self._max_seq(connection, event.session_id)
        if event.seq <= max_seq:
            raise EventSequenceError(
                f"event sequence must be greater than {max_seq} for {event.session_id}"
            )

        connection.execute(
            """
            INSERT INTO events(
                event_id, session_id, turn_id, trace_id, seq, type,
                timestamp_ms, cancel_token, payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.session_id,
                event.turn_id,
                event.trace_id,
                event.seq,
                event.type,
                event.timestamp_ms,
                event.cancel_token,
                json.dumps(event.payload, ensure_ascii=False, default=str),
            ),
        )
        connection.execute(
            "INSERT INTO outbox(event_id, created_at_ms) VALUES (?, ?)",
            (event.event_id, event.timestamp_ms),
        )
        return True

    @staticmethod
    def _ensure_session(connection: sqlite3.Connection, session_id: str) -> None:
        connection.execute(
            "INSERT OR IGNORE INTO sessions(session_id, created_at_ms) VALUES (?, ?)",
            (session_id, int(time.time() * 1000)),
        )

    @staticmethod
    def _max_seq(connection: sqlite3.Connection, session_id: str) -> int:
        row = connection.execute(
            "SELECT COALESCE(MAX(seq), 0) FROM events WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row[0])
