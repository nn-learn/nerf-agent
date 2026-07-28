import sqlite3
from pathlib import Path
from uuid import uuid4

from app.memory.models import MemoryCandidate, MemoryItem, MemoryKind, MemoryState


class MemoryRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        schema = (
            Path(__file__).parents[1] / "events" / "schema.sql"
        ).read_text(encoding="utf-8")
        with sqlite3.connect(self.database_path) as connection:
            connection.executescript(schema)

    def add_active(
        self,
        *,
        user_id: str,
        candidate: MemoryCandidate,
    ) -> MemoryItem:
        item = MemoryItem(
            memory_id=f"memory_{uuid4().hex}",
            user_id=user_id,
            candidate=candidate,
            state=MemoryState.ACTIVE,
        )
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO memory_items(
                    memory_id, user_id, source, contains_sensitive_content,
                    source_turn_id, kind, state, content
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.memory_id,
                    item.user_id,
                    candidate.source,
                    int(candidate.contains_sensitive_content),
                    candidate.source_turn_id,
                    candidate.kind.value,
                    item.state.value,
                    candidate.text,
                ),
            )
        return item

    def revoke(self, memory_id: str) -> None:
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "UPDATE memory_items SET state = ? WHERE memory_id = ?",
                (MemoryState.REVOKED.value, memory_id),
            )

    def list_active(self, user_id: str) -> list[MemoryItem]:
        with sqlite3.connect(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT memory_id, user_id, source, contains_sensitive_content,
                       source_turn_id, kind, state, content
                FROM memory_items
                WHERE user_id = ? AND state = ?
                ORDER BY rowid ASC
                """,
                (user_id, MemoryState.ACTIVE.value),
            ).fetchall()
        return [
            MemoryItem(
                memory_id=row[0],
                user_id=row[1],
                candidate=MemoryCandidate(
                    source=row[2],
                    contains_sensitive_content=bool(row[3]),
                    source_turn_id=row[4],
                    kind=MemoryKind(row[5]),
                    text=row[7],
                ),
                state=MemoryState(row[6]),
            )
            for row in rows
        ]
