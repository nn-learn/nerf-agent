import hashlib
import hmac
import secrets
import sqlite3
import time
from contextlib import closing
from pathlib import Path


class InvalidMemorySubjectToken(ValueError):
    pass


class MemorySubjectStore:
    """Binds high-entropy opaque subject tokens to pseudonymous memory users."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def issue_or_resolve(
        self,
        *,
        session_id: str,
        supplied_token: str | None,
        now_ms: int | None = None,
    ) -> tuple[str, str | None]:
        timestamp = now_ms if now_ms is not None else int(time.time() * 1000)
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.execute("PRAGMA foreign_keys = ON")
            if supplied_token is None:
                token = f"pms_{secrets.token_urlsafe(32)}"
                user_id = f"memory_user_{secrets.token_hex(16)}"
                connection.execute(
                    """
                    INSERT INTO memory_subjects(
                        user_id, token_digest, created_at_ms, last_seen_at_ms
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (user_id, self._digest(token), timestamp, timestamp),
                )
                returned_token: str | None = token
            else:
                self._validate_token(supplied_token)
                row = connection.execute(
                    """
                    SELECT user_id, token_digest FROM memory_subjects
                    WHERE token_digest = ?
                    """,
                    (self._digest(supplied_token),),
                ).fetchone()
                if row is None or not hmac.compare_digest(
                    str(row[1]),
                    self._digest(supplied_token),
                ):
                    raise InvalidMemorySubjectToken("invalid memory subject token")
                user_id = str(row[0])
                returned_token = None
                connection.execute(
                    "UPDATE memory_subjects SET last_seen_at_ms = ? WHERE user_id = ?",
                    (timestamp, user_id),
                )
            connection.execute(
                """
                INSERT INTO session_memory_subjects(session_id, user_id, bound_at_ms)
                VALUES (?, ?, ?)
                """,
                (session_id, user_id, timestamp),
            )
        return user_id, returned_token

    def validate_existing(self, token: str) -> None:
        self._validate_token(token)
        digest = self._digest(token)
        with closing(sqlite3.connect(self.database_path)) as connection:
            row = connection.execute(
                "SELECT token_digest FROM memory_subjects WHERE token_digest = ?",
                (digest,),
            ).fetchone()
        if row is None or not hmac.compare_digest(str(row[0]), digest):
            raise InvalidMemorySubjectToken("invalid memory subject token")

    def user_for_session(self, session_id: str) -> str:
        with closing(sqlite3.connect(self.database_path)) as connection:
            row = connection.execute(
                """
                SELECT user_id FROM session_memory_subjects WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            raise KeyError(session_id)
        return str(row[0])

    def pending_ended_sessions(self, *, limit: int = 128) -> list[str]:
        """Recover ended sessions whose last message has not been ingested."""
        with closing(sqlite3.connect(self.database_path)) as connection:
            rows = connection.execute(
                """
                SELECT binding.session_id
                FROM session_memory_subjects AS binding
                WHERE EXISTS (
                    SELECT 1 FROM events
                    WHERE session_id = binding.session_id AND type = 'session.ended'
                )
                AND COALESCE((
                    SELECT last_sequence FROM memory_ingestion_cursors
                    WHERE user_id = binding.user_id
                      AND session_id = binding.session_id
                ), 0) < COALESCE((
                    SELECT MAX(seq) FROM events
                    WHERE session_id = binding.session_id
                      AND type IN ('transcript.final', 'assistant.response.ready')
                ), 0)
                ORDER BY binding.bound_at_ms ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [str(row[0]) for row in rows]

    def ingestion_complete(self, session_id: str) -> bool:
        """Return whether every persisted conversation message was ingested."""
        user_id = self.user_for_session(session_id)
        with closing(sqlite3.connect(self.database_path)) as connection:
            row = connection.execute(
                """
                SELECT
                    COALESCE((
                        SELECT last_sequence FROM memory_ingestion_cursors
                        WHERE user_id = ? AND session_id = ?
                    ), 0),
                    COALESCE((
                        SELECT MAX(seq) FROM events
                        WHERE session_id = ?
                          AND type IN (
                              'transcript.final', 'assistant.response.ready'
                          )
                    ), 0)
                """,
                (user_id, session_id, session_id),
            ).fetchone()
        assert row is not None
        return int(row[0]) >= int(row[1])

    @staticmethod
    def _digest(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _validate_token(token: str) -> None:
        if not token.startswith("pms_") or not 40 <= len(token) <= 128:
            raise InvalidMemorySubjectToken("invalid memory subject token")
        if not all(character.isalnum() or character in {"-", "_"} for character in token):
            raise InvalidMemorySubjectToken("invalid memory subject token")
