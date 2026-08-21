import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import cast
from uuid import uuid4

from app.memory.models import (
    MemoryAllowedUse,
    MemoryAspect,
    MemoryCandidate,
    MemoryItem,
    MemoryKind,
    MemoryRecall,
    MemorySensitivity,
    MemorySourceType,
    MemoryState,
)

_SELECT_COLUMNS = """
    memory_id, user_id, source, contains_sensitive_content,
    source_turn_id, kind, state, content, aspect, subject_key,
    confidence, source_message_ids_json, source_window_id,
    purpose_scope, valid_from_ms, expires_at_ms, user_confirmed,
    integrity_flags_json, user_edited, created_at_ms, updated_at_ms,
    supersedes_memory_id, source_type, sensitivity, allowed_uses_json,
    observed_at_ms, valid_to_ms, derived_from_memory_ids_json
"""

_MIGRATION_COLUMNS = {
    "aspect": "TEXT NOT NULL DEFAULT 'FACT'",
    "subject_key": "TEXT NOT NULL DEFAULT ''",
    "confidence": "REAL NOT NULL DEFAULT 1.0",
    "source_message_ids_json": "TEXT NOT NULL DEFAULT '[]'",
    "source_window_id": "TEXT",
    "purpose_scope": "TEXT NOT NULL DEFAULT 'personalization'",
    "valid_from_ms": "INTEGER",
    "expires_at_ms": "INTEGER",
    "user_confirmed": "INTEGER NOT NULL DEFAULT 0",
    "integrity_flags_json": "TEXT NOT NULL DEFAULT '[]'",
    "user_edited": "INTEGER NOT NULL DEFAULT 0",
    "created_at_ms": "INTEGER NOT NULL DEFAULT 0",
    "updated_at_ms": "INTEGER NOT NULL DEFAULT 0",
    "supersedes_memory_id": "TEXT",
    "source_type": "TEXT NOT NULL DEFAULT 'LEGACY'",
    "sensitivity": "TEXT NOT NULL DEFAULT 'GENERAL'",
    "allowed_uses_json": (
        "TEXT NOT NULL DEFAULT '[\"PERSONALIZATION\",\"RESPONSE_CONTEXT\"]'"
    ),
    "observed_at_ms": "INTEGER",
    "valid_to_ms": "INTEGER",
    "derived_from_memory_ids_json": "TEXT NOT NULL DEFAULT '[]'",
}

_SHADOW_MIGRATION_COLUMNS = {
    "expires_at_ms": "INTEGER NOT NULL DEFAULT 0",
}

_OBSERVATION_MIGRATION_COLUMNS = {
    "valid_at_ms": "INTEGER NOT NULL DEFAULT 0",
}

_PROFILE_MIGRATION_COLUMNS = {
    "valid_to_ms": "INTEGER",
}


class MemoryRepository:
    """SQLite-backed governed memory with temporal supersession and expiry."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        schema = (
            Path(__file__).parents[1] / "events" / "schema.sql"
        ).read_text(encoding="utf-8")
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.executescript(schema)
            existing = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(memory_items)")
            }
            for name, definition in _MIGRATION_COLUMNS.items():
                if name not in existing:
                    connection.execute(
                        f"ALTER TABLE memory_items ADD COLUMN {name} {definition}"
                    )
            shadow_columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(memory_shadow_runs)"
                )
            }
            for name, definition in _SHADOW_MIGRATION_COLUMNS.items():
                if name not in shadow_columns:
                    connection.execute(
                        "ALTER TABLE memory_shadow_runs "
                        f"ADD COLUMN {name} {definition}"
                    )
            observation_columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(memory_observations)"
                )
            }
            for name, definition in _OBSERVATION_MIGRATION_COLUMNS.items():
                if name not in observation_columns:
                    connection.execute(
                        "ALTER TABLE memory_observations "
                        f"ADD COLUMN {name} {definition}"
                    )
            connection.execute(
                """
                UPDATE memory_observations SET valid_at_ms = observed_at_ms
                WHERE valid_at_ms = 0
                """
            )
            profile_columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(memory_profiles)")
            }
            for name, definition in _PROFILE_MIGRATION_COLUMNS.items():
                if name not in profile_columns:
                    connection.execute(
                        "ALTER TABLE memory_profiles "
                        f"ADD COLUMN {name} {definition}"
                    )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_active_scope
                ON memory_items(user_id, state, purpose_scope, expires_at_ms)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_subject_state
                ON memory_items(user_id, subject_key, state, updated_at_ms)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_ledger_governance
                ON memory_items(user_id, state, sensitivity, source_type)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_shadow_runs_expiry
                ON memory_shadow_runs(expires_at_ms)
                """
            )

    def add_active(
        self,
        *,
        user_id: str,
        candidate: MemoryCandidate,
        now_ms: int | None = None,
    ) -> MemoryItem:
        confirmed = candidate.model_copy(update={"user_confirmed": True})
        return self.store_candidates(
            user_id=user_id,
            candidates=[(confirmed, MemoryState.ACTIVE)],
            now_ms=now_ms,
        )[0]

    def add_awaiting_consent(
        self,
        *,
        user_id: str,
        candidate: MemoryCandidate,
        now_ms: int | None = None,
    ) -> MemoryItem:
        return self.store_candidates(
            user_id=user_id,
            candidates=[(candidate, MemoryState.AWAITING_CONSENT)],
            now_ms=now_ms,
        )[0]

    def store_candidates(
        self,
        *,
        user_id: str,
        candidates: list[tuple[MemoryCandidate, MemoryState]],
        now_ms: int | None = None,
    ) -> list[MemoryItem]:
        """Persist a batch in one transaction and avoid duplicate model proposals."""
        timestamp = now_ms if now_ms is not None else int(time.time() * 1000)
        stored: list[MemoryItem] = []
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.row_factory = sqlite3.Row
            connection.execute("BEGIN IMMEDIATE")
            for candidate, state in candidates:
                self._validate_lineage(connection, user_id, candidate)
                duplicate = self._find_duplicate(connection, user_id, candidate)
                if duplicate is not None:
                    stored.append(self._row_to_item(duplicate))
                    continue

                supersedes: str | None = None
                if state is MemoryState.ACTIVE and candidate.subject_key:
                    supersedes = self._supersede_current(
                        connection,
                        user_id=user_id,
                        candidate=candidate,
                        now_ms=timestamp,
                    )

                memory_id = f"memory_{uuid4().hex}"
                connection.execute(
                    """
                    INSERT INTO memory_items(
                        memory_id, user_id, source, contains_sensitive_content,
                        source_turn_id, kind, state, content, aspect, subject_key,
                        confidence, source_message_ids_json, source_window_id,
                        purpose_scope, valid_from_ms, expires_at_ms,
                        user_confirmed, integrity_flags_json, user_edited,
                        created_at_ms, updated_at_ms, supersedes_memory_id,
                        source_type, sensitivity, allowed_uses_json,
                        observed_at_ms, valid_to_ms,
                        derived_from_memory_ids_json
                    )
                    VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        memory_id,
                        user_id,
                        candidate.source,
                        int(candidate.contains_sensitive_content),
                        candidate.source_turn_id,
                        candidate.kind.value,
                        state.value,
                        candidate.text,
                        candidate.aspect.value,
                        candidate.subject_key,
                        candidate.confidence,
                        json.dumps(candidate.source_message_ids, ensure_ascii=False),
                        candidate.source_window_id,
                        candidate.purpose_scope,
                        candidate.valid_from_ms,
                        candidate.expires_at_ms,
                        int(candidate.user_confirmed or state is MemoryState.ACTIVE),
                        json.dumps(candidate.integrity_flags, ensure_ascii=False),
                        int(candidate.user_edited),
                        timestamp,
                        timestamp,
                        supersedes,
                        candidate.source_type.value,
                        candidate.sensitivity.value,
                        json.dumps(
                            [allowed_use.value for allowed_use in candidate.allowed_uses],
                            ensure_ascii=False,
                        ),
                        (
                            candidate.observed_at_ms
                            if candidate.observed_at_ms is not None
                            else timestamp
                        ),
                        candidate.valid_to_ms,
                        json.dumps(
                            candidate.derived_from_memory_ids,
                            ensure_ascii=False,
                        ),
                    ),
                )
                row = connection.execute(
                    f"SELECT {_SELECT_COLUMNS} FROM memory_items WHERE memory_id = ?",
                    (memory_id,),
                ).fetchone()
                assert row is not None
                stored.append(self._row_to_item(row))
        return stored

    def activate(
        self,
        memory_id: str,
        *,
        user_id: str | None = None,
        now_ms: int | None = None,
    ) -> MemoryItem:
        timestamp = now_ms if now_ms is not None else int(time.time() * 1000)
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.row_factory = sqlite3.Row
            connection.execute("BEGIN IMMEDIATE")
            if user_id is None:
                row = connection.execute(
                    f"SELECT {_SELECT_COLUMNS} FROM memory_items WHERE memory_id = ?",
                    (memory_id,),
                ).fetchone()
            else:
                row = connection.execute(
                    f"""
                    SELECT {_SELECT_COLUMNS} FROM memory_items
                    WHERE memory_id = ? AND user_id = ?
                    """,
                    (memory_id, user_id),
                ).fetchone()
            if row is None:
                raise KeyError(memory_id)
            item = self._row_to_item(row)
            if item.state is not MemoryState.AWAITING_CONSENT:
                raise ValueError("only awaiting-consent memory can be activated")
            supersedes = self._supersede_current(
                connection,
                user_id=item.user_id,
                candidate=item.candidate,
                now_ms=timestamp,
            )
            connection.execute(
                """
                UPDATE memory_items
                SET state = ?, user_confirmed = 1, updated_at_ms = ?,
                    supersedes_memory_id = ?
                WHERE memory_id = ?
                """,
                (MemoryState.ACTIVE.value, timestamp, supersedes, memory_id),
            )
            activated = connection.execute(
                f"SELECT {_SELECT_COLUMNS} FROM memory_items WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            assert activated is not None
            return self._row_to_item(activated)

    def revoke(
        self,
        memory_id: str,
        *,
        user_id: str | None = None,
        now_ms: int | None = None,
    ) -> None:
        timestamp = now_ms if now_ms is not None else int(time.time() * 1000)
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            query = """
                UPDATE memory_items SET state = ?, updated_at_ms = ?
                WHERE memory_id = ?
            """
            values: tuple[object, ...] = (
                MemoryState.REVOKED.value,
                timestamp,
                memory_id,
            )
            if user_id is not None:
                query += " AND user_id = ?"
                values += (user_id,)
            cursor = connection.execute(query, values)
            if user_id is not None and cursor.rowcount == 0:
                raise KeyError(memory_id)
            if cursor.rowcount > 0:
                self._invalidate_derived_profiles(
                    connection,
                    memory_id=memory_id,
                    now_ms=timestamp,
                )

    def purge(self, memory_id: str, *, user_id: str | None = None) -> bool:
        """Physically remove content after a user deletion request."""
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.execute("PRAGMA secure_delete = ON")
            if user_id is None:
                cursor = connection.execute(
                    "DELETE FROM memory_items WHERE memory_id = ?",
                    (memory_id,),
                )
            else:
                cursor = connection.execute(
                    "DELETE FROM memory_items WHERE memory_id = ? AND user_id = ?",
                    (memory_id, user_id),
                )
            deleted = cursor.rowcount > 0
            if deleted:
                self._delete_derived_episode_summaries(
                    connection,
                    memory_id=memory_id,
                )
                profile_rows = connection.execute(
                    """
                    SELECT DISTINCT profile_id FROM memory_profile_evidence
                    WHERE memory_id = ?
                    """,
                    (memory_id,),
                ).fetchall()
                for (profile_id,) in profile_rows:
                    connection.execute(
                        """
                        DELETE FROM memory_profile_changes
                        WHERE previous_profile_id = ? OR proposed_profile_id = ?
                        """,
                        (profile_id, profile_id),
                    )
                    conflict_rows = connection.execute(
                        """
                        SELECT conflict_id FROM memory_conflict_options
                        WHERE profile_id = ?
                        """,
                        (profile_id,),
                    ).fetchall()
                    for (conflict_id,) in conflict_rows:
                        connection.execute(
                            "DELETE FROM memory_conflict_options WHERE conflict_id = ?",
                            (conflict_id,),
                        )
                        connection.execute(
                            "DELETE FROM memory_conflict_groups WHERE conflict_id = ?",
                            (conflict_id,),
                        )
                    connection.execute(
                        "DELETE FROM memory_profile_evidence WHERE profile_id = ?",
                        (profile_id,),
                    )
                    connection.execute(
                        "DELETE FROM memory_retrievals WHERE memory_id = ?",
                        (profile_id,),
                    )
                    connection.execute(
                        "DELETE FROM memory_profiles WHERE profile_id = ?",
                        (profile_id,),
                    )
                connection.execute(
                    "DELETE FROM memory_observations WHERE memory_id = ?",
                    (memory_id,),
                )
                shadow_run_rows = connection.execute(
                    """
                    SELECT DISTINCT shadow_run_id FROM memory_shadow_rankings
                    WHERE memory_id = ?
                    """,
                    (memory_id,),
                ).fetchall()
                for (shadow_run_id,) in shadow_run_rows:
                    connection.execute(
                        "DELETE FROM memory_shadow_rankings WHERE shadow_run_id = ?",
                        (shadow_run_id,),
                    )
                    connection.execute(
                        "DELETE FROM memory_shadow_runs WHERE shadow_run_id = ?",
                        (shadow_run_id,),
                    )
                connection.execute(
                    "DELETE FROM memory_retrievals WHERE memory_id = ?",
                    (memory_id,),
                )
            return deleted

    def list_active(
        self,
        user_id: str,
        *,
        purpose_scope: str = "personalization",
        as_of_ms: int | None = None,
    ) -> list[MemoryItem]:
        timestamp = as_of_ms if as_of_ms is not None else int(time.time() * 1000)
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.row_factory = sqlite3.Row
            connection.execute(
                """
                UPDATE memory_items SET state = ?, updated_at_ms = ?
                WHERE user_id = ? AND state = ?
                  AND expires_at_ms IS NOT NULL AND expires_at_ms <= ?
                """,
                (
                    MemoryState.EXPIRED.value,
                    timestamp,
                    user_id,
                    MemoryState.ACTIVE.value,
                    timestamp,
                ),
            )
            rows = connection.execute(
                f"""
                SELECT {_SELECT_COLUMNS}
                FROM memory_items
                WHERE user_id = ? AND state = ? AND purpose_scope = ?
                  AND user_confirmed = 1
                  AND (valid_from_ms IS NULL OR valid_from_ms <= ?)
                  AND (valid_to_ms IS NULL OR valid_to_ms > ?)
                  AND (expires_at_ms IS NULL OR expires_at_ms > ?)
                ORDER BY updated_at_ms DESC, rowid DESC
                """,
                (
                    user_id,
                    MemoryState.ACTIVE.value,
                    purpose_scope,
                    timestamp,
                    timestamp,
                    timestamp,
                ),
            ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def list_awaiting_consent(self, user_id: str) -> list[MemoryItem]:
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                f"""
                SELECT {_SELECT_COLUMNS}
                FROM memory_items
                WHERE user_id = ? AND state = ?
                ORDER BY created_at_ms ASC, rowid ASC
                """,
                (user_id, MemoryState.AWAITING_CONSENT.value),
            ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def get(self, memory_id: str, *, user_id: str) -> MemoryItem:
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                f"""
                SELECT {_SELECT_COLUMNS} FROM memory_items
                WHERE memory_id = ? AND user_id = ?
                """,
                (memory_id, user_id),
            ).fetchone()
        if row is None:
            raise KeyError(memory_id)
        return self._row_to_item(row)

    def update_user_memory(
        self,
        memory_id: str,
        *,
        user_id: str,
        text: str,
        expires_at_ms: int | None,
        contains_sensitive_content: bool,
        aspect: MemoryAspect | None = None,
        subject_key: str | None = None,
        now_ms: int | None = None,
    ) -> MemoryItem:
        timestamp = now_ms if now_ms is not None else int(time.time() * 1000)
        normalized = " ".join(text.split()).strip()
        if not normalized:
            raise ValueError("memory text must not be blank")
        if expires_at_ms is not None and expires_at_ms <= timestamp:
            raise ValueError("memory expiry must be in the future")
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                f"""
                SELECT {_SELECT_COLUMNS} FROM memory_items
                WHERE memory_id = ? AND user_id = ?
                """,
                (memory_id, user_id),
            ).fetchone()
            if row is None:
                raise KeyError(memory_id)
            item = self._row_to_item(row)
            if item.state not in {
                MemoryState.AWAITING_CONSENT,
                MemoryState.ACTIVE,
                MemoryState.EXPIRED,
            }:
                raise ValueError("memory state does not allow editing")
            self._invalidate_derived_profiles(
                connection,
                memory_id=memory_id,
                now_ms=timestamp,
            )
            next_state = (
                MemoryState.ACTIVE
                if item.state is MemoryState.EXPIRED
                else item.state
            )
            next_aspect = aspect or item.candidate.aspect
            next_subject_key = (
                subject_key
                if subject_key is not None
                else item.candidate.subject_key
            )
            supersedes = item.supersedes_memory_id
            if next_state is MemoryState.ACTIVE and next_subject_key:
                conflicting = connection.execute(
                    """
                    SELECT memory_id FROM memory_items
                    WHERE user_id = ? AND subject_key = ? AND state = ?
                      AND memory_id <> ?
                    ORDER BY updated_at_ms DESC, rowid DESC LIMIT 1
                    """,
                    (
                        user_id,
                        next_subject_key,
                        MemoryState.ACTIVE.value,
                        memory_id,
                    ),
                ).fetchone()
                if conflicting is not None:
                    supersedes = str(conflicting[0])
                    connection.execute(
                        """
                        UPDATE memory_items SET state = ?, updated_at_ms = ?
                        WHERE memory_id = ?
                        """,
                        (
                            MemoryState.SUPERSEDED.value,
                            timestamp,
                            supersedes,
                        ),
                    )
            connection.execute(
                """
                UPDATE memory_items
                SET content = ?, contains_sensitive_content = ?,
                    aspect = ?, subject_key = ?, expires_at_ms = ?,
                    state = ?, confidence = 1.0, user_edited = 1,
                    updated_at_ms = ?, supersedes_memory_id = ?,
                    source_type = ?, sensitivity = ?, observed_at_ms = ?
                WHERE memory_id = ? AND user_id = ?
                """,
                (
                    normalized,
                    int(contains_sensitive_content),
                    next_aspect.value,
                    next_subject_key,
                    expires_at_ms,
                    next_state.value,
                    timestamp,
                    supersedes,
                    MemorySourceType.USER_EDIT.value,
                    (
                        MemorySensitivity.HEALTH_SENSITIVE.value
                        if contains_sensitive_content
                        else MemorySensitivity.GENERAL.value
                    ),
                    timestamp,
                    memory_id,
                    user_id,
                ),
            )
            updated = connection.execute(
                f"SELECT {_SELECT_COLUMNS} FROM memory_items WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
        assert updated is not None
        return self._row_to_item(updated)

    @staticmethod
    def _invalidate_derived_profiles(
        connection: sqlite3.Connection,
        *,
        memory_id: str,
        now_ms: int,
    ) -> None:
        """Fail closed when source evidence is revoked or materially edited."""
        MemoryRepository._delete_derived_episode_summaries(
            connection,
            memory_id=memory_id,
        )
        profile_rows = connection.execute(
            """
            SELECT DISTINCT profile_id FROM memory_profile_evidence
            WHERE memory_id = ?
            """,
            (memory_id,),
        ).fetchall()
        profile_ids = [str(row[0]) for row in profile_rows]
        for profile_id in profile_ids:
            connection.execute(
                """
                UPDATE memory_profile_changes SET state = 'REJECTED', updated_at_ms = ?
                WHERE (previous_profile_id = ? OR proposed_profile_id = ?)
                  AND state = 'OPEN'
                """,
                (now_ms, profile_id, profile_id),
            )
            connection.execute(
                """
                UPDATE memory_profiles SET state = 'STALE', updated_at_ms = ?
                WHERE profile_id = ?
                """,
                (now_ms, profile_id),
            )
            connection.execute(
                """
                UPDATE memory_conflict_groups
                SET state = 'DISMISSED', updated_at_ms = ?
                WHERE conflict_id IN (
                    SELECT conflict_id FROM memory_conflict_options
                    WHERE profile_id = ?
                ) AND state = 'OPEN'
                """,
                (now_ms, profile_id),
            )

    @staticmethod
    def _delete_derived_episode_summaries(
        connection: sqlite3.Connection,
        *,
        memory_id: str,
    ) -> None:
        """Physically remove a derived index before source text changes."""
        connection.execute("PRAGMA secure_delete = ON")
        rows = connection.execute(
            """
            SELECT DISTINCT summary_id FROM memory_episode_members
            WHERE memory_id = ?
            """,
            (memory_id,),
        ).fetchall()
        for (summary_id,) in rows:
            connection.execute(
                "DELETE FROM memory_episode_members WHERE summary_id = ?",
                (summary_id,),
            )
            connection.execute(
                "DELETE FROM memory_episode_summaries WHERE summary_id = ?",
                (summary_id,),
            )

    def record_retrievals(
        self,
        *,
        user_id: str,
        session_id: str,
        turn_id: str,
        retrievals: list[MemoryRecall],
    ) -> None:
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            for recall in retrievals:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO memory_retrievals(
                        retrieval_id, user_id, memory_id, session_id, turn_id,
                        score, relevance_score, reason_codes_json, used_at_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"recall_{uuid4().hex}",
                        user_id,
                        recall.memory_id,
                        session_id,
                        turn_id,
                        recall.score,
                        recall.relevance_score,
                        json.dumps(recall.reason_codes, ensure_ascii=False),
                        recall.used_at_ms,
                    ),
                )

    def latest_recall(
        self,
        memory_id: str,
        *,
        user_id: str,
    ) -> MemoryRecall | None:
        with closing(sqlite3.connect(self.database_path)) as connection:
            row = connection.execute(
                """
                SELECT memory_id, session_id, turn_id, score,
                       relevance_score, reason_codes_json, used_at_ms
                FROM memory_retrievals
                WHERE memory_id = ? AND user_id = ?
                ORDER BY used_at_ms DESC, rowid DESC LIMIT 1
                """,
                (memory_id, user_id),
            ).fetchone()
        if row is None:
            return None
        return MemoryRecall(
            memory_id=str(row[0]),
            session_id=str(row[1]),
            turn_id=str(row[2]),
            score=float(row[3]),
            relevance_score=float(row[4]),
            reason_codes=json.loads(str(row[5])),
            used_at_ms=int(row[6]),
        )

    def list_visible(
        self,
        user_id: str,
        *,
        as_of_ms: int | None = None,
    ) -> list[MemoryItem]:
        timestamp = as_of_ms if as_of_ms is not None else int(time.time() * 1000)
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute(
                """
                UPDATE memory_items SET state = ?, updated_at_ms = ?
                WHERE user_id = ? AND state = ?
                  AND expires_at_ms IS NOT NULL AND expires_at_ms <= ?
                """,
                (
                    MemoryState.EXPIRED.value,
                    timestamp,
                    user_id,
                    MemoryState.ACTIVE.value,
                    timestamp,
                ),
            )
            connection.commit()
            rows = connection.execute(
                f"""
                SELECT {_SELECT_COLUMNS} FROM memory_items
                WHERE user_id = ? AND state IN (?, ?, ?)
                ORDER BY updated_at_ms DESC, rowid DESC
                """,
                (
                    user_id,
                    MemoryState.AWAITING_CONSENT.value,
                    MemoryState.ACTIVE.value,
                    MemoryState.EXPIRED.value,
                ),
            ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def get_ingestion_cursor(self, *, user_id: str, session_id: str) -> int:
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            row = connection.execute(
                """
                SELECT last_sequence FROM memory_ingestion_cursors
                WHERE user_id = ? AND session_id = ?
                """,
                (user_id, session_id),
            ).fetchone()
        return int(row[0]) if row is not None else 0

    def update_ingestion_cursor(
        self,
        *,
        user_id: str,
        session_id: str,
        last_sequence: int,
        now_ms: int | None = None,
    ) -> None:
        timestamp = now_ms if now_ms is not None else int(time.time() * 1000)
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.execute(
                """
                INSERT INTO memory_ingestion_cursors(
                    user_id, session_id, last_sequence, updated_at_ms
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, session_id) DO UPDATE SET
                    last_sequence = MAX(last_sequence, excluded.last_sequence),
                    updated_at_ms = excluded.updated_at_ms
                """,
                (user_id, session_id, last_sequence, timestamp),
            )

    @staticmethod
    def _validate_lineage(
        connection: sqlite3.Connection,
        user_id: str,
        candidate: MemoryCandidate,
    ) -> None:
        if not candidate.derived_from_memory_ids:
            return
        placeholders = ",".join("?" for _ in candidate.derived_from_memory_ids)
        rows = connection.execute(
            f"""
            SELECT memory_id FROM memory_items
            WHERE user_id = ? AND memory_id IN ({placeholders})
            """,
            (user_id, *candidate.derived_from_memory_ids),
        ).fetchall()
        known = {str(row[0]) for row in rows}
        missing = set(candidate.derived_from_memory_ids) - known
        if missing:
            raise ValueError("derived memories must exist for the same user")

    @staticmethod
    def _find_duplicate(
        connection: sqlite3.Connection,
        user_id: str,
        candidate: MemoryCandidate,
    ) -> sqlite3.Row | None:
        return cast(
            sqlite3.Row | None,
            connection.execute(
                f"""
                SELECT {_SELECT_COLUMNS}
                FROM memory_items
                WHERE user_id = ? AND content = ? AND aspect = ?
                  AND state IN (?, ?)
                ORDER BY rowid DESC LIMIT 1
                """,
                (
                    user_id,
                    candidate.text,
                    candidate.aspect.value,
                    MemoryState.ACTIVE.value,
                    MemoryState.AWAITING_CONSENT.value,
                ),
            ).fetchone(),
        )

    @staticmethod
    def _supersede_current(
        connection: sqlite3.Connection,
        *,
        user_id: str,
        candidate: MemoryCandidate,
        now_ms: int,
    ) -> str | None:
        if not candidate.subject_key:
            return None
        row = connection.execute(
            """
            SELECT memory_id FROM memory_items
            WHERE user_id = ? AND subject_key = ? AND state = ?
            ORDER BY updated_at_ms DESC, rowid DESC LIMIT 1
            """,
            (
                user_id,
                candidate.subject_key,
                MemoryState.ACTIVE.value,
            ),
        ).fetchone()
        if row is None:
            return None
        memory_id = str(row[0])
        connection.execute(
            """
            UPDATE memory_items SET state = ?, updated_at_ms = ?
            WHERE memory_id = ?
            """,
            (MemoryState.SUPERSEDED.value, now_ms, memory_id),
        )
        return memory_id

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> MemoryItem:
        candidate = MemoryCandidate(
            source=str(row["source"]),
            contains_sensitive_content=bool(row["contains_sensitive_content"]),
            source_turn_id=str(row["source_turn_id"]),
            kind=MemoryKind(str(row["kind"])),
            text=str(row["content"]),
            aspect=MemoryAspect(str(row["aspect"])),
            subject_key=str(row["subject_key"]),
            confidence=float(row["confidence"]),
            source_message_ids=json.loads(str(row["source_message_ids_json"])),
            source_window_id=(
                str(row["source_window_id"])
                if row["source_window_id"] is not None
                else None
            ),
            purpose_scope=str(row["purpose_scope"]),
            valid_from_ms=(
                int(row["valid_from_ms"])
                if row["valid_from_ms"] is not None
                else None
            ),
            expires_at_ms=(
                int(row["expires_at_ms"])
                if row["expires_at_ms"] is not None
                else None
            ),
            user_confirmed=bool(row["user_confirmed"]),
            integrity_flags=json.loads(str(row["integrity_flags_json"])),
            user_edited=bool(row["user_edited"]),
            source_type=MemorySourceType(str(row["source_type"])),
            sensitivity=MemorySensitivity(str(row["sensitivity"])),
            allowed_uses=[
                MemoryAllowedUse(value)
                for value in json.loads(str(row["allowed_uses_json"]))
            ],
            observed_at_ms=(
                int(row["observed_at_ms"])
                if row["observed_at_ms"] is not None
                else None
            ),
            valid_to_ms=(
                int(row["valid_to_ms"])
                if row["valid_to_ms"] is not None
                else None
            ),
            derived_from_memory_ids=json.loads(
                str(row["derived_from_memory_ids_json"])
            ),
        )
        return MemoryItem(
            memory_id=str(row["memory_id"]),
            user_id=str(row["user_id"]),
            candidate=candidate,
            state=MemoryState(str(row["state"])),
            created_at_ms=int(row["created_at_ms"]),
            updated_at_ms=int(row["updated_at_ms"]),
            supersedes_memory_id=(
                str(row["supersedes_memory_id"])
                if row["supersedes_memory_id"] is not None
                else None
            ),
        )
