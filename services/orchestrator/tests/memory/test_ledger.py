import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.memory.models import (
    MemoryAllowedUse,
    MemoryCandidate,
    MemoryKind,
    MemorySensitivity,
    MemorySourceType,
)
from app.memory.repository import MemoryRepository


def _candidate(**updates: object) -> MemoryCandidate:
    values: dict[str, object] = {
        "source": "transcript",
        "contains_sensitive_content": True,
        "text": "用户明确说最近睡眠不好",
        "kind": MemoryKind.SEMANTIC,
        "source_turn_id": "turn_1",
        "source_message_ids": ["message_1"],
        "valid_from_ms": 1_000,
        "source_type": MemorySourceType.USER_STATEMENT,
        "sensitivity": MemorySensitivity.HEALTH_SENSITIVE,
        "allowed_uses": [MemoryAllowedUse.RESPONSE_CONTEXT],
        "observed_at_ms": 2_000,
        "valid_to_ms": 9_000,
        "derived_from_memory_ids": [],
    }
    values.update(updates)
    return MemoryCandidate.model_validate(values)


def test_ledger_metadata_round_trips_through_repository(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory.sqlite3")
    repository.initialize()
    source = repository.add_active(
        user_id="user_1",
        candidate=_candidate(
            text="用户原始陈述",
            valid_to_ms=None,
            derived_from_memory_ids=[],
        ),
        now_ms=2_500,
    )

    stored = repository.add_active(
        user_id="user_1",
        candidate=_candidate(derived_from_memory_ids=[source.memory_id]),
        now_ms=3_000,
    )
    loaded = repository.get(stored.memory_id, user_id="user_1")

    assert loaded.candidate.source_type is MemorySourceType.USER_STATEMENT
    assert loaded.candidate.sensitivity is MemorySensitivity.HEALTH_SENSITIVE
    assert loaded.candidate.allowed_uses == [MemoryAllowedUse.RESPONSE_CONTEXT]
    assert loaded.candidate.observed_at_ms == 2_000
    assert loaded.candidate.valid_to_ms == 9_000
    assert loaded.candidate.derived_from_memory_ids == [source.memory_id]


def test_repository_backfills_legacy_ledger_defaults(tmp_path: Path) -> None:
    database_path = tmp_path / "memory.sqlite3"
    repository = MemoryRepository(database_path)
    repository.initialize()
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO memory_items(
                memory_id, user_id, source, contains_sensitive_content,
                source_turn_id, kind, state, content
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "memory_legacy",
                "user_1",
                "transcript",
                0,
                "turn_1",
                "SEMANTIC",
                "ACTIVE",
                "legacy preference",
            ),
        )

    loaded = repository.get("memory_legacy", user_id="user_1")

    assert loaded.candidate.source_type is MemorySourceType.LEGACY
    assert loaded.candidate.sensitivity is MemorySensitivity.GENERAL
    assert loaded.candidate.allowed_uses == [
        MemoryAllowedUse.PERSONALIZATION,
        MemoryAllowedUse.RESPONSE_CONTEXT,
    ]


def test_user_edit_is_a_new_observation_source(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory.sqlite3")
    repository.initialize()
    stored = repository.add_active(
        user_id="user_1",
        candidate=_candidate(valid_to_ms=None),
        now_ms=3_000,
    )

    edited = repository.update_user_memory(
        stored.memory_id,
        user_id="user_1",
        text="用户更正：最近睡眠已经改善",
        expires_at_ms=None,
        contains_sensitive_content=True,
        now_ms=4_000,
    )

    assert edited.candidate.source_type is MemorySourceType.USER_EDIT
    assert edited.candidate.observed_at_ms == 4_000
    assert edited.candidate.sensitivity is MemorySensitivity.HEALTH_SENSITIVE

    declassified = repository.update_user_memory(
        stored.memory_id,
        user_id="user_1",
        text="用户更正：只保留普通偏好",
        expires_at_ms=None,
        contains_sensitive_content=False,
        now_ms=5_000,
    )
    assert declassified.candidate.sensitivity is MemorySensitivity.GENERAL


def test_zero_observation_time_is_not_replaced_by_storage_time(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory.sqlite3")
    repository.initialize()

    stored = repository.add_active(
        user_id="user_1",
        candidate=_candidate(observed_at_ms=0, valid_to_ms=None),
        now_ms=3_000,
    )

    assert stored.candidate.observed_at_ms == 0


def test_ledger_rejects_invalid_time_and_duplicate_governed_uses() -> None:
    with pytest.raises(ValidationError, match="valid_to_ms"):
        _candidate(valid_from_ms=5_000, valid_to_ms=5_000)

    with pytest.raises(ValidationError, match="allowed_uses"):
        _candidate(
            allowed_uses=[
                MemoryAllowedUse.RESPONSE_CONTEXT,
                MemoryAllowedUse.RESPONSE_CONTEXT,
            ]
        )


def test_repository_rejects_cross_user_or_missing_lineage(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory.sqlite3")
    repository.initialize()
    other_user_source = repository.add_active(
        user_id="user_2",
        candidate=_candidate(valid_to_ms=None, derived_from_memory_ids=[]),
    )

    with pytest.raises(ValueError, match="same user"):
        repository.add_active(
            user_id="user_1",
            candidate=_candidate(
                valid_to_ms=None,
                derived_from_memory_ids=[other_user_source.memory_id],
            ),
        )


def test_valid_time_end_suppresses_recall_without_deleting_history(
    tmp_path: Path,
) -> None:
    repository = MemoryRepository(tmp_path / "memory.sqlite3")
    repository.initialize()
    stored = repository.add_active(
        user_id="user_1",
        candidate=_candidate(derived_from_memory_ids=[]),
        now_ms=3_000,
    )

    assert repository.list_active("user_1", as_of_ms=8_999) == [stored]
    assert repository.list_active("user_1", as_of_ms=9_000) == []
    assert repository.get(stored.memory_id, user_id="user_1").state.value == "ACTIVE"
