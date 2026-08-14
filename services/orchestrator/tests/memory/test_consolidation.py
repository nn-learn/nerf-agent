import sqlite3
from pathlib import Path

import pytest

from app.memory.consolidation import MemoryConsolidator, MemoryProfileRepository
from app.memory.models import (
    MemoryAspect,
    MemoryCandidate,
    MemoryConflictState,
    MemoryKind,
    MemoryProfileState,
)
from app.memory.repository import MemoryRepository
from app.memory.retrieval import (
    GovernedMemoryRetriever,
    GovernedProfileRetriever,
    LayeredMemoryRetriever,
)


def _candidate(
    text: str,
    *,
    turn_id: str,
    subject_key: str = "communication.response_style",
) -> MemoryCandidate:
    return MemoryCandidate(
        source="transcript",
        contains_sensitive_content=False,
        text=text,
        kind=MemoryKind.SEMANTIC,
        source_turn_id=turn_id,
        aspect=MemoryAspect.PREFERENCE,
        subject_key=subject_key,
        confidence=0.9,
        source_message_ids=[f"message_{turn_id}"],
    )


@pytest.fixture
def memory_v2(
    tmp_path: Path,
) -> tuple[MemoryRepository, MemoryProfileRepository, MemoryConsolidator]:
    repository = MemoryRepository(tmp_path / "memory.sqlite3")
    repository.initialize()
    profiles = MemoryProfileRepository(repository.database_path)
    return repository, profiles, MemoryConsolidator(profiles)


def test_single_session_never_becomes_a_profile(
    memory_v2: tuple[MemoryRepository, MemoryProfileRepository, MemoryConsolidator],
) -> None:
    repository, profiles, consolidator = memory_v2
    item = repository.add_active(
        user_id="user_1",
        candidate=_candidate("用户偏好：回答时简短一点", turn_id="turn_1"),
        now_ms=1_000,
    )
    profiles.record_observation(item, session_id="session_1", now_ms=1_000)

    assert consolidator.rebuild_user("user_1", now_ms=2_000) == []
    assert profiles.list_profiles("user_1") == []


def test_repeated_cross_session_evidence_requires_confirmation_before_recall(
    memory_v2: tuple[MemoryRepository, MemoryProfileRepository, MemoryConsolidator],
) -> None:
    repository, profiles, consolidator = memory_v2
    item = repository.add_active(
        user_id="user_1",
        candidate=_candidate("用户偏好：回答时简短一点", turn_id="turn_1"),
        now_ms=1_000,
    )
    profiles.record_observation(item, session_id="session_1", now_ms=1_000)
    profiles.record_observation(item, session_id="session_2", now_ms=2_000)

    proposed = consolidator.rebuild_user("user_1", now_ms=3_000)
    assert len(proposed) == 1
    assert proposed[0].state is MemoryProfileState.AWAITING_CONFIRMATION
    assert proposed[0].distinct_session_count == 2

    profile_retriever = GovernedProfileRetriever(profiles)
    assert profile_retriever.retrieve("怎样回答", user_id="user_1", as_of_ms=4_000) == []

    confirmed = profiles.confirm_profile(
        proposed[0].profile_id,
        user_id="user_1",
        now_ms=4_000,
    )
    assert confirmed.state is MemoryProfileState.ACTIVE
    recalled = profile_retriever.retrieve(
        "怎样回答",
        user_id="user_1",
        as_of_ms=5_000,
    )
    assert [result.memory_id for result in recalled] == [confirmed.profile_id]
    assert "USER_CONFIRMED_PROFILE" in recalled[0].reason_codes

    layered = LayeredMemoryRetriever(
        GovernedMemoryRetriever(repository),
        profile_retriever,
    ).retrieve("怎样回答", user_id="user_1", as_of_ms=5_000)
    assert [result.memory_id for result in layered] == [confirmed.profile_id]


def test_conflicting_cross_session_evidence_freezes_recall_until_user_selects(
    memory_v2: tuple[MemoryRepository, MemoryProfileRepository, MemoryConsolidator],
) -> None:
    repository, profiles, consolidator = memory_v2
    concise = repository.add_active(
        user_id="user_1",
        candidate=_candidate("用户偏好：回答要简短", turn_id="turn_1"),
        now_ms=1_000,
    )
    profiles.record_observation(concise, session_id="session_1", now_ms=1_000)
    detailed = repository.add_active(
        user_id="user_1",
        candidate=_candidate("用户偏好：请详细展开解释", turn_id="turn_2"),
        now_ms=2_000,
    )
    profiles.record_observation(detailed, session_id="session_2", now_ms=2_000)

    consolidator.rebuild_user("user_1", now_ms=3_000)
    conflicts = profiles.list_conflicts("user_1", state=MemoryConflictState.OPEN)
    assert len(conflicts) == 1
    assert len(conflicts[0].options) == 2
    assert {option.state for option in conflicts[0].options} == {
        MemoryProfileState.STALE
    }
    retriever = GovernedProfileRetriever(profiles)
    assert retriever.retrieve("回答方式", user_id="user_1", as_of_ms=4_000) == []

    selected = next(
        option for option in conflicts[0].options if "详细" in option.statement
    )
    profiles.confirm_profile(selected.profile_id, user_id="user_1", now_ms=4_000)
    resolved = profiles.get_conflict(conflicts[0].conflict_id, user_id="user_1")
    assert resolved.state is MemoryConflictState.RESOLVED
    assert resolved.selected_profile_id == selected.profile_id
    assert [item.memory_id for item in retriever.retrieve(
        "回答方式", user_id="user_1", as_of_ms=5_000
    )] == [selected.profile_id]


def test_rejected_profile_is_not_recreated_from_same_evidence(
    memory_v2: tuple[MemoryRepository, MemoryProfileRepository, MemoryConsolidator],
) -> None:
    repository, profiles, consolidator = memory_v2
    item = repository.add_active(
        user_id="user_1",
        candidate=_candidate("用户偏好：回答时简短", turn_id="turn_1"),
    )
    profiles.record_observation(item, session_id="session_1")
    profiles.record_observation(item, session_id="session_2")
    proposed = consolidator.rebuild_user("user_1")

    profiles.reject_profile(proposed[0].profile_id, user_id="user_1")

    assert consolidator.rebuild_user("user_1") == []
    assert profiles.list_profiles("user_1") == []


def test_revoke_and_purge_cascade_to_derived_profile(
    memory_v2: tuple[MemoryRepository, MemoryProfileRepository, MemoryConsolidator],
) -> None:
    repository, profiles, consolidator = memory_v2
    item = repository.add_active(
        user_id="user_1",
        candidate=_candidate("用户偏好：回答时简短", turn_id="turn_1"),
    )
    profiles.record_observation(item, session_id="session_1")
    profiles.record_observation(item, session_id="session_2")
    proposed = consolidator.rebuild_user("user_1")
    active = profiles.confirm_profile(proposed[0].profile_id, user_id="user_1")

    repository.revoke(item.memory_id, user_id="user_1")

    assert profiles.get_profile(
        active.profile_id,
        user_id="user_1",
    ).state is MemoryProfileState.STALE
    assert profiles.list_active_profiles("user_1") == []
    assert repository.purge(item.memory_id, user_id="user_1") is True
    with pytest.raises(KeyError):
        profiles.get_profile(active.profile_id, user_id="user_1")


def test_observation_uses_source_turn_session_instead_of_review_session(
    memory_v2: tuple[MemoryRepository, MemoryProfileRepository, MemoryConsolidator],
) -> None:
    repository, profiles, _ = memory_v2
    with sqlite3.connect(repository.database_path) as connection:
        connection.execute(
            "INSERT INTO turns(turn_id, session_id, trace_id, created_at_ms) VALUES (?, ?, ?, ?)",
            ("turn_original", "session_original", "trace_1", 1_000),
        )
    item = repository.add_active(
        user_id="user_1",
        candidate=_candidate("用户偏好：回答时简短", turn_id="turn_original"),
    )

    observation = profiles.record_observation(
        item,
        session_id="session_where_review_happened",
        resolve_source_session=True,
    )

    assert observation is not None
    assert observation.session_id == "session_original"
