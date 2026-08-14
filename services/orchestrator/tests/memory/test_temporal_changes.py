import sqlite3
from pathlib import Path

import pytest

from app.memory.consolidation import MemoryConsolidator, MemoryProfileRepository
from app.memory.models import (
    MemoryAspect,
    MemoryCandidate,
    MemoryChangeState,
    MemoryKind,
    MemoryProfileState,
)
from app.memory.repository import MemoryRepository


def _candidate(text: str, *, turn_id: str, valid_at_ms: int) -> MemoryCandidate:
    return MemoryCandidate(
        source="transcript",
        contains_sensitive_content=False,
        text=text,
        kind=MemoryKind.SEMANTIC,
        source_turn_id=turn_id,
        aspect=MemoryAspect.PREFERENCE,
        subject_key="communication.response_style",
        confidence=0.9,
        valid_from_ms=valid_at_ms,
    )


def _active_baseline(
    tmp_path: Path,
) -> tuple[MemoryRepository, MemoryProfileRepository, MemoryConsolidator, str]:
    repository = MemoryRepository(tmp_path / "memory.sqlite3")
    repository.initialize()
    profiles = MemoryProfileRepository(repository.database_path)
    consolidator = MemoryConsolidator(profiles)
    concise = repository.add_active(
        user_id="user_1",
        candidate=_candidate(
            "用户偏好：回答时简短",
            turn_id="turn_1",
            valid_at_ms=1_000,
        ),
        now_ms=10_000,
    )
    profiles.record_observation(
        concise,
        session_id="session_1",
        now_ms=10_000,
    )
    profiles.record_observation(
        concise,
        session_id="session_2",
        now_ms=20_000,
    )
    proposed = consolidator.rebuild_user("user_1", now_ms=30_000)
    active = profiles.confirm_profile(
        proposed[0].profile_id,
        user_id="user_1",
        now_ms=30_000,
    )
    return repository, profiles, consolidator, active.profile_id


def test_observation_persists_valid_and_transaction_time_separately(
    tmp_path: Path,
) -> None:
    repository = MemoryRepository(tmp_path / "memory.sqlite3")
    repository.initialize()
    profiles = MemoryProfileRepository(repository.database_path)
    item = repository.add_active(
        user_id="user_1",
        candidate=_candidate(
            "用户偏好：回答时简短",
            turn_id="turn_1",
            valid_at_ms=1_000,
        ),
        now_ms=9_000,
    )

    observation = profiles.record_observation(
        item,
        session_id="session_1",
        now_ms=12_000,
    )

    assert observation is not None
    assert observation.valid_at_ms == 1_000
    assert observation.observed_at_ms == 12_000


def test_later_opposite_claim_without_explicit_transition_remains_conflict(
    tmp_path: Path,
) -> None:
    repository, profiles, consolidator, active_id = _active_baseline(tmp_path)
    detailed = repository.add_active(
        user_id="user_1",
        candidate=_candidate(
            "用户偏好：请详细展开解释",
            turn_id="turn_3",
            valid_at_ms=40_000,
        ),
        now_ms=40_000,
    )
    profiles.record_observation(detailed, session_id="session_3", now_ms=40_000)

    consolidator.rebuild_user("user_1", now_ms=50_000)

    assert profiles.list_changes("user_1") == []
    assert len(profiles.list_conflicts("user_1")) == 1
    assert profiles.get_profile(
        active_id,
        user_id="user_1",
    ).state is MemoryProfileState.STALE


def test_past_only_claim_does_not_conflict_with_current_profile(tmp_path: Path) -> None:
    repository, profiles, consolidator, active_id = _active_baseline(tmp_path)
    historical = repository.add_active(
        user_id="user_1",
        candidate=_candidate(
            "以前我喜欢详细回答",
            turn_id="turn_3",
            valid_at_ms=5_000,
        ),
        now_ms=40_000,
    )
    profiles.record_observation(historical, session_id="session_3", now_ms=40_000)

    consolidator.rebuild_user("user_1", now_ms=50_000)

    assert profiles.list_changes("user_1") == []
    assert profiles.list_conflicts("user_1") == []
    assert profiles.get_profile(
        active_id,
        user_id="user_1",
    ).state is MemoryProfileState.ACTIVE


def test_transition_uses_current_clause_when_old_value_appears_first(
    tmp_path: Path,
) -> None:
    repository = MemoryRepository(tmp_path / "memory.sqlite3")
    repository.initialize()
    profiles = MemoryProfileRepository(repository.database_path)
    consolidator = MemoryConsolidator(profiles)
    detailed = repository.add_active(
        user_id="user_1",
        candidate=_candidate(
            "用户偏好：请详细展开解释",
            turn_id="turn_1",
            valid_at_ms=1_000,
        ),
    )
    profiles.record_observation(detailed, session_id="session_1")
    profiles.record_observation(detailed, session_id="session_2")
    baseline = consolidator.rebuild_user("user_1")[0]
    profiles.confirm_profile(baseline.profile_id, user_id="user_1")
    transition = repository.add_active(
        user_id="user_1",
        candidate=_candidate(
            "以前我喜欢详细回答，现在改成简短结论",
            turn_id="turn_3",
            valid_at_ms=40_000,
        ),
    )
    profiles.record_observation(transition, session_id="session_3")

    consolidator.rebuild_user("user_1")

    change = profiles.list_changes("user_1", state=MemoryChangeState.OPEN)[0]
    assert change.proposed_profile.statement == "用户偏好：回答时先给简短结论"


def test_latest_transition_back_to_current_closes_an_unconfirmed_change(
    tmp_path: Path,
) -> None:
    repository, profiles, consolidator, active_id = _active_baseline(tmp_path)
    detailed = repository.add_active(
        user_id="user_1",
        candidate=_candidate(
            "以前我喜欢简短回答，现在改成详细解释",
            turn_id="turn_3",
            valid_at_ms=40_000,
        ),
    )
    profiles.record_observation(detailed, session_id="session_3")
    consolidator.rebuild_user("user_1")
    first = profiles.list_changes("user_1", state=MemoryChangeState.OPEN)[0]
    reverted = repository.add_active(
        user_id="user_1",
        candidate=_candidate(
            "以前想改成详细回答，现在还是改成简短结论",
            turn_id="turn_4",
            valid_at_ms=50_000,
        ),
    )
    profiles.record_observation(reverted, session_id="session_4")

    consolidator.rebuild_user("user_1")

    assert profiles.list_changes("user_1", state=MemoryChangeState.OPEN) == []
    assert profiles.get_change(
        first.change_id,
        user_id="user_1",
    ).state is MemoryChangeState.REJECTED
    assert profiles.get_profile(
        active_id,
        user_id="user_1",
    ).state is MemoryProfileState.ACTIVE


def test_change_before_current_profile_start_is_rejected(tmp_path: Path) -> None:
    repository, profiles, consolidator, _ = _active_baseline(tmp_path)
    invalid = repository.add_active(
        user_id="user_1",
        candidate=_candidate(
            "以前我喜欢简短回答，现在改成详细解释",
            turn_id="turn_3",
            valid_at_ms=500,
        ),
    )
    profiles.record_observation(invalid, session_id="session_3")
    consolidator.rebuild_user("user_1")
    change = profiles.list_changes("user_1", state=MemoryChangeState.OPEN)[0]

    with pytest.raises(ValueError, match="must start after"):
        profiles.apply_change(change.change_id, user_id="user_1")


def test_explicit_transition_creates_change_without_disabling_current_profile(
    tmp_path: Path,
) -> None:
    repository, profiles, consolidator, active_id = _active_baseline(tmp_path)
    changed = repository.add_active(
        user_id="user_1",
        candidate=_candidate(
            "以前我喜欢简短回答，现在改成详细解释",
            turn_id="turn_3",
            valid_at_ms=40_000,
        ),
        now_ms=45_000,
    )
    profiles.record_observation(changed, session_id="session_3", now_ms=45_000)

    consolidator.rebuild_user("user_1", now_ms=50_000)

    change = profiles.list_changes("user_1", state=MemoryChangeState.OPEN)[0]
    assert change.previous_profile_id == active_id
    assert change.effective_at_ms == 40_000
    assert change.observed_at_ms == 45_000
    assert change.proposed_profile.state is MemoryProfileState.AWAITING_CONFIRMATION
    assert profiles.get_profile(
        active_id,
        user_id="user_1",
    ).state is MemoryProfileState.ACTIVE
    assert profiles.list_conflicts("user_1") == []


def test_applying_change_builds_non_overlapping_bitemporal_profile_history(
    tmp_path: Path,
) -> None:
    repository, profiles, consolidator, active_id = _active_baseline(tmp_path)
    changed = repository.add_active(
        user_id="user_1",
        candidate=_candidate(
            "以前我喜欢简短回答，现在改成详细解释",
            turn_id="turn_3",
            valid_at_ms=40_000,
        ),
        now_ms=45_000,
    )
    profiles.record_observation(changed, session_id="session_3", now_ms=45_000)
    consolidator.rebuild_user("user_1", now_ms=50_000)
    change = profiles.list_changes("user_1", state=MemoryChangeState.OPEN)[0]

    applied = profiles.apply_change(change.change_id, user_id="user_1", now_ms=60_000)

    old = profiles.get_profile(active_id, user_id="user_1")
    current = profiles.get_profile(applied.proposed_profile_id, user_id="user_1")
    assert old.state is MemoryProfileState.HISTORICAL
    assert old.valid_to_ms == 40_000
    assert current.state is MemoryProfileState.ACTIVE
    assert current.valid_from_ms == 40_000
    assert profiles.list_active_profiles("user_1", as_of_ms=39_999) == [old]
    assert profiles.list_active_profiles("user_1", as_of_ms=40_000) == [current]


def test_rejecting_change_keeps_old_profile_and_suppresses_reproposal(
    tmp_path: Path,
) -> None:
    repository, profiles, consolidator, active_id = _active_baseline(tmp_path)
    changed = repository.add_active(
        user_id="user_1",
        candidate=_candidate(
            "以前我喜欢简短回答，现在改成详细解释",
            turn_id="turn_3",
            valid_at_ms=40_000,
        ),
    )
    profiles.record_observation(changed, session_id="session_3")
    consolidator.rebuild_user("user_1")
    change = profiles.list_changes("user_1", state=MemoryChangeState.OPEN)[0]

    profiles.reject_change(change.change_id, user_id="user_1")
    consolidator.rebuild_user("user_1")

    assert profiles.list_changes("user_1", state=MemoryChangeState.OPEN) == []
    assert profiles.get_profile(
        active_id,
        user_id="user_1",
    ).state is MemoryProfileState.ACTIVE
    assert profiles.get_profile(
        change.proposed_profile_id,
        user_id="user_1",
    ).state is MemoryProfileState.STALE


def test_later_support_enriches_applied_version_without_erasing_history(
    tmp_path: Path,
) -> None:
    repository, profiles, consolidator, old_profile_id = _active_baseline(tmp_path)
    changed = repository.add_active(
        user_id="user_1",
        candidate=_candidate(
            "以前我喜欢简短回答，现在改成详细解释",
            turn_id="turn_3",
            valid_at_ms=40_000,
        ),
        now_ms=45_000,
    )
    profiles.record_observation(changed, session_id="session_3", now_ms=45_000)
    consolidator.rebuild_user("user_1", now_ms=50_000)
    change = profiles.list_changes("user_1", state=MemoryChangeState.OPEN)[0]
    applied = profiles.apply_change(change.change_id, user_id="user_1", now_ms=60_000)
    corroboration = repository.add_active(
        user_id="user_1",
        candidate=_candidate(
            "用户偏好：请详细展开解释",
            turn_id="turn_4",
            valid_at_ms=70_000,
        ),
        now_ms=70_000,
    )
    profiles.record_observation(
        corroboration,
        session_id="session_4",
        now_ms=70_000,
    )

    consolidator.rebuild_user("user_1", now_ms=80_000)

    current = profiles.list_active_profiles("user_1", as_of_ms=80_000)
    old = profiles.get_profile(old_profile_id, user_id="user_1")
    assert [profile.profile_id for profile in current] == [applied.proposed_profile_id]
    assert current[0].distinct_session_count == 2
    assert current[0].valid_from_ms == 40_000
    assert profiles.list_active_profiles("user_1", as_of_ms=40_000) == [current[0]]
    assert old.state is MemoryProfileState.HISTORICAL
    assert old.valid_to_ms == 40_000


def test_revoked_change_evidence_cannot_be_applied(
    tmp_path: Path,
) -> None:
    repository, profiles, consolidator, _ = _active_baseline(tmp_path)
    changed = repository.add_active(
        user_id="user_1",
        candidate=_candidate(
            "以前我喜欢简短回答，现在改成详细解释",
            turn_id="turn_3",
            valid_at_ms=40_000,
        ),
    )
    profiles.record_observation(changed, session_id="session_3")
    consolidator.rebuild_user("user_1")
    change = profiles.list_changes("user_1", state=MemoryChangeState.OPEN)[0]

    repository.revoke(changed.memory_id, user_id="user_1")

    with pytest.raises(ValueError, match="already closed"):
        profiles.apply_change(change.change_id, user_id="user_1")


def test_old_database_migration_backfills_valid_time(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    repository = MemoryRepository(path)
    repository.initialize()
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO memory_observations(
                observation_id, memory_id, user_id, session_id, turn_id,
                valid_at_ms, observed_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("observation_1", "memory_1", "user_1", "session_1", "turn_1", 0, 9_000),
        )

    repository.initialize()

    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT valid_at_ms, observed_at_ms FROM memory_observations"
        ).fetchone()
    assert row == (9_000, 9_000)
