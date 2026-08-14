import sqlite3
from pathlib import Path

from app.memory.consolidation import MemoryProfileRepository
from app.memory.episodes import (
    EpisodeAugmentedMemoryRetriever,
    FreshnessAwareMemoryRetriever,
    MemoryEpisodeRepository,
)
from app.memory.models import MemoryAspect, MemoryCandidate, MemoryKind
from app.memory.repository import MemoryRepository
from app.memory.retrieval import GovernedMemoryRetriever, RetrievedMemory


def _candidate(
    text: str,
    *,
    turn_id: str,
    subject_key: str,
    aspect: MemoryAspect = MemoryAspect.FACT,
    valid_from_ms: int = 1_000,
) -> MemoryCandidate:
    return MemoryCandidate(
        source="transcript",
        contains_sensitive_content=False,
        text=text,
        kind=MemoryKind.EPISODIC,
        source_turn_id=turn_id,
        aspect=aspect,
        subject_key=subject_key,
        confidence=0.9,
        valid_from_ms=valid_from_ms,
    )


def test_episode_summary_is_bounded_idempotent_and_uses_only_active_memories(
    tmp_path: Path,
) -> None:
    repository = MemoryRepository(tmp_path / "memory.sqlite3")
    repository.initialize()
    profiles = MemoryProfileRepository(repository.database_path)
    episodes = MemoryEpisodeRepository(
        repository.database_path,
        max_members=2,
        max_summary_chars=80,
    )
    first = repository.add_active(
        user_id="user_1",
        candidate=_candidate(
            "用户事实：下周要去上海出差",
            turn_id="turn_1",
            subject_key="travel.next_trip",
        ),
        now_ms=2_000,
    )
    second = repository.add_active(
        user_id="user_1",
        candidate=_candidate(
            "用户事实：猫咪豆豆需要朋友照顾",
            turn_id="turn_2",
            subject_key="pet.care",
            valid_from_ms=1_100,
        ),
        now_ms=2_100,
    )
    awaiting = repository.add_awaiting_consent(
        user_id="user_1",
        candidate=_candidate(
            "用户事实：这条尚未确认",
            turn_id="turn_3",
            subject_key="unconfirmed.fact",
        ),
        now_ms=2_200,
    )
    profiles.record_observation(first, session_id="session_1", now_ms=2_000)
    profiles.record_observation(second, session_id="session_1", now_ms=2_100)
    assert profiles.record_observation(
        awaiting,
        session_id="session_1",
        now_ms=2_200,
    ) is None

    built = episodes.rebuild_session(
        user_id="user_1",
        session_id="session_1",
        memory_repository=repository,
        now_ms=3_000,
    )
    rebuilt = episodes.rebuild_session(
        user_id="user_1",
        session_id="session_1",
        memory_repository=repository,
        now_ms=4_000,
    )

    assert len(built) == 1
    assert built[0].member_memory_ids == [first.memory_id, second.memory_id]
    assert awaiting.memory_id not in built[0].member_memory_ids
    assert len(built[0].text) <= 80
    assert rebuilt[0].summary_id == built[0].summary_id
    assert rebuilt[0].source_digest == built[0].source_digest


def test_episode_summary_expands_governed_sources_but_never_enters_context(
    tmp_path: Path,
) -> None:
    repository = MemoryRepository(tmp_path / "memory.sqlite3")
    repository.initialize()
    profiles = MemoryProfileRepository(repository.database_path)
    episodes = MemoryEpisodeRepository(repository.database_path)
    trip = repository.add_active(
        user_id="user_1",
        candidate=_candidate(
            "用户事实：下周有一项上海出差安排",
            turn_id="turn_1",
            subject_key="travel.next_trip",
        ),
    )
    pet = repository.add_active(
        user_id="user_1",
        candidate=_candidate(
            "用户事实：猫咪豆豆需要朋友照顾",
            turn_id="turn_2",
            subject_key="pet.care",
            valid_from_ms=1_100,
        ),
    )
    profiles.record_observation(trip, session_id="session_1")
    profiles.record_observation(pet, session_id="session_1")
    summary = episodes.rebuild_session(
        user_id="user_1",
        session_id="session_1",
        memory_repository=repository,
    )[0]
    base = GovernedMemoryRetriever(repository)
    flat = base.retrieve("上海出差安排", user_id="user_1", k=3, as_of_ms=2_000)
    augmented = EpisodeAugmentedMemoryRetriever(
        base,
        episodes=episodes,
        memories=repository,
    )

    recalled = augmented.retrieve(
        "上海出差安排",
        user_id="user_1",
        k=3,
        as_of_ms=2_000,
    )
    context = augmented.to_model_context(recalled)

    assert [item.memory_id for item in flat] == [trip.memory_id]
    assert {item.memory_id for item in recalled} == {trip.memory_id, pet.memory_id}
    expanded = next(item for item in recalled if item.memory_id == pet.memory_id)
    assert "EPISODE_SUMMARY_EXPANSION" in expanded.reason_codes
    assert {str(item["fact"]) for item in context} == {
        trip.candidate.text,
        pet.candidate.text,
    }
    assert summary.text not in {str(item["fact"]) for item in context}


def test_revocation_physically_removes_episode_summary_text(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory.sqlite3")
    repository.initialize()
    profiles = MemoryProfileRepository(repository.database_path)
    episodes = MemoryEpisodeRepository(repository.database_path)
    marker = "仅用于删除验证的派生摘要内容"
    item = repository.add_active(
        user_id="user_1",
        candidate=_candidate(
            marker,
            turn_id="turn_1",
            subject_key="deletion.marker",
        ),
    )
    profiles.record_observation(item, session_id="session_1")
    episodes.rebuild_session(
        user_id="user_1",
        session_id="session_1",
        memory_repository=repository,
    )

    repository.revoke(item.memory_id, user_id="user_1")

    assert episodes.list_for_user("user_1") == []
    with sqlite3.connect(repository.database_path) as connection:
        rows = connection.execute(
            "SELECT summary_text FROM memory_episode_summaries"
        ).fetchall()
    assert marker not in str(rows)


class _StaticRetriever:
    def __init__(self, items: list[RetrievedMemory]) -> None:
        self.items = items

    def retrieve(
        self,
        query: str,
        *,
        user_id: str,
        purpose_scope: str = "personalization",
        k: int = 5,
        as_of_ms: int | None = None,
    ) -> list[RetrievedMemory]:
        _ = (query, user_id, purpose_scope, as_of_ms)
        return self.items[:k]


def test_freshness_only_reranks_and_exempts_governed_preferences() -> None:
    day_ms = 86_400_000
    fact = RetrievedMemory(
        memory_id="memory_fact",
        text="用户事实：旧事实",
        aspect=MemoryAspect.FACT,
        score=0.8,
        relevance_score=0.8,
        reason_codes=["USER_CONFIRMED"],
        source_turn_id="turn_1",
        source_message_ids=[],
        valid_at_ms=0,
    )
    preference = RetrievedMemory(
        memory_id="memory_preference",
        text="用户偏好：先给结论",
        aspect=MemoryAspect.PREFERENCE,
        score=0.8,
        relevance_score=0.8,
        reason_codes=["USER_CONFIRMED"],
        source_turn_id="turn_2",
        source_message_ids=[],
        valid_at_ms=0,
    )
    retriever = FreshnessAwareMemoryRetriever(_StaticRetriever([fact, preference]))

    results = retriever.retrieve(
        "测试",
        user_id="user_1",
        k=2,
        as_of_ms=730 * day_ms,
    )

    assert {item.memory_id for item in results} == {fact.memory_id, preference.memory_id}
    fact_result = next(item for item in results if item.memory_id == fact.memory_id)
    preference_result = next(
        item for item in results if item.memory_id == preference.memory_id
    )
    assert fact_result.score < fact.score
    assert "FRESHNESS_RERANKED" in fact_result.reason_codes
    assert preference_result.score == preference.score
    assert "FRESHNESS_EXEMPT" in preference_result.reason_codes
