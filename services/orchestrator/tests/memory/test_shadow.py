import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path

import pytest

from app.events.store import EventStore
from app.memory.identity import MemorySubjectStore
from app.memory.models import MemoryAspect
from app.memory.repository import MemoryRepository
from app.memory.retrieval import RetrievedMemory
from app.memory.shadow import (
    LazyMemoryRetriever,
    MemoryShadowRepository,
    MemoryShadowRunner,
)


def retrieved(memory_id: str, text: str = "private memory text") -> RetrievedMemory:
    return RetrievedMemory(
        memory_id=memory_id,
        text=text,
        aspect=MemoryAspect.FACT,
        score=0.8,
        relevance_score=0.7,
        reason_codes=["TOPIC_MATCH"],
        source_turn_id="source_turn",
        source_message_ids=["message_1"],
    )


async def initialized_subject(database_path: Path) -> str:
    store = EventStore(database_path)
    await store.initialize()
    await store.reserve_session("session_1")
    MemoryRepository(database_path).initialize()
    subjects = MemorySubjectStore(database_path)
    user_id, _ = subjects.issue_or_resolve(
        session_id="session_1",
        supplied_token=None,
        now_ms=1_000,
    )
    return user_id


class StaticRetriever:
    def __init__(self, result: list[RetrievedMemory]) -> None:
        self.result = result
        self.calls = 0

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
        self.calls += 1
        return self.result[:k]


class SlowRetriever(StaticRetriever):
    def __init__(self, *, delay_seconds: float) -> None:
        super().__init__([])
        self.delay_seconds = delay_seconds

    def retrieve(
        self,
        query: str,
        *,
        user_id: str,
        purpose_scope: str = "personalization",
        k: int = 5,
        as_of_ms: int | None = None,
    ) -> list[RetrievedMemory]:
        time.sleep(self.delay_seconds)
        return super().retrieve(
            query,
            user_id=user_id,
            purpose_scope=purpose_scope,
            k=k,
            as_of_ms=as_of_ms,
        )


class GatedRetriever(StaticRetriever):
    def __init__(self) -> None:
        super().__init__([retrieved("memory_shadow")])
        self.started = threading.Event()
        self.release = threading.Event()

    def retrieve(
        self,
        query: str,
        *,
        user_id: str,
        purpose_scope: str = "personalization",
        k: int = 5,
        as_of_ms: int | None = None,
    ) -> list[RetrievedMemory]:
        self.started.set()
        self.release.wait(timeout=1)
        return super().retrieve(
            query,
            user_id=user_id,
            purpose_scope=purpose_scope,
            k=k,
            as_of_ms=as_of_ms,
        )


class PreparableRetriever(StaticRetriever):
    def __init__(self) -> None:
        super().__init__([retrieved("memory_1")])
        self.prepares = 0

    def prepare(self) -> None:
        self.prepares += 1


@pytest.mark.asyncio
async def test_shadow_ledger_excludes_raw_content_and_revoke_purges_it(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "events.sqlite3"
    user_id = await initialized_subject(database_path)
    repository = MemoryShadowRepository(database_path)
    now_ms = int(time.time() * 1000)
    repository.set_consent(
        user_id=user_id,
        granted=True,
        policy_version="policy-v1",
        now_ms=now_ms,
    )
    run_id = repository.create_run(
        user_id=user_id,
        session_id="session_1",
        turn_id="turn_1",
        policy_version="policy-v1",
        strategy_version="hybrid-v1",
        baseline=[retrieved("memory_baseline", "do not persist this memory text")],
        baseline_latency_ms=2.5,
        now_ms=now_ms + 1,
    )
    assert run_id is not None
    assert repository.complete_run(
        run_id,
        shadow=[retrieved("memory_shadow", "nor this shadow text")],
        latency_ms=7.5,
        completed_at_ms=now_ms + 2,
    )

    with closing(sqlite3.connect(database_path)) as connection:
        run_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(memory_shadow_runs)")
        }
        ranking_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(memory_shadow_rankings)")
        }
    all_columns = run_columns | ranking_columns
    assert not ({"query", "content", "text", "embedding"} & all_columns)
    report = repository.aggregate(user_id)
    assert report.completed_count == 1
    assert report.mean_overlap_at_5 == 0
    assert report.shadow_latency_ms_p95 == 7.5
    assert report.baseline_latency_ms_p95 == 2.5
    assert report.reliable is False
    assert report.minimum_reliable_runs == 100
    assert report.warnings

    assert repository.purge_expired(now_ms=now_ms + 10) == 0

    repository.set_consent(
        user_id=user_id,
        granted=False,
        policy_version="policy-v1",
        now_ms=now_ms + 3,
    )
    assert repository.list_runs(user_id) == []
    with closing(sqlite3.connect(database_path)) as connection:
        ranking_count = connection.execute(
            "SELECT COUNT(*) FROM memory_shadow_rankings"
        ).fetchone()
    assert ranking_count == (0,)


@pytest.mark.asyncio
async def test_shadow_retention_physically_expires_runs(tmp_path: Path) -> None:
    database_path = tmp_path / "events.sqlite3"
    user_id = await initialized_subject(database_path)
    repository = MemoryShadowRepository(database_path, retention_days=1)
    repository.set_consent(
        user_id=user_id,
        granted=True,
        policy_version="policy-v1",
        now_ms=1_000,
    )
    run_id = repository.create_run(
        user_id=user_id,
        session_id="session_1",
        turn_id="turn_expiring",
        policy_version="policy-v1",
        strategy_version="hybrid-v1",
        baseline=[retrieved("memory_baseline")],
        baseline_latency_ms=1,
        now_ms=2_000,
    )
    assert run_id is not None
    assert repository.purge_expired(now_ms=2_000 + 24 * 60 * 60 * 1000) == 1
    with closing(sqlite3.connect(database_path)) as connection:
        ranking_count = connection.execute(
            "SELECT COUNT(*) FROM memory_shadow_rankings"
        ).fetchone()
    assert ranking_count == (0,)


@pytest.mark.asyncio
async def test_runner_requires_consent_and_lazy_model_loads_only_after_submit(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "events.sqlite3"
    user_id = await initialized_subject(database_path)
    repository = MemoryShadowRepository(database_path)
    constructed = 0
    static = StaticRetriever([retrieved("memory_1")])

    def factory() -> StaticRetriever:
        nonlocal constructed
        constructed += 1
        return static

    runner = MemoryShadowRunner(
        repository=repository,
        retriever=LazyMemoryRetriever(factory),
        strategy_version="hybrid-v1",
        policy_version="policy-v1",
    )
    arguments = {
        "user_id": user_id,
        "session_id": "session_1",
        "turn_id": "turn_1",
        "query": "raw query stays in process",
        "baseline": [retrieved("memory_1")],
        "baseline_latency_ms": 1.0,
    }
    assert not runner.submit(**arguments)
    assert constructed == 0

    repository.set_consent(
        user_id=user_id,
        granted=True,
        policy_version="policy-v1",
    )
    assert runner.submit(**arguments)
    await runner.drain()
    assert constructed == 1
    assert static.calls == 1
    assert repository.aggregate(user_id).completed_count == 1


@pytest.mark.asyncio
async def test_consent_prewarm_is_idempotent_and_exposes_model_state(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "events.sqlite3"
    await initialized_subject(database_path)
    retriever = PreparableRetriever()
    runner = MemoryShadowRunner(
        repository=MemoryShadowRepository(database_path),
        retriever=retriever,
        strategy_version="hybrid-v1",
    )
    assert runner.status().model_state == "cold"

    runner.start_prewarm()
    runner.start_prewarm()
    assert runner.status().model_state == "warming"
    await runner.close()

    assert retriever.prepares == 1
    assert runner.status().model_state == "ready"


@pytest.mark.asyncio
async def test_timeout_opens_circuit_without_affecting_caller(tmp_path: Path) -> None:
    database_path = tmp_path / "events.sqlite3"
    user_id = await initialized_subject(database_path)
    repository = MemoryShadowRepository(database_path)
    repository.set_consent(
        user_id=user_id,
        granted=True,
        policy_version="policy-v1",
    )
    runner = MemoryShadowRunner(
        repository=repository,
        retriever=SlowRetriever(delay_seconds=0.05),
        strategy_version="hybrid-v1",
        policy_version="policy-v1",
        timeout_seconds=0.005,
        failure_threshold=1,
        cooldown_seconds=60,
    )
    assert runner.submit(
        user_id=user_id,
        session_id="session_1",
        turn_id="turn_timeout",
        query="query",
        baseline=[],
        baseline_latency_ms=1,
    )
    await runner.drain()
    assert not runner.submit(
        user_id=user_id,
        session_id="session_1",
        turn_id="turn_circuit",
        query="query",
        baseline=[],
        baseline_latency_ms=1,
    )
    report = repository.aggregate(user_id)
    assert report.timeout_count == 1
    assert report.circuit_open_count == 1
    assert report.shadow_latency_ms_p95 is not None


@pytest.mark.asyncio
async def test_revocation_cancels_inflight_result_and_removes_pending_run(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "events.sqlite3"
    user_id = await initialized_subject(database_path)
    repository = MemoryShadowRepository(database_path)
    repository.set_consent(
        user_id=user_id,
        granted=True,
        policy_version="policy-v1",
    )
    retriever = GatedRetriever()
    runner = MemoryShadowRunner(
        repository=repository,
        retriever=retriever,
        strategy_version="hybrid-v1",
        policy_version="policy-v1",
    )
    assert runner.submit(
        user_id=user_id,
        session_id="session_1",
        turn_id="turn_1",
        query="query",
        baseline=[],
        baseline_latency_ms=1,
    )
    assert await asyncio_to_thread_wait(retriever.started)
    repository.set_consent(
        user_id=user_id,
        granted=False,
        policy_version="policy-v1",
    )
    runner.cancel_user(user_id)
    retriever.release.set()
    await runner.drain()
    assert repository.list_runs(user_id) == []


async def asyncio_to_thread_wait(event: threading.Event) -> bool:
    import asyncio

    return await asyncio.to_thread(event.wait, 1)
