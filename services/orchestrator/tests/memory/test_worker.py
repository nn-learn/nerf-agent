import asyncio
from pathlib import Path

import pytest

from app.events.store import EventStore
from app.memory.extraction import RuleBasedMemoryExtractor
from app.memory.identity import MemorySubjectStore
from app.memory.pipeline import MemoryPipeline
from app.memory.reader import EventMessageReader
from app.memory.repository import MemoryRepository
from app.memory.service import MemoryIngestionService
from app.memory.worker import MemoryIngestionWorker


@pytest.mark.asyncio
async def test_worker_ingests_ended_session_off_event_loop(tmp_path: Path) -> None:
    database_path = tmp_path / "events.sqlite3"
    store = EventStore(database_path)
    await store.initialize()
    await store.reserve_session("session_1")
    repository = MemoryRepository(database_path)
    repository.initialize()
    subjects = MemorySubjectStore(database_path)
    user_id, _ = subjects.issue_or_resolve(
        session_id="session_1",
        supplied_token=None,
    )
    await store.append_payload(
        session_id="session_1",
        turn_id="turn_1",
        event_type="transcript.final",
        payload={"text": "我更喜欢简短回答"},
    )
    worker = MemoryIngestionWorker(
        service=MemoryIngestionService(
            reader=EventMessageReader(database_path),
            pipeline=MemoryPipeline(
                repository=repository,
                extractor=RuleBasedMemoryExtractor(),
            ),
            repository=repository,
        ),
        subjects=subjects,
    )
    worker.start()
    try:
        await worker.enqueue("session_1")
        await asyncio.wait_for(worker.join(), timeout=2)
    finally:
        await worker.stop()

    assert worker.completed_count == 1
    assert worker.failed_count == 0
    assert len(repository.list_awaiting_consent(user_id)) == 1


@pytest.mark.asyncio
async def test_pending_ended_sessions_are_recoverable_after_restart(tmp_path: Path) -> None:
    database_path = tmp_path / "events.sqlite3"
    store = EventStore(database_path)
    await store.initialize()
    await store.reserve_session("session_1")
    MemoryRepository(database_path).initialize()
    subjects = MemorySubjectStore(database_path)
    subjects.issue_or_resolve(session_id="session_1", supplied_token=None)
    await store.append_payload(
        session_id="session_1",
        event_type="transcript.final",
        payload={"text": "我喜欢简短回答"},
    )
    await store.append_payload(
        session_id="session_1",
        event_type="session.ended",
        payload={"reason": "user_ended"},
    )

    assert subjects.pending_ended_sessions() == ["session_1"]


@pytest.mark.asyncio
async def test_queue_overflow_never_blocks_session_shutdown(tmp_path: Path) -> None:
    database_path = tmp_path / "events.sqlite3"
    repository = MemoryRepository(database_path)
    repository.initialize()
    worker = MemoryIngestionWorker(
        service=MemoryIngestionService(
            reader=EventMessageReader(database_path),
            pipeline=MemoryPipeline(
                repository=repository,
                extractor=RuleBasedMemoryExtractor(),
            ),
            repository=repository,
        ),
        subjects=MemorySubjectStore(database_path),
        max_queue_size=1,
    )

    await asyncio.wait_for(worker.enqueue("session_1"), timeout=0.1)
    await asyncio.wait_for(worker.enqueue("session_2"), timeout=0.1)

    assert worker.queue_size == 1
    assert worker.overflow_count == 1
    assert worker.session_state("session_1") == "queued"
    assert worker.session_state("session_2") is None
