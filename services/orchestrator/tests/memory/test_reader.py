from pathlib import Path

import pytest

from app.events.store import EventStore
from app.memory.extraction import RuleBasedMemoryExtractor
from app.memory.models import MessageRole
from app.memory.pipeline import MemoryPipeline
from app.memory.reader import EventMessageReader
from app.memory.repository import MemoryRepository
from app.memory.service import MemoryIngestionService


@pytest.mark.asyncio
async def test_event_reader_uses_keyset_batches_and_ignores_non_message_events(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "events.sqlite3"
    store = EventStore(database_path)
    await store.initialize()
    for index in range(3):
        await store.append_payload(
            session_id="session_1",
            turn_id=f"turn_{index}",
            event_type="transcript.final",
            payload={"text": f"用户消息{index}"},
        )
        await store.append_payload(
            session_id="session_1",
            turn_id=f"turn_{index}",
            event_type="risk.updated",
            payload={"level": "GREEN"},
        )
        await store.append_payload(
            session_id="session_1",
            turn_id=f"turn_{index}",
            event_type="assistant.response.ready",
            payload={"display_text": f"助手回复{index}"},
        )

    batches = list(EventMessageReader(database_path, batch_size=2).iter_batches("session_1"))

    assert [len(batch) for batch in batches] == [2, 2, 2]
    messages = [item for batch in batches for item in batch]
    assert [item.role for item in messages] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]


@pytest.mark.asyncio
async def test_session_ingestion_is_incremental_and_idempotent(tmp_path: Path) -> None:
    database_path = tmp_path / "events.sqlite3"
    store = EventStore(database_path)
    await store.initialize()
    await store.append_payload(
        session_id="session_1",
        turn_id="turn_1",
        event_type="transcript.final",
        payload={"text": "我更喜欢简短回答"},
    )
    repository = MemoryRepository(database_path)
    repository.initialize()
    service = MemoryIngestionService(
        reader=EventMessageReader(database_path, batch_size=1),
        pipeline=MemoryPipeline(
            repository=repository,
            extractor=RuleBasedMemoryExtractor(),
        ),
        repository=repository,
    )

    first = service.ingest_session(
        user_id="user_1",
        session_id="session_1",
        consent_granted=False,
        now_ms=2_000,
    )
    second = service.ingest_session(
        user_id="user_1",
        session_id="session_1",
        consent_granted=False,
        now_ms=3_000,
    )

    assert first.metrics.candidate_count == 1
    assert second.metrics.message_count == 0
    assert len(repository.list_awaiting_consent("user_1")) == 1
