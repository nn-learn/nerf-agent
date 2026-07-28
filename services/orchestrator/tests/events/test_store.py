import sqlite3

import pytest

from app.contracts.events import EventEnvelope
from app.events.store import (
    EventSequenceError,
    EventStore,
    MediaPersistenceError,
)


@pytest.mark.asyncio
async def test_store_refuses_nested_raw_camera_bytes(tmp_path) -> None:
    """Catches raw media hidden inside nested payloads before JSON persistence."""
    store = EventStore(tmp_path / "events.sqlite3")
    await store.initialize()

    with pytest.raises(MediaPersistenceError, match=r"payload\.frames\[0\]\.jpeg"):
        await store.append_payload(
            session_id="session_1",
            event_type="vision.frame.sampled",
            payload={"frames": [{"jpeg": b"\xff\xd8\xff"}]},
        )

    assert await store.list_session("session_1") == []


@pytest.mark.asyncio
async def test_store_orders_events_and_ignores_duplicate_event_ids(tmp_path) -> None:
    """Catches duplicate delivery or persistence that reorders a session timeline."""
    store = EventStore(tmp_path / "events.sqlite3")
    await store.initialize()
    first = EventEnvelope[dict[str, str]](
        event_id="evt_1",
        session_id="session_1",
        turn_id="turn_1",
        trace_id="trace_1",
        seq=1,
        type="transcript.final",
        timestamp_ms=1_000,
        cancel_token="ct_1",
        payload={"text": "第一条"},
    )
    second = first.model_copy(
        update={
            "event_id": "evt_2",
            "seq": 2,
            "type": "risk.updated",
            "payload": {"level": "GREEN"},
        }
    )

    assert await store.append(first)
    assert not await store.append(first)
    assert await store.append(second)

    restored = await store.list_session("session_1")
    assert [event.event_id for event in restored] == ["evt_1", "evt_2"]
    assert [event.seq for event in restored] == [1, 2]


@pytest.mark.asyncio
async def test_store_rejects_non_monotonic_session_sequence(tmp_path) -> None:
    """Catches late events that would make replay order ambiguous."""
    store = EventStore(tmp_path / "events.sqlite3")
    await store.initialize()
    await store.append_payload(
        session_id="session_1",
        event_type="transcript.final",
        payload={"text": "第一条"},
    )
    stale = EventEnvelope[dict[str, str]](
        event_id="evt_stale",
        session_id="session_1",
        turn_id="turn_1",
        trace_id="trace_1",
        seq=1,
        type="risk.updated",
        timestamp_ms=2_000,
        cancel_token="ct_1",
        payload={"level": "GREEN"},
    )

    with pytest.raises(EventSequenceError, match="greater than 1"):
        await store.append(stale)


@pytest.mark.asyncio
async def test_visual_observation_schema_has_no_binary_column(tmp_path) -> None:
    """Catches schema changes that create a path for persisted camera frames."""
    database_path = tmp_path / "events.sqlite3"
    store = EventStore(database_path)
    await store.initialize()

    with sqlite3.connect(database_path) as connection:
        columns = connection.execute(
            "PRAGMA table_info(visual_observations)"
        ).fetchall()

    assert columns
    assert all(str(column[2]).upper() != "BLOB" for column in columns)

