import pytest
from pydantic import ValidationError

from app.contracts.events import EventEnvelope
from app.contracts.session import CancellationRegistry


def test_event_sequence_must_be_positive() -> None:
    """Catches event producers that emit zero or negative session sequence numbers."""
    with pytest.raises(ValidationError):
        EventEnvelope[dict[str, str]](
            event_id="evt_1",
            session_id="session_1",
            turn_id="turn_1",
            trace_id="trace_1",
            seq=0,
            type="transcript.final",
            timestamp_ms=1_785_240_000_000,
            cancel_token="ct_1",
            payload={"text": "你好"},
        )


def test_event_envelope_round_trips_payload_without_losing_trace_fields() -> None:
    """Catches serialization changes that drop correlation or cancellation fields."""
    event = EventEnvelope[dict[str, str]](
        event_id="evt_1",
        session_id="session_1",
        turn_id="turn_1",
        trace_id="trace_1",
        seq=7,
        type="transcript.final",
        timestamp_ms=1_785_240_000_000,
        cancel_token="ct_1",
        payload={"text": "你好"},
    )

    restored = EventEnvelope[dict[str, str]].model_validate_json(event.model_dump_json())

    assert restored == event
    assert restored.payload == {"text": "你好"}


@pytest.mark.asyncio
async def test_issuing_a_new_token_invalidates_the_previous_turn_token() -> None:
    """Catches stale model or media work that remains publishable after cancellation."""
    registry = CancellationRegistry()
    first = await registry.issue("turn_1")
    second = await registry.issue("turn_1")

    assert first != second
    assert not await registry.is_current("turn_1", first)
    assert await registry.is_current("turn_1", second)

    await registry.cancel(second)

    assert not await registry.is_current("turn_1", second)

