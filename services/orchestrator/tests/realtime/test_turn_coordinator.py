import asyncio

import pytest

from app.contracts.session import CancellationRegistry
from app.realtime.audio_publisher import AudioPublisher
from app.realtime.models import PcmChunk
from app.realtime.turn_coordinator import TurnCoordinator


@pytest.mark.asyncio
async def test_barge_in_stops_every_downstream_queue_within_one_tick() -> None:
    """Catches stale assistant audio or Avatar frames surviving user interruption."""
    coordinator = TurnCoordinator()
    turn = await coordinator.start_turn("session_1")
    for sequence in range(20):
        await coordinator.enqueue_audio(
            turn,
            PcmChunk(
                sequence=sequence,
                pts_ms=sequence * 20,
                pcm_s16le=b"\x00\x00" * 320,
            ),
        )
        await coordinator.enqueue_avatar_frame(
            turn,
            frame_id=f"frame_{sequence}",
        )

    result = await coordinator.interrupt("session_1", reason="user_speech")

    assert result.cancelled_token == turn.cancel_token
    assert result.reason == "user_speech"
    assert coordinator.audio_queue_size == 0
    assert coordinator.avatar_queue_size == 0
    assert not await coordinator.tokens.is_current(
        turn.turn_id,
        turn.cancel_token,
    )


@pytest.mark.asyncio
async def test_stale_turn_cannot_enqueue_after_interruption() -> None:
    """Catches late provider chunks re-entering playback after cancellation."""
    coordinator = TurnCoordinator()
    turn = await coordinator.start_turn("session_1")
    await coordinator.interrupt("session_1", reason="user_speech")

    accepted = await coordinator.enqueue_audio(
        turn,
        PcmChunk(
            sequence=0,
            pts_ms=0,
            pcm_s16le=b"\x00\x00" * 320,
        ),
    )

    assert not accepted
    assert coordinator.audio_queue_size == 0


@pytest.mark.asyncio
async def test_audio_publishes_after_avatar_ready_or_short_timeout() -> None:
    """Catches Avatar degradation blocking voice playback indefinitely."""
    registry = CancellationRegistry()
    token = await registry.issue("turn_1")
    published: list[PcmChunk] = []

    async def chunks():
        yield PcmChunk(
            sequence=0,
            pts_ms=0,
            pcm_s16le=b"\x00\x00" * 320,
        )

    async def sink(chunk: PcmChunk) -> None:
        published.append(chunk)

    publisher = AudioPublisher(
        registry=registry,
        first_frame_timeout_seconds=0.001,
    )
    result = await publisher.publish(
        chunks(),
        avatar_ready=asyncio.Event(),
        sink=sink,
        turn_id="turn_1",
        cancel_token=token,
    )

    assert not result.avatar_ready_before_audio
    assert result.published_chunks == 1
    assert published[0].sequence == 0
