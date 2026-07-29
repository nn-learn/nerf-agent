import pytest

from avatar.worker.service import AvatarWorker


@pytest.mark.asyncio
async def test_cancel_discards_old_audio_and_frames() -> None:
    """Catches stale lip-sync work surviving a user barge-in."""
    worker = AvatarWorker()
    await worker.start_session("session_1")
    await worker.enqueue_audio(
        turn_id="turn_1",
        cancel_token="token_old",
        pcm_s16le=b"\x00\x00" * 320,
    )
    await worker.enqueue_frame(
        turn_id="turn_1",
        cancel_token="token_old",
        rgb=b"synthetic-rgb-frame",
        pts_ms=2_000,
    )

    await worker.cancel_turn(
        turn_id="turn_1",
        cancel_token="token_old",
    )

    assert worker.audio_queue_size == 0
    assert worker.frame_queue_size == 0
    assert worker.state == "neutral_listening"


@pytest.mark.asyncio
async def test_stale_token_cannot_reenter_avatar_queue() -> None:
    """Catches late gRPC delivery resurrecting a cancelled turn."""
    worker = AvatarWorker()
    await worker.start_session("session_1")
    await worker.cancel_turn(
        turn_id="turn_1",
        cancel_token="token_old",
    )

    accepted = await worker.enqueue_audio(
        turn_id="turn_1",
        cancel_token="token_old",
        pcm_s16le=b"\x00\x00" * 320,
    )

    assert not accepted
    assert worker.audio_queue_size == 0
