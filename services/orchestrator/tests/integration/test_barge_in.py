import asyncio

import pytest

from app.events.store import EventStore
from app.realtime.avatar_client import AvatarRenderResult
from app.realtime.models import TurnHandle
from app.realtime.session import SessionManager, TurnStatus


class BlockingAvatarClient:
    def __init__(self) -> None:
        self.render_started = asyncio.Event()
        self.release_render = asyncio.Event()
        self.cancelled_turns: list[str] = []

    async def start_session(self, session_id: str) -> None:
        _ = session_id

    async def render(
        self,
        turn: TurnHandle,
        *,
        style: str,
        audio_chunk_count: int,
    ) -> AvatarRenderResult:
        _ = (style, audio_chunk_count)
        self.render_started.set()
        await self.release_render.wait()
        return AvatarRenderResult(frame_ids=[f"frame_{turn.turn_id}"])

    async def cancel_turn(self, turn: TurnHandle) -> None:
        self.cancelled_turns.append(turn.turn_id)
        self.release_render.set()

    async def end_session(self, session_id: str) -> None:
        _ = session_id


@pytest.mark.asyncio
async def test_barge_in_prevents_stale_avatar_and_playback_events(tmp_path) -> None:
    """Catches late Avatar output appearing after the user has interrupted."""
    store = EventStore(tmp_path / "events.sqlite3")
    avatar = BlockingAvatarClient()
    manager = SessionManager(store=store, avatar_client=avatar)
    session = await manager.create_session(camera_consent=False)

    running_turn = asyncio.create_task(
        manager.process_text_turn(
            session.session_id,
            text="我最近有点累。",
        )
    )
    await asyncio.wait_for(avatar.render_started.wait(), timeout=1)

    interrupted = await manager.interrupt(
        session.session_id,
        reason="user_speech",
    )
    result = await asyncio.wait_for(running_turn, timeout=1)

    assert interrupted.interrupted
    assert result.status is TurnStatus.INTERRUPTED
    assert avatar.cancelled_turns == [result.turn_id]

    events = await store.list_session(session.session_id)
    event_types = [event.type for event in events]
    interruption_index = event_types.index("playback.interrupted")
    assert "avatar.frame.ready" not in event_types[interruption_index + 1 :]
    assert "playback.started" not in event_types[interruption_index + 1 :]


@pytest.mark.asyncio
async def test_interrupt_is_idempotent_when_no_turn_is_active(tmp_path) -> None:
    store = EventStore(tmp_path / "events.sqlite3")
    manager = SessionManager(store=store)
    session = await manager.create_session(camera_consent=False)

    result = await manager.interrupt(session.session_id, reason="user_speech")

    assert not result.interrupted
    assert result.turn_id is None
