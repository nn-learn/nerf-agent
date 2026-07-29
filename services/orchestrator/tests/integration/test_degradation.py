import pytest

from app.events.store import EventStore
from app.realtime.avatar_client import MockAvatarClient
from app.realtime.session import (
    MockAudioBridge,
    MockVisionBridge,
    SessionManager,
    TurnStatus,
)


@pytest.mark.asyncio
async def test_vision_failure_keeps_voice_and_avatar(tmp_path) -> None:
    store = EventStore(tmp_path / "events.sqlite3")
    manager = SessionManager(
        store=store,
        vision_bridge=MockVisionBridge(available=False),
    )
    session = await manager.create_session(camera_consent=True)

    result = await manager.process_text_turn(
        session.session_id,
        text="帮我看看这张卡片。",
        visual_summary="一张睡眠记录卡。",
    )

    events = await store.list_session(session.session_id)
    event_types = [event.type for event in events]
    assert result.status is TurnStatus.COMPLETED
    assert result.delivery_mode == "voice_avatar"
    assert "vision.degraded" in event_types
    assert "playback.started" in event_types


@pytest.mark.asyncio
async def test_avatar_failure_falls_back_to_voice_and_text(tmp_path) -> None:
    store = EventStore(tmp_path / "events.sqlite3")
    manager = SessionManager(
        store=store,
        avatar_client=MockAvatarClient(available=False),
    )
    session = await manager.create_session(camera_consent=False)

    result = await manager.process_text_turn(
        session.session_id,
        text="今天睡得不太好。",
    )

    events = await store.list_session(session.session_id)
    event_types = [event.type for event in events]
    assert result.status is TurnStatus.COMPLETED
    assert result.delivery_mode == "voice_text"
    assert "avatar.degraded" in event_types
    assert "playback.started" in event_types


@pytest.mark.asyncio
async def test_tts_failure_falls_back_to_text_without_blocking(tmp_path) -> None:
    store = EventStore(tmp_path / "events.sqlite3")
    manager = SessionManager(
        store=store,
        audio_bridge=MockAudioBridge(available=False),
    )
    session = await manager.create_session(camera_consent=False)

    result = await manager.process_text_turn(
        session.session_id,
        text="我想用文字继续。",
    )

    events = await store.list_session(session.session_id)
    event_types = [event.type for event in events]
    assert result.status is TurnStatus.COMPLETED
    assert result.delivery_mode == "text"
    assert "tts.degraded" in event_types
    assert "response.displayed" in event_types
    assert "avatar.frame.ready" not in event_types
