import pytest
from httpx import ASGITransport, AsyncClient

from app.events.store import EventStore
from app.main import create_app
from app.realtime.session import SessionManager, TurnStatus
from app.safety.models import RiskLevel
from app.settings import Settings


@pytest.mark.asyncio
async def test_visual_voice_turn_uses_one_trace_and_ordered_events(tmp_path) -> None:
    """Catches multimodal providers drifting into unrelated audit traces."""
    store = EventStore(tmp_path / "events.sqlite3")
    manager = SessionManager(store=store)
    session = await manager.create_session(camera_consent=True)

    result = await manager.process_text_turn(
        session.session_id,
        text="最近工作压力有点大，帮我看看这张睡眠记录。",
        visual_summary="画面中可见一张睡眠记录卡，显示多次夜间醒来。",
    )

    assert result.status is TurnStatus.COMPLETED
    assert result.risk_level is RiskLevel.GREEN
    assert result.delivery_mode == "voice_avatar"

    events = await store.list_session(session.session_id)
    turn_events = [event for event in events if event.turn_id == result.turn_id]
    assert [event.type for event in turn_events] == [
        "transcript.final",
        "vision.burst.requested",
        "vision.observation.ready",
        "risk.updated",
        "retrieval.completed",
        "assistant.response.ready",
        "tts.audio.chunk",
        "avatar.frame.ready",
        "playback.started",
        "turn.completed",
    ]
    traced_types = {
        "transcript.final",
        "vision.observation.ready",
        "risk.updated",
        "assistant.response.ready",
        "playback.started",
    }
    assert {
        event.trace_id for event in turn_events if event.type in traced_types
    } == {result.trace_id}


@pytest.mark.asyncio
async def test_session_http_fallback_supports_typed_turns(tmp_path) -> None:
    """Catches a LiveKit outage making the support Agent completely unusable."""
    store = EventStore(tmp_path / "events.sqlite3")
    manager = SessionManager(store=store)
    session = await manager.create_session(camera_consent=False)

    result = await manager.process_text_turn(
        session.session_id,
        text="我想先用文字聊一聊。",
    )

    assert result.status is TurnStatus.COMPLETED
    assert result.delivery_mode == "voice_avatar"
    assert "http_text" in session.fallback_capabilities


@pytest.mark.asyncio
async def test_client_cannot_reclaim_an_existing_session_id(tmp_path) -> None:
    """Catches unauthenticated session creation leaking an existing access token."""
    store = EventStore(tmp_path / "events.sqlite3")
    manager = SessionManager(store=store)

    first = await manager.create_session(
        requested_id="session_browser_1",
        camera_consent=False,
    )
    with pytest.raises(ValueError, match="already exists"):
        await manager.create_session(
            requested_id="session_browser_1",
            camera_consent=False,
        )

    events = await store.list_session(first.session_id)
    assert [event.type for event in events] == ["session.started"]


@pytest.mark.asyncio
async def test_session_api_creates_turns_and_ends_cleanly(tmp_path) -> None:
    """Catches the browser shell depending on orchestration routes that do not exist."""
    settings = Settings(
        provider_mode="mock",
        event_database_path=tmp_path / "events.sqlite3",
    )
    transport = ASGITransport(app=create_app(settings))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post(
            "/api/sessions",
            json={"camera_consent": False},
        )
        session_id = created.json()["session_id"]
        headers = {
            "Authorization": f"Bearer {created.json()['access_token']}"
        }
        turn = await client.post(
            f"/api/sessions/{session_id}/turns",
            headers=headers,
            json={"text": "我想先聊聊最近的睡眠。"},
        )
        inactive_interrupt = await client.post(
            f"/api/sessions/{session_id}/interrupt",
            headers=headers,
            json={"reason": "user_speech"},
        )
        ended = await client.delete(
            f"/api/sessions/{session_id}",
            headers=headers,
        )

    assert created.status_code == 201
    assert created.json()["provider_mode"] == "mock"
    assert turn.status_code == 200
    assert turn.json()["status"] == "completed"
    assert inactive_interrupt.json()["interrupted"] is False
    assert ended.json()["status"] == "ended"
