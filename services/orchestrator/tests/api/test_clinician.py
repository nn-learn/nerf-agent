import pytest
from httpx import ASGITransport, AsyncClient

from app.api.handoff import HandoffService
from app.main import create_app
from app.settings import Settings


async def _seed_session(
    client: AsyncClient,
    session_id: str,
) -> tuple[str, str]:
    created = await client.post(
        "/api/sessions",
        json={"client_session_id": session_id},
    )
    access_token = created.json()["access_token"]
    await client.post(
        f"/api/sessions/{session_id}/turns",
        headers={"Authorization": f"Bearer {access_token}"},
        json={"text": "最近持续低落，但我想先聊聊睡眠。"},
    )
    return session_id, access_token


def _clinician_headers(session_id: str) -> dict[str, str]:
    return {
        "X-Demo-Role": "CLINICIAN_DEMO",
        "X-Demo-Session": session_id,
    }


@pytest.mark.asyncio
async def test_clinician_summary_is_scoped_and_redacted(tmp_path) -> None:
    """Catches the handoff console exposing raw media or an unrestricted transcript."""
    app = create_app(
        Settings(
            provider_mode="mock",
            event_database_path=tmp_path / "events.sqlite3",
        )
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        session_id, _ = await _seed_session(client, "session_clinical_1")
        denied_user = await client.get(
            f"/api/clinician/sessions/{session_id}/summary",
            headers={
                "X-Demo-Role": "USER",
                "X-Demo-Session": session_id,
            },
        )
        denied_scope = await client.get(
            f"/api/clinician/sessions/{session_id}/summary",
            headers=_clinician_headers("session_other"),
        )
        response = await client.get(
            f"/api/clinician/sessions/{session_id}/summary",
            headers=_clinician_headers(session_id),
        )

    assert denied_user.status_code == 403
    assert denied_scope.status_code == 403
    assert response.status_code == 200
    body = response.json()
    assert set(body) >= {
        "session_id",
        "risk_level",
        "reasons",
        "key_turns",
        "handoff_state",
    }
    assert "full_transcript" not in body
    assert "raw_frame" not in response.text
    assert "pcm_s16le" not in response.text
    assert body["key_turns"][0]["transcript_excerpt"].endswith("…")


@pytest.mark.asyncio
async def test_clinician_timeline_contains_metadata_not_message_bodies(tmp_path) -> None:
    app = create_app(
        Settings(
            provider_mode="mock",
            event_database_path=tmp_path / "events.sqlite3",
        )
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        session_id, _ = await _seed_session(client, "session_clinical_1")
        response = await client.get(
            f"/api/clinician/sessions/{session_id}/timeline",
            headers=_clinician_headers(session_id),
        )

    assert response.status_code == 200
    timeline = response.json()["events"]
    transcript = next(
        event for event in timeline if event["type"] == "transcript.final"
    )
    assert transcript["metadata"] == {
        "input_mode": "typed_text",
        "content_redacted": True,
    }
    assert "最近持续低落" not in response.text


@pytest.mark.asyncio
async def test_clinician_accepts_one_scoped_handoff_event(tmp_path) -> None:
    app = create_app(
        Settings(
            provider_mode="mock",
            event_database_path=tmp_path / "events.sqlite3",
        )
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        session_id, _ = await _seed_session(client, "session_clinical_1")
        handoff = await HandoffService(app.state.event_store).request(
            session_id=session_id,
            risk_assessment_id="risk_demo_1",
            idempotency_key="handoff_session_clinical_1",
        )
        first = await client.post(
            f"/api/clinician/sessions/{session_id}/accept",
            headers=_clinician_headers(session_id),
            json={"handoff_id": handoff.handoff_id},
        )
        second = await client.post(
            f"/api/clinician/sessions/{session_id}/accept",
            headers=_clinician_headers(session_id),
            json={"handoff_id": handoff.handoff_id},
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["state"] == "ACCEPTED"
    events = await app.state.event_store.list_session(session_id)
    assert [event.type for event in events].count("handoff.accepted") == 1
