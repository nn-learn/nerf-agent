import base64
import json

import pytest
from httpx import ASGITransport, AsyncClient

from app.events.store import EventStore
from app.main import create_app
from app.settings import Settings


def _decode_claims(token: str) -> dict[str, object]:
    payload = token.split(".")[1]
    padding = "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload + padding))


@pytest.mark.asyncio
async def test_user_token_is_short_lived_and_scoped_to_one_session_room() -> None:
    """Catches broad publish grants and client-controlled agent identities."""
    settings = Settings(
        provider_mode="mock",
        livekit_api_key="devkey",
        livekit_api_secret="secret-with-at-least-32-characters",
        livekit_url="ws://localhost:7880",
    )
    assert "secret-with-at-least-32-characters" not in repr(settings)
    transport = ASGITransport(app=create_app(settings))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post(
            "/api/sessions",
            json={
                "client_session_id": "session_42",
                "camera_consent": False,
            },
        )
        access_token = created.json()["access_token"]
        response = await client.post(
            "/api/livekit/token",
            headers={"Authorization": f"Bearer {access_token}"},
            json={
                "session_id": "session_42",
                "participant_name": "演示用户",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["server_url"] == "ws://localhost:7880"
    assert body["room_name"] == "psyavatar_session_42"
    claims = _decode_claims(body["participant_token"])
    video = claims["video"]
    assert isinstance(video, dict)
    assert claims["sub"].startswith("user_")
    assert claims["sub"] != "avatar"
    assert video == {
        "roomJoin": True,
        "room": "psyavatar_session_42",
        "canPublish": True,
        "canPublishData": True,
        "canPublishSources": ["camera", "microphone"],
        "canSubscribe": True,
    }
    assert 0 < claims["exp"] - claims["nbf"] <= 600


@pytest.mark.asyncio
async def test_token_endpoint_rejects_unconfigured_livekit(tmp_path) -> None:
    """Catches accidental token signing with empty or fallback production secrets."""
    transport = ASGITransport(
        app=create_app(
            Settings(
                provider_mode="mock",
                event_database_path=tmp_path / "events.sqlite3",
            )
        )
    )
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post(
            "/api/sessions",
            json={"client_session_id": "session_42"},
        )
        response = await client.post(
            "/api/livekit/token",
            headers={
                "Authorization": f"Bearer {created.json()['access_token']}"
            },
            json={"session_id": "session_42", "participant_name": "演示用户"},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "LiveKit is not configured"


@pytest.mark.asyncio
async def test_camera_consent_endpoint_persists_only_audited_transitions(
    tmp_path,
) -> None:
    """Catches browser-only consent flags that the server cannot enforce or audit."""
    database_path = tmp_path / "events.sqlite3"
    settings = Settings(
        provider_mode="mock",
        event_database_path=database_path,
    )
    transport = ASGITransport(app=create_app(settings))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post(
            "/api/sessions",
            json={
                "client_session_id": "session_42",
                "camera_consent": False,
            },
        )
        headers = {
            "Authorization": f"Bearer {created.json()['access_token']}"
        }
        granted = await client.post(
            "/api/sessions/session_42/consents/camera",
            headers=headers,
            json={"granted": True},
        )
        revoked = await client.post(
            "/api/sessions/session_42/consents/camera",
            headers=headers,
            json={"granted": False},
        )

    assert granted.status_code == 200
    assert granted.json() == {
        "session_id": "session_42",
        "kind": "camera",
        "granted": True,
    }
    assert revoked.status_code == 200
    store = EventStore(database_path)
    events = await store.list_session("session_42")
    assert [event.type for event in events][1:] == [
        "camera_consent_granted",
        "camera_consent_revoked",
    ]
