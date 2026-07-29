import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.settings import Settings


async def _create_session(
    client: AsyncClient,
    session_id: str,
) -> tuple[str, str]:
    response = await client.post(
        "/api/sessions",
        json={
            "camera_consent": False,
            "client_session_id": session_id,
        },
    )
    assert response.status_code == 201
    return response.json()["session_id"], response.json()["access_token"]


@pytest.mark.asyncio
async def test_session_routes_require_the_matching_bearer_token(tmp_path) -> None:
    """Catches one browser controlling another user's support session."""
    app = create_app(
        Settings(
            provider_mode="mock",
            event_database_path=tmp_path / "events.sqlite3",
        )
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        session_id, access_token = await _create_session(client, "session_owner")
        _, other_token = await _create_session(client, "session_other")
        missing = await client.post(
            f"/api/sessions/{session_id}/interrupt",
            json={"reason": "user_speech"},
        )
        wrong = await client.post(
            f"/api/sessions/{session_id}/interrupt",
            headers={"Authorization": f"Bearer {other_token}"},
            json={"reason": "user_speech"},
        )
        allowed = await client.post(
            f"/api/sessions/{session_id}/interrupt",
            headers={"Authorization": f"Bearer {access_token}"},
            json={"reason": "user_speech"},
        )

    assert missing.status_code == 401
    assert wrong.status_code == 403
    assert allowed.status_code == 200


@pytest.mark.asyncio
async def test_livekit_token_cannot_be_minted_for_an_unowned_session(
    tmp_path,
) -> None:
    """Catches room token minting based only on a guessed session id."""
    settings = Settings(
        provider_mode="mock",
        event_database_path=tmp_path / "events.sqlite3",
        livekit_api_key="devkey",
        livekit_api_secret="secret-with-at-least-32-characters",
        livekit_url="ws://localhost:7880",
    )
    transport = ASGITransport(app=create_app(settings))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        session_id, _ = await _create_session(client, "session_owner")
        response = await client.post(
            "/api/livekit/token",
            json={
                "session_id": session_id,
                "participant_name": "演示用户",
            },
        )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_api_responses_include_browser_security_headers(tmp_path) -> None:
    app = create_app(
        Settings(
            provider_mode="mock",
            event_database_path=tmp_path / "events.sqlite3",
        )
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health/live")

    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert "camera=(self)" in response.headers["permissions-policy"]


@pytest.mark.asyncio
async def test_mutating_requests_are_rate_limited_per_client(tmp_path) -> None:
    settings = Settings(
        provider_mode="mock",
        event_database_path=tmp_path / "events.sqlite3",
        request_rate_limit_per_minute=2,
    )
    transport = ASGITransport(app=create_app(settings))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        responses = [
            await client.post(
                "/api/sessions",
                json={"client_session_id": f"session_{index}"},
            )
            for index in range(3)
        ]

    assert [response.status_code for response in responses] == [201, 201, 429]
    assert responses[-1].headers["retry-after"] == "60"
