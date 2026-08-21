from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.memory.models import (
    MemoryAllowedUse,
    MemoryCandidate,
    MemoryKind,
    MemorySensitivity,
    MemorySourceType,
)
from app.settings import Settings


async def _session(client: AsyncClient, session_id: str) -> dict[str, str]:
    response = await client.post(
        "/api/sessions",
        json={"camera_consent": False, "client_session_id": session_id},
    )
    assert response.status_code == 201
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _candidate(
    text: str,
    *,
    derived_from_memory_ids: list[str] | None = None,
) -> MemoryCandidate:
    return MemoryCandidate(
        source="transcript",
        contains_sensitive_content=True,
        text=text,
        kind=MemoryKind.SEMANTIC,
        source_turn_id="turn_1",
        user_confirmed=True,
        source_type=MemorySourceType.USER_STATEMENT,
        sensitivity=MemorySensitivity.HEALTH_SENSITIVE,
        allowed_uses=[MemoryAllowedUse.RESPONSE_CONTEXT],
        observed_at_ms=1_000,
        valid_from_ms=900,
        derived_from_memory_ids=derived_from_memory_ids or [],
    )


@pytest.mark.asyncio
async def test_user_can_pause_resume_scope_and_export_memory(tmp_path: Path) -> None:
    app = create_app(
        Settings(provider_mode="mock", event_database_path=tmp_path / "events.sqlite3")
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        headers = await _session(client, "session_owner")
        user_id = app.state.memory_subjects.user_for_session("session_owner")
        item = app.state.memory_repository.add_active(
            user_id=user_id,
            candidate=_candidate("用户最近睡眠不好"),
        )

        listed = await client.get(
            "/api/sessions/session_owner/memories",
            headers=headers,
        )
        scoped = await client.put(
            f"/api/sessions/session_owner/memories/{item.memory_id}/uses",
            headers=headers,
            json={"allowed_uses": ["PERSONALIZATION"]},
        )
        paused = await client.post(
            f"/api/sessions/session_owner/memories/{item.memory_id}/control",
            headers=headers,
            json={"action": "pause"},
        )
        exported = await client.get(
            "/api/sessions/session_owner/memories/export",
            headers=headers,
        )
        resumed = await client.post(
            f"/api/sessions/session_owner/memories/{item.memory_id}/control",
            headers=headers,
            json={"action": "resume"},
        )

    body = listed.json()[0]
    assert body["source_type"] == "USER_STATEMENT"
    assert body["sensitivity"] == "HEALTH_SENSITIVE"
    assert body["observed_at_ms"] == 1_000
    assert body["valid_from_ms"] == 900
    assert scoped.json()["allowed_uses"] == ["PERSONALIZATION"]
    assert paused.json()["state"] == "PAUSED"
    export_body = exported.json()
    assert export_body["schema_version"] == "memory-user-export-1.0"
    assert export_body["memories"][0]["state"] == "PAUSED"
    assert "user_id" not in export_body["memories"][0]
    assert resumed.json()["state"] == "ACTIVE"


@pytest.mark.asyncio
async def test_forget_returns_receipt_and_cascades_without_content(tmp_path: Path) -> None:
    app = create_app(
        Settings(provider_mode="mock", event_database_path=tmp_path / "events.sqlite3")
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        headers = await _session(client, "session_owner")
        user_id = app.state.memory_subjects.user_for_session("session_owner")
        root = app.state.memory_repository.add_active(
            user_id=user_id,
            candidate=_candidate("private root memory"),
        )
        app.state.memory_repository.add_active(
            user_id=user_id,
            candidate=_candidate(
                "private derived memory",
                derived_from_memory_ids=[root.memory_id],
            ),
        )

        forgotten = await client.post(
            f"/api/sessions/session_owner/memories/{root.memory_id}/forget",
            headers=headers,
        )
        listed = await client.get(
            "/api/sessions/session_owner/memories",
            headers=headers,
        )

    assert forgotten.status_code == 200
    receipt = forgotten.json()
    assert receipt["deleted_memory_count"] == 2
    assert receipt["deleted_derived_count"] == 1
    assert root.memory_id not in str(receipt)
    assert "private" not in str(receipt)
    assert listed.json() == []
