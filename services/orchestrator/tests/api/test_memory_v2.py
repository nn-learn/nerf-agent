from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.memory.models import MemoryAspect, MemoryCandidate, MemoryKind
from app.settings import Settings


async def _create_session(
    client: AsyncClient,
    session_id: str,
) -> dict[str, str]:
    response = await client.post(
        "/api/sessions",
        json={"camera_consent": False, "client_session_id": session_id},
    )
    assert response.status_code == 201
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _candidate(text: str, turn_id: str) -> MemoryCandidate:
    return MemoryCandidate(
        source="transcript",
        contains_sensitive_content=False,
        text=text,
        kind=MemoryKind.SEMANTIC,
        source_turn_id=turn_id,
        aspect=MemoryAspect.PREFERENCE,
        subject_key="communication.response_style",
        confidence=0.9,
    )


@pytest.mark.asyncio
async def test_profile_api_hides_internal_identity_and_requires_owner_confirmation(
    tmp_path: Path,
) -> None:
    app = create_app(
        Settings(provider_mode="mock", event_database_path=tmp_path / "events.sqlite3")
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        owner_headers = await _create_session(client, "session_owner")
        stranger_headers = await _create_session(client, "session_stranger")
        user_id = app.state.memory_subjects.user_for_session("session_owner")
        item = app.state.memory_repository.add_active(
            user_id=user_id,
            candidate=_candidate("用户偏好：回答时简短", "turn_1"),
        )
        app.state.memory_profiles.record_observation(item, session_id="session_1")
        app.state.memory_profiles.record_observation(item, session_id="session_2")
        app.state.memory_consolidator.rebuild_user(user_id)

        listed = await client.get(
            "/api/sessions/session_owner/memory-profiles",
            headers=owner_headers,
        )
        profile_id = listed.json()[0]["profile_id"]
        stranger_decision = await client.post(
            f"/api/sessions/session_stranger/memory-profiles/{profile_id}/decision",
            headers=stranger_headers,
            json={"decision": "confirm"},
        )
        confirmed = await client.post(
            f"/api/sessions/session_owner/memory-profiles/{profile_id}/decision",
            headers=owner_headers,
            json={"decision": "confirm"},
        )

    assert listed.status_code == 200
    body = listed.json()[0]
    assert body["state"] == "AWAITING_CONFIRMATION"
    assert body["distinct_session_count"] == 2
    assert "user_id" not in body
    assert "signature_hash" not in body
    assert "evidence_digest" not in body
    assert "session_id" not in body["evidence"][0]
    assert "observation_id" not in body["evidence"][0]
    assert stranger_decision.status_code == 404
    assert confirmed.json()["state"] == "ACTIVE"


@pytest.mark.asyncio
async def test_conflict_api_resolves_once_and_delete_cascades_profiles(
    tmp_path: Path,
) -> None:
    app = create_app(
        Settings(provider_mode="mock", event_database_path=tmp_path / "events.sqlite3")
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        headers = await _create_session(client, "session_owner")
        user_id = app.state.memory_subjects.user_for_session("session_owner")
        concise = app.state.memory_repository.add_active(
            user_id=user_id,
            candidate=_candidate("用户偏好：回答时简短", "turn_1"),
        )
        app.state.memory_profiles.record_observation(concise, session_id="session_1")
        detailed = app.state.memory_repository.add_active(
            user_id=user_id,
            candidate=_candidate("用户偏好：请详细展开解释", "turn_2"),
        )
        app.state.memory_profiles.record_observation(detailed, session_id="session_2")
        app.state.memory_consolidator.rebuild_user(user_id)

        listed = await client.get(
            "/api/sessions/session_owner/memory-conflicts",
            headers=headers,
        )
        conflict = listed.json()[0]
        selected = next(
            option for option in conflict["options"] if "详细" in option["statement"]
        )
        resolved = await client.post(
            f"/api/sessions/session_owner/memory-conflicts/{conflict['conflict_id']}/decision",
            headers=headers,
            json={"decision": "select", "profile_id": selected["profile_id"]},
        )
        repeated = await client.post(
            f"/api/sessions/session_owner/memory-conflicts/{conflict['conflict_id']}/decision",
            headers=headers,
            json={"decision": "select", "profile_id": selected["profile_id"]},
        )
        deleted = await client.delete(
            f"/api/sessions/session_owner/memories/{detailed.memory_id}",
            headers=headers,
        )
        profiles_after = await client.get(
            "/api/sessions/session_owner/memory-profiles",
            headers=headers,
        )

    assert listed.status_code == 200
    assert "user_id" not in conflict
    assert resolved.json()["state"] == "RESOLVED"
    assert resolved.json()["selected_profile_id"] == selected["profile_id"]
    assert repeated.status_code == 409
    assert deleted.status_code == 204
    assert profiles_after.json() == []


@pytest.mark.asyncio
async def test_temporal_change_api_is_owner_gated_audited_and_cannot_be_bypassed(
    tmp_path: Path,
) -> None:
    app = create_app(
        Settings(provider_mode="mock", event_database_path=tmp_path / "events.sqlite3")
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        owner_headers = await _create_session(client, "session_owner")
        stranger_headers = await _create_session(client, "session_stranger")
        user_id = app.state.memory_subjects.user_for_session("session_owner")
        concise = app.state.memory_repository.add_active(
            user_id=user_id,
            candidate=_candidate("用户偏好：回答时简短", "turn_1").model_copy(
                update={"valid_from_ms": 1_000}
            ),
            now_ms=10_000,
        )
        app.state.memory_profiles.record_observation(
            concise,
            session_id="source_session_1",
            now_ms=10_000,
        )
        app.state.memory_profiles.record_observation(
            concise,
            session_id="source_session_2",
            now_ms=20_000,
        )
        proposed = app.state.memory_consolidator.rebuild_user(user_id, now_ms=30_000)
        old_profile = app.state.memory_profiles.confirm_profile(
            proposed[0].profile_id,
            user_id=user_id,
            now_ms=30_000,
        )
        transition = app.state.memory_repository.add_active(
            user_id=user_id,
            candidate=_candidate(
                "以前我喜欢简短回答，现在改成详细解释",
                "turn_3",
            ).model_copy(update={"valid_from_ms": 40_000}),
            now_ms=45_000,
        )
        app.state.memory_profiles.record_observation(
            transition,
            session_id="source_session_3",
            now_ms=45_000,
        )
        app.state.memory_consolidator.rebuild_user(user_id, now_ms=50_000)

        regular_profiles = await client.get(
            "/api/sessions/session_owner/memory-profiles",
            headers=owner_headers,
        )
        listed = await client.get(
            "/api/sessions/session_owner/memory-changes",
            headers=owner_headers,
        )
        body = listed.json()[0]
        proposed_id = body["proposed_profile"]["profile_id"]
        bypass = await client.post(
            f"/api/sessions/session_owner/memory-profiles/{proposed_id}/decision",
            headers=owner_headers,
            json={"decision": "confirm"},
        )
        stranger = await client.post(
            f"/api/sessions/session_stranger/memory-changes/{body['change_id']}/decision",
            headers=stranger_headers,
            json={"decision": "apply"},
        )
        applied = await client.post(
            f"/api/sessions/session_owner/memory-changes/{body['change_id']}/decision",
            headers=owner_headers,
            json={"decision": "apply"},
        )
        after = await client.get(
            "/api/sessions/session_owner/memory-profiles",
            headers=owner_headers,
        )
        audit_events = await app.state.event_store.list_session("session_owner")

    assert len(regular_profiles.json()) == 1
    assert regular_profiles.json()[0]["profile_id"] == old_profile.profile_id
    assert listed.status_code == 200
    assert body["state"] == "OPEN"
    assert body["effective_at_ms"] == 40_000
    assert body["observed_at_ms"] == 45_000
    assert "user_id" not in body
    assert "evidence_digest" not in body
    assert "session_id" not in body["proposed_profile"]["evidence"][0]
    assert bypass.status_code == 409
    assert stranger.status_code == 404
    assert applied.json()["state"] == "APPLIED"
    assert applied.json()["previous_profile"]["state"] == "HISTORICAL"
    assert applied.json()["previous_profile"]["valid_to_ms"] == 40_000
    assert applied.json()["proposed_profile"]["state"] == "ACTIVE"
    assert after.json()[0]["profile_id"] == proposed_id
    change_audit = [event for event in audit_events if event.type == "memory.change.applied"]
    assert len(change_audit) == 1
    assert change_audit[0].payload == {"change_id": body["change_id"]}
    assert transition.candidate.text not in str(change_audit[0].payload)
