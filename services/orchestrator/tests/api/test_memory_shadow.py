from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.memory.models import MemoryAspect
from app.memory.retrieval import RetrievedMemory
from app.settings import Settings


class ShadowRetriever:
    def retrieve(
        self,
        query: str,
        *,
        user_id: str,
        purpose_scope: str = "personalization",
        k: int = 5,
        as_of_ms: int | None = None,
    ) -> list[RetrievedMemory]:
        _ = (query, user_id, purpose_scope, k, as_of_ms)
        return [
            RetrievedMemory(
                memory_id="memory_" + "a" * 32,
                text="returned only in process",
                aspect=MemoryAspect.FACT,
                score=0.8,
                relevance_score=0.8,
                reason_codes=["SEMANTIC_MATCH"],
                source_turn_id="source_turn",
                source_message_ids=[],
            )
        ]


async def create_session(
    client: AsyncClient,
    *,
    session_id: str,
) -> dict[str, str]:
    response = await client.post(
        "/api/sessions",
        json={"camera_consent": False, "client_session_id": session_id},
    )
    assert response.status_code == 201
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.mark.asyncio
async def test_disabled_deployment_cannot_collect_research_consent(
    tmp_path: Path,
) -> None:
    app = create_app(
        Settings(
            provider_mode="mock",
            event_database_path=tmp_path / "events.sqlite3",
            memory_shadow_enabled=False,
        )
    )
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        headers = await create_session(client, session_id="session_disabled")
        current = await client.get(
            "/api/sessions/session_disabled/memories/research-consent",
            headers=headers,
        )
        granted = await client.put(
            "/api/sessions/session_disabled/memories/research-consent",
            headers=headers,
            json={
                "granted": True,
                "acknowledged_policy_version": "memory-shadow-research-v1",
            },
        )

    assert current.status_code == 200
    assert current.json()["enabled"] is False
    assert current.json()["granted"] is False
    assert granted.status_code == 409


@pytest.mark.asyncio
async def test_authorized_turn_runs_shadow_and_revocation_clears_report(
    tmp_path: Path,
) -> None:
    policy_version = "test-policy-v1"
    app = create_app(
        Settings(
            provider_mode="mock",
            event_database_path=tmp_path / "events.sqlite3",
            memory_shadow_enabled=True,
            memory_shadow_policy_version=policy_version,
            memory_shadow_strategy_version="test-hybrid-v1",
        ),
        memory_shadow_retriever=ShadowRetriever(),
    )
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        headers = await create_session(client, session_id="session_shadow")
        stale_policy = await client.put(
            "/api/sessions/session_shadow/memories/research-consent",
            headers=headers,
            json={"granted": True, "acknowledged_policy_version": "old"},
        )
        granted = await client.put(
            "/api/sessions/session_shadow/memories/research-consent",
            headers=headers,
            json={
                "granted": True,
                "acknowledged_policy_version": policy_version,
            },
        )
        turn = await client.post(
            "/api/sessions/session_shadow/turns",
            headers=headers,
            json={"text": "我最近有点累", "visual_summary": ""},
        )
        await app.state.memory_shadow_runner.drain()
        report = await client.get(
            "/api/sessions/session_shadow/memories/shadow-report",
            headers=headers,
        )
        revoked = await client.put(
            "/api/sessions/session_shadow/memories/research-consent",
            headers=headers,
            json={
                "granted": False,
                "acknowledged_policy_version": policy_version,
            },
        )
        cleared = await client.get(
            "/api/sessions/session_shadow/memories/shadow-report",
            headers=headers,
        )

    assert stale_policy.status_code == 409
    assert granted.status_code == 200
    assert granted.json()["granted"] is True
    assert "query text" in granted.json()["excluded_fields"]
    assert turn.status_code == 200
    assert report.json()["aggregate"]["completed_count"] == 1
    assert report.json()["runtime"]["strategy_version"] == "test-hybrid-v1"
    assert revoked.json()["granted"] is False
    assert cleared.json()["consent_granted"] is False
    assert cleared.json()["aggregate"]["run_count"] == 0
    events = await app.state.event_store.list_session("session_shadow")
    consent_events = [event for event in events if "shadow_consent" in event.type]
    assert [event.type for event in consent_events] == [
        "memory.shadow_consent.granted",
        "memory.shadow_consent.revoked",
    ]
    assert all(event.payload == {"policy_version": policy_version} for event in consent_events)
