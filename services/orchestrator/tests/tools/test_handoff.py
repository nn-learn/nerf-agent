import pytest

from app.api.handoff import HandoffPermissionError, HandoffService, HandoffState
from app.events.store import EventStore


@pytest.mark.asyncio
async def test_demo_clinician_accepts_handoff_once_with_audit_events(tmp_path) -> None:
    """Catches duplicate acceptance and non-auditable demo handoff transitions."""
    event_store = EventStore(tmp_path / "events.sqlite3")
    await event_store.initialize()
    service = HandoffService(event_store)

    requested = await service.request(
        session_id="session_1",
        risk_assessment_id="risk_1",
        idempotency_key="handoff_session_1_risk_1",
    )
    accepted = await service.accept(
        requested.handoff_id,
        actor_role="CLINICIAN_DEMO",
    )
    accepted_again = await service.accept(
        requested.handoff_id,
        actor_role="CLINICIAN_DEMO",
    )

    assert accepted.state is HandoffState.ACCEPTED
    assert accepted_again == accepted
    assert accepted.accepted_by == "CLINICIAN_DEMO"
    events = await event_store.list_session("session_1")
    assert [event.type for event in events] == [
        "handoff.requested",
        "handoff.accepted",
    ]


@pytest.mark.asyncio
async def test_user_cannot_accept_clinician_handoff(tmp_path) -> None:
    """Catches the patient-side demo UI bypassing the clinician role boundary."""
    event_store = EventStore(tmp_path / "events.sqlite3")
    await event_store.initialize()
    service = HandoffService(event_store)
    requested = await service.request(
        session_id="session_1",
        risk_assessment_id="risk_1",
        idempotency_key="handoff_session_1_risk_1",
    )

    with pytest.raises(HandoffPermissionError, match="CLINICIAN_DEMO"):
        await service.accept(requested.handoff_id, actor_role="USER")
