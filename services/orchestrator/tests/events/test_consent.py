import pytest

from app.events.consent import ConsentKind, ConsentService
from app.events.store import EventStore


@pytest.mark.asyncio
async def test_camera_consent_is_scoped_to_one_session(tmp_path) -> None:
    """Catches consent leaking from one session into the next."""
    store = EventStore(tmp_path / "events.sqlite3")
    await store.initialize()
    consent = ConsentService(store)

    await consent.grant("session_1", ConsentKind.CAMERA)

    assert await consent.is_granted("session_1", ConsentKind.CAMERA)
    assert not await consent.is_granted("session_2", ConsentKind.CAMERA)


@pytest.mark.asyncio
async def test_revoking_camera_consent_runs_cleanup_and_audits_transition(tmp_path) -> None:
    """Catches revoked camera sessions that keep queued frames or lack an audit event."""
    store = EventStore(tmp_path / "events.sqlite3")
    await store.initialize()
    consent = ConsentService(store)
    cleaned: list[tuple[str, ConsentKind]] = []

    async def clear_camera_buffer(session_id: str, kind: ConsentKind) -> None:
        cleaned.append((session_id, kind))

    consent.register_revocation_callback(clear_camera_buffer)
    await consent.grant("session_1", ConsentKind.CAMERA)
    await consent.revoke("session_1", ConsentKind.CAMERA)

    assert cleaned == [("session_1", ConsentKind.CAMERA)]
    assert not await consent.is_granted("session_1", ConsentKind.CAMERA)
    events = await store.list_session("session_1")
    assert [event.type for event in events] == [
        "consent.granted",
        "camera_consent_revoked",
    ]
