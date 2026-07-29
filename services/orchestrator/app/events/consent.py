import asyncio
from collections.abc import Awaitable, Callable
from enum import StrEnum

from app.events.store import EventStore


class ConsentKind(StrEnum):
    MICROPHONE = "microphone"
    CAMERA = "camera"
    MEMORY = "memory"


RevocationCallback = Callable[[str, ConsentKind], Awaitable[None]]


class ConsentService:
    def __init__(self, store: EventStore) -> None:
        self._store = store
        self._callbacks: list[RevocationCallback] = []

    def register_revocation_callback(self, callback: RevocationCallback) -> None:
        self._callbacks.append(callback)

    async def grant(self, session_id: str, kind: ConsentKind) -> None:
        await self._store.set_consent(session_id, kind.value, True)
        event_type = (
            "camera_consent_granted"
            if kind is ConsentKind.CAMERA
            else "consent.granted"
        )
        await self._store.append_payload(
            session_id=session_id,
            event_type=event_type,
            payload={"kind": kind.value},
        )

    async def revoke(self, session_id: str, kind: ConsentKind) -> None:
        await self._store.set_consent(session_id, kind.value, False)
        event_type = (
            "camera_consent_revoked"
            if kind is ConsentKind.CAMERA
            else "consent.revoked"
        )
        await self._store.append_payload(
            session_id=session_id,
            event_type=event_type,
            payload={"kind": kind.value},
        )
        await asyncio.gather(
            *(callback(session_id, kind) for callback in self._callbacks)
        )

    async def is_granted(self, session_id: str, kind: ConsentKind) -> bool:
        return await self._store.is_consent_granted(session_id, kind.value)
