from fastapi import APIRouter
from pydantic import BaseModel

from app.events.consent import ConsentKind, ConsentService
from app.events.store import EventStore


class ConsentUpdate(BaseModel):
    granted: bool


class ConsentStatus(BaseModel):
    session_id: str
    kind: ConsentKind
    granted: bool


def create_consent_router(store: EventStore) -> APIRouter:
    router = APIRouter(prefix="/api/sessions", tags=["consent"])
    consent = ConsentService(store)

    @router.post(
        "/{session_id}/consents/{kind}",
        response_model=ConsentStatus,
    )
    async def update_consent(
        session_id: str,
        kind: ConsentKind,
        update: ConsentUpdate,
    ) -> ConsentStatus:
        await store.initialize()
        if update.granted:
            await consent.grant(session_id, kind)
        else:
            await consent.revoke(session_id, kind)
        return ConsentStatus(
            session_id=session_id,
            kind=kind,
            granted=update.granted,
        )

    return router
