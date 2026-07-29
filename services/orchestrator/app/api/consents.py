from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from app.events.consent import ConsentKind, ConsentService
from app.events.store import EventStore
from app.realtime.session import SessionManager, SessionNotFoundError
from app.security.auth import require_session_access


class ConsentUpdate(BaseModel):
    granted: bool


class ConsentStatus(BaseModel):
    session_id: str
    kind: ConsentKind
    granted: bool


def create_consent_router(
    store: EventStore,
    session_manager: SessionManager,
) -> APIRouter:
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
        authorization: str | None = Header(default=None),
    ) -> ConsentStatus:
        await store.initialize()
        try:
            await require_session_access(
                session_manager,
                session_id=session_id,
                authorization=authorization,
            )
        except SessionNotFoundError as error:
            raise HTTPException(status_code=404, detail="session not found") from error
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
