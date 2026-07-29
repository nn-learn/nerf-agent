from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Path, status
from pydantic import BaseModel, Field

from app.contracts.events import EventEnvelope
from app.realtime.session import (
    InterruptOutcome,
    SessionCreated,
    SessionDescriptor,
    SessionEndedError,
    SessionManager,
    SessionNotFoundError,
    TurnInProgressError,
    TurnResult,
)
from app.security.auth import require_session_access

SessionId = Annotated[str, Path(min_length=1, max_length=96)]


class CreateSessionRequest(BaseModel):
    camera_consent: bool = False
    client_session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=96,
        pattern=r"^[A-Za-z0-9_-]+$",
    )


class TextTurnRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8_000)
    visual_summary: str = Field(default="", max_length=2_000)


class InterruptRequest(BaseModel):
    reason: str = Field(default="user_speech", min_length=1, max_length=64)


def create_sessions_router(manager: SessionManager) -> APIRouter:
    router = APIRouter(prefix="/api/sessions", tags=["sessions"])

    @router.post(
        "",
        response_model=SessionCreated,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_session(request: CreateSessionRequest) -> SessionCreated:
        try:
            return await manager.create_session(
                camera_consent=request.camera_consent,
                requested_id=request.client_session_id,
            )
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @router.get("/{session_id}", response_model=SessionDescriptor)
    async def get_session(
        session_id: SessionId,
        authorization: str | None = Header(default=None),
    ) -> SessionDescriptor:
        try:
            await require_session_access(
                manager,
                session_id=session_id,
                authorization=authorization,
            )
            return await manager.get_session(session_id)
        except SessionNotFoundError as error:
            raise HTTPException(status_code=404, detail="session not found") from error

    @router.post("/{session_id}/turns", response_model=TurnResult)
    async def create_text_turn(
        session_id: SessionId,
        request: TextTurnRequest,
        authorization: str | None = Header(default=None),
    ) -> TurnResult:
        try:
            await require_session_access(
                manager,
                session_id=session_id,
                authorization=authorization,
            )
            return await manager.process_text_turn(
                session_id,
                text=request.text,
                visual_summary=request.visual_summary,
            )
        except SessionNotFoundError as error:
            raise HTTPException(status_code=404, detail="session not found") from error
        except (SessionEndedError, TurnInProgressError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @router.post(
        "/{session_id}/interrupt",
        response_model=InterruptOutcome,
    )
    async def interrupt(
        session_id: SessionId,
        request: InterruptRequest,
        authorization: str | None = Header(default=None),
    ) -> InterruptOutcome:
        try:
            await require_session_access(
                manager,
                session_id=session_id,
                authorization=authorization,
            )
            return await manager.interrupt(session_id, reason=request.reason)
        except SessionNotFoundError as error:
            raise HTTPException(status_code=404, detail="session not found") from error

    @router.get(
        "/{session_id}/events",
        response_model=list[EventEnvelope[dict[str, object]]],
    )
    async def list_events(
        session_id: SessionId,
        authorization: str | None = Header(default=None),
    ) -> list[EventEnvelope[dict[str, object]]]:
        try:
            await require_session_access(
                manager,
                session_id=session_id,
                authorization=authorization,
            )
            await manager.get_session(session_id)
        except SessionNotFoundError as error:
            raise HTTPException(status_code=404, detail="session not found") from error
        return await manager.store.list_session(session_id)

    @router.delete("/{session_id}", response_model=SessionDescriptor)
    async def end_session(
        session_id: SessionId,
        authorization: str | None = Header(default=None),
    ) -> SessionDescriptor:
        try:
            await require_session_access(
                manager,
                session_id=session_id,
                authorization=authorization,
            )
            return await manager.end_session(session_id)
        except SessionNotFoundError as error:
            raise HTTPException(status_code=404, detail="session not found") from error

    return router
