from collections.abc import Callable
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
BindMemorySubject = Callable[[str, str | None], str | None]
ValidateMemorySubject = Callable[[str], None]


class CreateSessionRequest(BaseModel):
    camera_consent: bool = False
    client_session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=96,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    memory_subject_token: str | None = Field(
        default=None,
        min_length=40,
        max_length=128,
        pattern=r"^pms_[A-Za-z0-9_-]+$",
    )


class TextTurnRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8_000)
    visual_summary: str = Field(default="", max_length=2_000)


class InterruptRequest(BaseModel):
    reason: str = Field(default="user_speech", min_length=1, max_length=64)


def create_sessions_router(
    manager: SessionManager,
    *,
    bind_memory_subject: BindMemorySubject | None = None,
    validate_memory_subject: ValidateMemorySubject | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/api/sessions", tags=["sessions"])

    @router.post(
        "",
        response_model=SessionCreated,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_session(request: CreateSessionRequest) -> SessionCreated:
        try:
            if (
                request.memory_subject_token is not None
                and validate_memory_subject is not None
            ):
                validate_memory_subject(request.memory_subject_token)
            session = await manager.create_session(
                camera_consent=request.camera_consent,
                requested_id=request.client_session_id,
            )
            if bind_memory_subject is None:
                return session
            issued_token = bind_memory_subject(
                session.session_id,
                request.memory_subject_token,
            )
            return session.model_copy(
                update={"memory_subject_token": issued_token}
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
                synthesize_audio=False,
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
