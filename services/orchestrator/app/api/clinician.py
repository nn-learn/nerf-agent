from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Path
from pydantic import BaseModel, Field

from app.api.handoff import (
    HandoffNotFoundError,
    HandoffRecord,
    HandoffService,
)
from app.contracts.events import EventEnvelope
from app.events.store import EventStore

SessionId = Annotated[str, Path(min_length=1, max_length=96)]


class KeyTurn(BaseModel):
    turn_id: str
    transcript_excerpt: str
    risk_level: str | None = None
    support_mode: str | None = None


class ClinicianSummary(BaseModel):
    session_id: str
    risk_level: str
    reasons: list[str]
    key_turns: list[KeyTurn]
    visual_context_used: bool
    visual_summaries: list[str]
    handoff_state: str
    handoff_id: str | None = None
    safety_policy: str = (
        "AI support only: no diagnosis, prescribing, or autonomous emergency claims."
    )


class TimelineItem(BaseModel):
    seq: int
    type: str
    timestamp_ms: int
    turn_id: str
    trace_id: str
    metadata: dict[str, object]


class ClinicianTimeline(BaseModel):
    session_id: str
    events: list[TimelineItem]


class AcceptHandoffRequest(BaseModel):
    handoff_id: str = Field(min_length=1, max_length=96)


def _require_clinician_scope(
    session_id: str,
    role: str | None,
    scoped_session: str | None,
) -> None:
    if role != "CLINICIAN_DEMO":
        raise HTTPException(
            status_code=403,
            detail="CLINICIAN_DEMO role required",
        )
    if scoped_session != session_id:
        raise HTTPException(
            status_code=403,
            detail="clinician session scope denied",
        )


def _excerpt(text: object, limit: int = 16) -> str:
    normalized = str(text).strip()
    if not normalized:
        return "（内容已隐藏）"
    return f"{normalized[:limit]}…"


def _safe_metadata(
    event: EventEnvelope[dict[str, object]],
) -> dict[str, object]:
    payload = event.payload
    if event.type == "transcript.final":
        return {
            "input_mode": payload.get("input_mode", "unknown"),
            "content_redacted": True,
        }
    if event.type == "assistant.response.ready":
        return {
            "support_mode": payload.get("support_mode"),
            "risk_level": payload.get("risk_level"),
            "content_redacted": True,
        }
    if event.type == "vision.observation.ready":
        return {
            "scene_summary": _excerpt(payload.get("summary", ""), 60),
            "raw_frames_available": False,
        }
    allowed_keys = {
        "risk.updated": ("level", "reasons", "confidence"),
        "handoff.requested": ("handoff_id", "state", "simulated"),
        "handoff.accepted": ("handoff_id", "state", "simulated", "accepted_by"),
        "playback.interrupted": ("reason",),
        "playback.started": ("delivery_mode",),
        "tts.audio.chunk": ("sequence", "pts_ms", "duration_ms"),
        "avatar.frame.ready": ("track",),
        "session.started": ("provider_mode", "camera_consent"),
        "session.ended": ("reason",),
    }
    keys = allowed_keys.get(event.type, ())
    return {key: payload[key] for key in keys if key in payload}


def _build_summary(
    session_id: str,
    events: list[EventEnvelope[dict[str, object]]],
) -> ClinicianSummary:
    latest_risk = next(
        (event for event in reversed(events) if event.type == "risk.updated"),
        None,
    )
    risk_level = (
        str(latest_risk.payload.get("level", "UNKNOWN"))
        if latest_risk is not None
        else "UNKNOWN"
    )
    raw_reasons = latest_risk.payload.get("reasons", []) if latest_risk else []
    reasons = [str(reason) for reason in raw_reasons] if isinstance(raw_reasons, list) else []

    responses = {
        event.turn_id: event
        for event in events
        if event.type == "assistant.response.ready"
    }
    risk_by_turn = {
        event.turn_id: str(event.payload.get("level", "UNKNOWN"))
        for event in events
        if event.type == "risk.updated"
    }
    key_turns = [
        KeyTurn(
            turn_id=event.turn_id,
            transcript_excerpt=_excerpt(event.payload.get("text", "")),
            risk_level=risk_by_turn.get(event.turn_id),
            support_mode=(
                str(responses[event.turn_id].payload.get("support_mode"))
                if event.turn_id in responses
                else None
            ),
        )
        for event in events
        if event.type == "transcript.final"
    ][-5:]
    visual_summaries = [
        _excerpt(event.payload.get("summary", ""), 60)
        for event in events
        if event.type == "vision.observation.ready"
    ][-3:]
    handoff_event = next(
        (
            event
            for event in reversed(events)
            if event.type in {"handoff.requested", "handoff.accepted"}
        ),
        None,
    )
    return ClinicianSummary(
        session_id=session_id,
        risk_level=risk_level,
        reasons=reasons,
        key_turns=key_turns,
        visual_context_used=bool(visual_summaries),
        visual_summaries=visual_summaries,
        handoff_state=(
            str(handoff_event.payload.get("state", "NONE"))
            if handoff_event is not None
            else "NONE"
        ),
        handoff_id=(
            str(handoff_event.payload.get("handoff_id"))
            if handoff_event is not None
            else None
        ),
    )


def create_clinician_router(store: EventStore) -> APIRouter:
    router = APIRouter(prefix="/api/clinician", tags=["clinician"])
    handoffs = HandoffService(store)

    @router.get(
        "/sessions/{session_id}/summary",
        response_model=ClinicianSummary,
    )
    async def summary(
        session_id: SessionId,
        role: str | None = Header(default=None, alias="X-Demo-Role"),
        scoped_session: str | None = Header(
            default=None,
            alias="X-Demo-Session",
        ),
    ) -> ClinicianSummary:
        _require_clinician_scope(session_id, role, scoped_session)
        events = await store.list_session(session_id)
        if not events:
            raise HTTPException(status_code=404, detail="session not found")
        return _build_summary(session_id, events)

    @router.get(
        "/sessions/{session_id}/timeline",
        response_model=ClinicianTimeline,
    )
    async def timeline(
        session_id: SessionId,
        role: str | None = Header(default=None, alias="X-Demo-Role"),
        scoped_session: str | None = Header(
            default=None,
            alias="X-Demo-Session",
        ),
    ) -> ClinicianTimeline:
        _require_clinician_scope(session_id, role, scoped_session)
        events = await store.list_session(session_id)
        if not events:
            raise HTTPException(status_code=404, detail="session not found")
        return ClinicianTimeline(
            session_id=session_id,
            events=[
                TimelineItem(
                    seq=event.seq,
                    type=event.type,
                    timestamp_ms=event.timestamp_ms,
                    turn_id=event.turn_id,
                    trace_id=event.trace_id,
                    metadata=_safe_metadata(event),
                )
                for event in events
            ],
        )

    @router.post(
        "/sessions/{session_id}/accept",
        response_model=HandoffRecord,
    )
    async def accept(
        session_id: SessionId,
        request: AcceptHandoffRequest,
        role: str | None = Header(default=None, alias="X-Demo-Role"),
        scoped_session: str | None = Header(
            default=None,
            alias="X-Demo-Session",
        ),
    ) -> HandoffRecord:
        _require_clinician_scope(session_id, role, scoped_session)
        record = handoffs.get(request.handoff_id)
        if record is None:
            raise HTTPException(status_code=404, detail="handoff not found")
        if record.session_id != session_id:
            raise HTTPException(status_code=403, detail="handoff scope denied")
        try:
            return await handoffs.accept(
                request.handoff_id,
                actor_role="CLINICIAN_DEMO",
            )
        except HandoffNotFoundError as error:
            raise HTTPException(status_code=404, detail="handoff not found") from error

    return router
