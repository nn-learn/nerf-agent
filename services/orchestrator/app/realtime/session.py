import asyncio
import hashlib
import hmac
import secrets
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Protocol, cast
from uuid import uuid4

from pydantic import BaseModel, Field

from app.events.consent import ConsentKind, ConsentService
from app.events.store import EventStore
from app.graph.build import GraphDependencies, build_graph
from app.realtime.avatar_client import (
    AvatarClient,
    AvatarUnavailable,
    MockAvatarClient,
)
from app.realtime.models import PcmChunk, TurnHandle
from app.realtime.turn_coordinator import TurnCoordinator
from app.safety.models import AgentResponse, RiskAssessment, RiskLevel


class SessionNotFoundError(LookupError):
    pass


class SessionEndedError(RuntimeError):
    pass


class TurnInProgressError(RuntimeError):
    pass


class VisionUnavailable(RuntimeError):
    pass


class AudioUnavailable(RuntimeError):
    pass


class SessionStatus(StrEnum):
    ACTIVE = "active"
    ENDED = "ended"


class TurnStatus(StrEnum):
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"


class SessionDescriptor(BaseModel):
    session_id: str
    status: SessionStatus
    provider_mode: str
    camera_consent: bool
    fallback_capabilities: list[str] = Field(
        default_factory=lambda: [
            "voice_text",
            "typed_text",
            "http_text",
        ]
    )


class SessionCreated(SessionDescriptor):
    access_token: str = Field(min_length=32, repr=False)


class TurnResult(BaseModel):
    session_id: str
    turn_id: str
    trace_id: str
    status: TurnStatus
    risk_level: RiskLevel | None = None
    response: AgentResponse | None = None
    delivery_mode: Literal["voice_avatar", "voice_text", "text", "interrupted"]


class InterruptOutcome(BaseModel):
    session_id: str
    interrupted: bool
    turn_id: str | None = None
    cancelled_token: str | None = None
    reason: str


class VisionBridge(Protocol):
    async def observe(
        self,
        *,
        session_id: str,
        turn: TurnHandle,
        supplied_summary: str,
    ) -> str: ...


class AudioBridge(Protocol):
    async def synthesize(
        self,
        text: str,
        *,
        turn: TurnHandle,
    ) -> list[PcmChunk]: ...


class MockVisionBridge:
    def __init__(self, *, available: bool = True) -> None:
        self.available = available

    async def observe(
        self,
        *,
        session_id: str,
        turn: TurnHandle,
        supplied_summary: str,
    ) -> str:
        _ = (session_id, turn)
        if not self.available:
            raise VisionUnavailable("mock vision provider is unavailable")
        return supplied_summary.strip() or "本轮没有可可靠概括的画面信息。"


class MockAudioBridge:
    def __init__(self, *, available: bool = True) -> None:
        self.available = available

    async def synthesize(
        self,
        text: str,
        *,
        turn: TurnHandle,
    ) -> list[PcmChunk]:
        _ = (text, turn)
        if not self.available:
            raise AudioUnavailable("mock TTS provider is unavailable")
        return [
            PcmChunk(
                sequence=0,
                pts_ms=0,
                pcm_s16le=b"\x00\x00" * 320,
            )
        ]


@dataclass(slots=True)
class ActiveTurn:
    handle: TurnHandle
    trace_id: str


@dataclass(slots=True)
class SessionRuntime:
    descriptor: SessionDescriptor
    access_token_digest: str
    event_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    active_turn: ActiveTurn | None = None


class SessionManager:
    """Coordinates one auditable multimodal turn without persisting raw media."""

    def __init__(
        self,
        *,
        store: EventStore,
        avatar_client: AvatarClient | None = None,
        vision_bridge: VisionBridge | None = None,
        audio_bridge: AudioBridge | None = None,
        provider_mode: str = "mock",
    ) -> None:
        self.store = store
        self._avatar = avatar_client or MockAvatarClient()
        self._vision = vision_bridge or MockVisionBridge()
        self._audio = audio_bridge or MockAudioBridge()
        self._provider_mode = provider_mode
        self._coordinator = TurnCoordinator()
        self._graph = build_graph(GraphDependencies.for_mock())
        self._sessions: dict[str, SessionRuntime] = {}
        self._sessions_lock = asyncio.Lock()
        self._consent = ConsentService(store)

    async def create_session(
        self,
        *,
        camera_consent: bool,
        requested_id: str | None = None,
    ) -> SessionCreated:
        await self.store.initialize()
        session_id = requested_id or f"session_{uuid4().hex}"
        access_token = f"pst_{secrets.token_urlsafe(32)}"
        descriptor = SessionDescriptor(
            session_id=session_id,
            status=SessionStatus.ACTIVE,
            provider_mode=self._provider_mode,
            camera_consent=camera_consent,
        )
        runtime = SessionRuntime(
            descriptor=descriptor,
            access_token_digest=self._token_digest(access_token),
        )
        async with self._sessions_lock:
            if session_id in self._sessions:
                raise ValueError("session id already exists")
            self._sessions[session_id] = runtime
        await self.store.append_payload(
            session_id=session_id,
            event_type="session.started",
            payload={
                "provider_mode": self._provider_mode,
                "camera_consent": camera_consent,
            },
        )
        if camera_consent:
            await self._consent.grant(session_id, ConsentKind.CAMERA)
        await self._avatar.start_session(session_id)
        return SessionCreated(
            **descriptor.model_dump(),
            access_token=access_token,
        )

    async def has_access(self, session_id: str, access_token: str) -> bool:
        runtime = await self._runtime(session_id)
        supplied_digest = self._token_digest(access_token)
        return hmac.compare_digest(
            runtime.access_token_digest,
            supplied_digest,
        )

    async def get_session(self, session_id: str) -> SessionDescriptor:
        runtime = await self._runtime(session_id)
        async with runtime.event_lock:
            return runtime.descriptor.model_copy(deep=True)

    async def process_text_turn(
        self,
        session_id: str,
        *,
        text: str,
        visual_summary: str = "",
    ) -> TurnResult:
        transcript = text.strip()
        if not transcript:
            raise ValueError("turn text must not be blank")
        runtime = await self._runtime(session_id)
        active = await self._begin_turn(runtime)
        turn = active.handle
        trace_id = active.trace_id
        risk: RiskAssessment | None = None
        response: AgentResponse | None = None
        delivery_mode: Literal[
            "voice_avatar", "voice_text", "text", "interrupted"
        ] = "voice_avatar"

        try:
            if not await self._emit_current(
                runtime,
                active,
                "transcript.final",
                {"text": transcript, "input_mode": "typed_text"},
            ):
                return self._interrupted_result(active)

            effective_visual_summary = ""
            camera_consent = await self.store.is_consent_granted(
                session_id,
                ConsentKind.CAMERA.value,
            )
            if camera_consent:
                if not await self._emit_current(
                    runtime,
                    active,
                    "vision.burst.requested",
                    {"max_frames": 4, "retention": "in_memory_only"},
                ):
                    return self._interrupted_result(active)
                try:
                    effective_visual_summary = await self._vision.observe(
                        session_id=session_id,
                        turn=turn,
                        supplied_summary=visual_summary,
                    )
                except VisionUnavailable:
                    if not await self._emit_current(
                        runtime,
                        active,
                        "vision.degraded",
                        {"fallback": "voice_text_avatar"},
                    ):
                        return self._interrupted_result(active)
                else:
                    if not await self._emit_current(
                        runtime,
                        active,
                        "vision.observation.ready",
                        {
                            "summary": effective_visual_summary,
                            "valid_for_ms": 10_000,
                        },
                    ):
                        return self._interrupted_result(active)

            graph_result: dict[str, Any] = await self._graph.ainvoke(
                {
                    "transcript": transcript,
                    "visual_summary": effective_visual_summary,
                    "visited": [],
                }
            )
            risk = cast(RiskAssessment, graph_result["risk"])
            response = cast(AgentResponse, graph_result["response"])
            if not await self._emit_current(
                runtime,
                active,
                "risk.updated",
                risk.model_dump(mode="json"),
            ):
                return self._interrupted_result(active, risk=risk, response=response)
            if not await self._emit_current(
                runtime,
                active,
                "retrieval.completed",
                {
                    "evidence_ids": response.evidence_ids,
                    "reviewed_only": True,
                },
            ):
                return self._interrupted_result(active, risk=risk, response=response)
            if not await self._emit_current(
                runtime,
                active,
                "assistant.response.ready",
                response.model_dump(mode="json"),
            ):
                return self._interrupted_result(active, risk=risk, response=response)

            try:
                audio_chunks = await self._audio.synthesize(
                    response.spoken_text,
                    turn=turn,
                )
            except AudioUnavailable:
                delivery_mode = "text"
                if not await self._emit_current(
                    runtime,
                    active,
                    "tts.degraded",
                    {"fallback": "text"},
                ):
                    return self._interrupted_result(active, risk=risk, response=response)
                if not await self._emit_current(
                    runtime,
                    active,
                    "response.displayed",
                    {"text": response.display_text},
                ):
                    return self._interrupted_result(active, risk=risk, response=response)
            else:
                for chunk in audio_chunks:
                    if not await self._emit_current(
                        runtime,
                        active,
                        "tts.audio.chunk",
                        {
                            "sequence": chunk.sequence,
                            "pts_ms": chunk.pts_ms,
                            "duration_ms": chunk.duration_ms,
                        },
                    ):
                        return self._interrupted_result(
                            active,
                            risk=risk,
                            response=response,
                        )
                try:
                    avatar_result = await self._avatar.render(
                        turn,
                        style=response.avatar_style,
                        audio_chunk_count=len(audio_chunks),
                    )
                except AvatarUnavailable:
                    delivery_mode = "voice_text"
                    if not await self._emit_current(
                        runtime,
                        active,
                        "avatar.degraded",
                        {"fallback": "voice_text"},
                    ):
                        return self._interrupted_result(
                            active,
                            risk=risk,
                            response=response,
                        )
                else:
                    for frame_id in avatar_result.frame_ids:
                        if not await self._emit_current(
                            runtime,
                            active,
                            "avatar.frame.ready",
                            {"frame_id": frame_id, "track": "avatar-video"},
                        ):
                            return self._interrupted_result(
                                active,
                                risk=risk,
                                response=response,
                            )
                if not await self._emit_current(
                    runtime,
                    active,
                    "playback.started",
                    {"delivery_mode": delivery_mode},
                ):
                    return self._interrupted_result(
                        active,
                        risk=risk,
                        response=response,
                    )

            if not await self._emit_current(
                runtime,
                active,
                "turn.completed",
                {"delivery_mode": delivery_mode},
            ):
                return self._interrupted_result(active, risk=risk, response=response)
            return TurnResult(
                session_id=session_id,
                turn_id=turn.turn_id,
                trace_id=trace_id,
                status=TurnStatus.COMPLETED,
                risk_level=risk.level,
                response=response,
                delivery_mode=delivery_mode,
            )
        finally:
            await self._finish_if_active(runtime, active)

    async def interrupt(
        self,
        session_id: str,
        *,
        reason: str,
    ) -> InterruptOutcome:
        runtime = await self._runtime(session_id)
        interrupted_turn: TurnHandle | None = None
        cancelled_token: str | None = None
        async with runtime.event_lock:
            active = runtime.active_turn
            if active is None:
                return InterruptOutcome(
                    session_id=session_id,
                    interrupted=False,
                    reason=reason,
                )
            result = await self._coordinator.interrupt(
                session_id,
                reason=reason,
            )
            runtime.active_turn = None
            interrupted_turn = active.handle
            cancelled_token = result.cancelled_token
            await self.store.append_payload(
                session_id=session_id,
                turn_id=active.handle.turn_id,
                trace_id=active.trace_id,
                cancel_token=active.handle.cancel_token,
                event_type="playback.interrupted",
                payload={"reason": reason},
            )
        await self._avatar.cancel_turn(interrupted_turn)
        return InterruptOutcome(
            session_id=session_id,
            interrupted=True,
            turn_id=interrupted_turn.turn_id,
            cancelled_token=cancelled_token,
            reason=reason,
        )

    async def end_session(self, session_id: str) -> SessionDescriptor:
        runtime = await self._runtime(session_id)
        async with runtime.event_lock:
            has_active_turn = runtime.active_turn is not None
        if has_active_turn:
            await self.interrupt(session_id, reason="session_ended")
        async with runtime.event_lock:
            if runtime.descriptor.status is SessionStatus.ENDED:
                return runtime.descriptor.model_copy(deep=True)
            runtime.descriptor.status = SessionStatus.ENDED
            await self.store.append_payload(
                session_id=session_id,
                event_type="session.ended",
                payload={"reason": "user_ended"},
            )
        await self._avatar.end_session(session_id)
        return runtime.descriptor.model_copy(deep=True)

    async def _runtime(self, session_id: str) -> SessionRuntime:
        async with self._sessions_lock:
            runtime = self._sessions.get(session_id)
        if runtime is None:
            raise SessionNotFoundError(session_id)
        return runtime

    async def _begin_turn(self, runtime: SessionRuntime) -> ActiveTurn:
        async with runtime.event_lock:
            if runtime.descriptor.status is SessionStatus.ENDED:
                raise SessionEndedError(runtime.descriptor.session_id)
            if runtime.active_turn is not None:
                raise TurnInProgressError(runtime.descriptor.session_id)
            handle = await self._coordinator.start_turn(
                runtime.descriptor.session_id
            )
            active = ActiveTurn(
                handle=handle,
                trace_id=f"trace_{uuid4().hex}",
            )
            runtime.active_turn = active
            return active

    async def _emit_current(
        self,
        runtime: SessionRuntime,
        active: ActiveTurn,
        event_type: str,
        payload: dict[str, object],
    ) -> bool:
        async with runtime.event_lock:
            if runtime.active_turn is not active:
                return False
            if not await self._coordinator.tokens.is_current(
                active.handle.turn_id,
                active.handle.cancel_token,
            ):
                return False
            await self.store.append_payload(
                session_id=active.handle.session_id,
                turn_id=active.handle.turn_id,
                trace_id=active.trace_id,
                cancel_token=active.handle.cancel_token,
                event_type=event_type,
                payload=payload,
            )
            return True

    async def _finish_if_active(
        self,
        runtime: SessionRuntime,
        active: ActiveTurn,
    ) -> None:
        async with runtime.event_lock:
            if runtime.active_turn is not active:
                return
            runtime.active_turn = None
            await self._coordinator.complete_turn(active.handle)

    @staticmethod
    def _interrupted_result(
        active: ActiveTurn,
        *,
        risk: RiskAssessment | None = None,
        response: AgentResponse | None = None,
    ) -> TurnResult:
        return TurnResult(
            session_id=active.handle.session_id,
            turn_id=active.handle.turn_id,
            trace_id=active.trace_id,
            status=TurnStatus.INTERRUPTED,
            risk_level=risk.level if risk is not None else None,
            response=response,
            delivery_mode="interrupted",
        )

    @staticmethod
    def _token_digest(access_token: str) -> str:
        return hashlib.sha256(access_token.encode("utf-8")).hexdigest()
