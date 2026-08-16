import asyncio
import hashlib
import hmac
import secrets
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Protocol, cast
from uuid import uuid4

from pydantic import BaseModel, Field

from app.agent.avatar import AvatarResponsePlan
from app.agent.care import CareLoopState
from app.agent.interventions import InterventionRuntimeState
from app.events.consent import ConsentKind, ConsentService
from app.events.store import EventStore
from app.graph.build import GraphDependencies, MemoryContextLoader, build_graph
from app.providers.faster_whisper import FasterWhisperProvider
from app.providers.mock import MockAgentProvider
from app.providers.protocols import AgentProvider
from app.realtime.avatar_client import (
    AvatarClient,
    AvatarUnavailable,
    MockAvatarClient,
)
from app.realtime.models import PcmChunk, TranscriptKind, TurnHandle
from app.realtime.observer import NullTurnObserver, TurnObserver
from app.realtime.turn_coordinator import TurnCoordinator
from app.safety.models import AgentResponse, RiskAssessment, RiskLevel

SessionEndCallback = Callable[[str], Awaitable[None]]


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
    memory_subject_token: str | None = Field(default=None, min_length=40, repr=False)


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
    def synthesize(
        self,
        text: str,
        *,
        turn: TurnHandle,
        speech_rate: float = 1.0,
    ) -> AsyncIterator[PcmChunk]: ...


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
        speech_rate: float = 1.0,
    ) -> AsyncIterator[PcmChunk]:
        _ = (text, turn, speech_rate)
        if not self.available:
            raise AudioUnavailable("mock TTS provider is unavailable")
        yield PcmChunk(
            sequence=0,
            pts_ms=0,
            pcm_s16le=b"\x00\x00" * 320,
        )


@dataclass(slots=True)
class ActiveTurn:
    handle: TurnHandle
    trace_id: str
    synthesize_audio: bool = True


@dataclass(slots=True)
class SessionRuntime:
    descriptor: SessionDescriptor
    access_token_digest: str
    event_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    active_turn: ActiveTurn | None = None
    care_loop_state: CareLoopState = field(default_factory=CareLoopState)
    intervention_state: InterventionRuntimeState = field(
        default_factory=InterventionRuntimeState
    )


class SessionManager:
    """Coordinates one auditable multimodal turn without persisting raw media."""

    def __init__(
        self,
        *,
        store: EventStore,
        coordinator: TurnCoordinator | None = None,
        agent_provider: AgentProvider | None = None,
        avatar_client: AvatarClient | None = None,
        vision_bridge: VisionBridge | None = None,
        audio_bridge: AudioBridge | None = None,
        provider_mode: str = "mock",
        session_end_callback: SessionEndCallback | None = None,
        memory_context_loader: MemoryContextLoader | None = None,
    ) -> None:
        self.store = store
        self._avatar = avatar_client or MockAvatarClient()
        self._vision = vision_bridge or MockVisionBridge()
        self._audio = audio_bridge or MockAudioBridge()
        self._provider_mode = provider_mode
        self._coordinator = coordinator or TurnCoordinator()
        self._graph = build_graph(
            GraphDependencies(
                agent_provider=agent_provider or MockAgentProvider(),
                output_guard=GraphDependencies.for_mock().output_guard,
                memory_context_loader=memory_context_loader,
            )
        )
        self._sessions: dict[str, SessionRuntime] = {}
        self._sessions_lock = asyncio.Lock()
        self._consent = ConsentService(store)
        self._session_end_callback = session_end_callback

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
            if not await self.store.reserve_session(session_id):
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
        synthesize_audio: bool = True,
        observer: TurnObserver | None = None,
    ) -> TurnResult:
        transcript = text.strip()
        if not transcript:
            raise ValueError("turn text must not be blank")
        runtime = await self._runtime(session_id)
        active = await self._begin_turn(
            runtime,
            synthesize_audio=synthesize_audio,
        )
        current_observer = observer or NullTurnObserver()
        try:
            return await self._process_active_turn(
                runtime,
                active,
                transcript=transcript,
                input_mode="typed_text",
                visual_summary=visual_summary,
                observer=current_observer,
            )
        finally:
            await self._finish_if_active(runtime, active)

    async def process_audio_turn(
        self,
        session_id: str,
        *,
        chunks: AsyncIterator[PcmChunk],
        transcriber: FasterWhisperProvider,
        observer: TurnObserver | None = None,
    ) -> TurnResult:
        runtime = await self._runtime(session_id)
        active = await self._begin_turn(runtime)
        current_observer = observer or NullTurnObserver()
        turn = active.handle
        transcript = ""
        stt_started_ns = time.perf_counter_ns()
        try:
            async for transcript_event in transcriber.transcribe(
                chunks,
                turn_id=turn.turn_id,
                cancel_token=turn.cancel_token,
            ):
                if transcript_event.kind is TranscriptKind.PARTIAL:
                    observed = await self._observe_current(
                        runtime,
                        active,
                        "transcript.partial",
                        transcript_event.model_dump(mode="json"),
                        observer=current_observer,
                    )
                    if not observed:
                        return self._interrupted_result(active)
                elif transcript_event.kind is TranscriptKind.FINAL:
                    transcript = transcript_event.text.strip()

            if not await self._emit_current(
                runtime,
                active,
                "provider.stt.metrics",
                {
                    "duration_ns": time.perf_counter_ns() - stt_started_ns,
                    "model": "small",
                    "device": "cpu",
                },
                observer=current_observer,
            ):
                return self._interrupted_result(active)

            if not transcript:
                if not await self._emit_current(
                    runtime,
                    active,
                    "transcript.empty",
                    {"input_mode": "voice"},
                    observer=current_observer,
                ):
                    return self._interrupted_result(active)
                if not await self._complete_active_turn(
                    runtime,
                    active,
                    {"delivery_mode": "text"},
                    observer=current_observer,
                ):
                    return self._interrupted_result(active)
                return TurnResult(
                    session_id=session_id,
                    turn_id=turn.turn_id,
                    trace_id=active.trace_id,
                    status=TurnStatus.COMPLETED,
                    response=None,
                    delivery_mode="text",
                )

            return await self._process_active_turn(
                runtime,
                active,
                transcript=transcript,
                input_mode="voice",
                visual_summary="",
                observer=current_observer,
            )
        finally:
            await self._finish_if_active(runtime, active)

    async def _process_active_turn(
        self,
        runtime: SessionRuntime,
        active: ActiveTurn,
        *,
        transcript: str,
        input_mode: Literal["typed_text", "voice"],
        visual_summary: str,
        observer: TurnObserver,
    ) -> TurnResult:
        turn = active.handle
        risk: RiskAssessment | None = None
        response: AgentResponse | None = None
        delivery_mode: Literal[
            "voice_avatar", "voice_text", "text", "interrupted"
        ] = "voice_avatar"

        if not await self._emit_current(
            runtime,
            active,
            "transcript.final",
            {"text": transcript, "input_mode": input_mode},
            observer=observer,
        ):
            return self._interrupted_result(active)

        effective_visual_summary = ""
        camera_consent = await self.store.is_consent_granted(
            turn.session_id,
            ConsentKind.CAMERA.value,
        )
        if camera_consent:
            if not await self._emit_current(
                runtime,
                active,
                "vision.burst.requested",
                {"max_frames": 4, "retention": "in_memory_only"},
                observer=observer,
            ):
                return self._interrupted_result(active)
            try:
                effective_visual_summary = await self._vision.observe(
                    session_id=turn.session_id,
                    turn=turn,
                    supplied_summary=visual_summary,
                )
            except VisionUnavailable:
                if not await self._emit_current(
                    runtime,
                    active,
                    "vision.degraded",
                    {"fallback": "voice_text_avatar"},
                    observer=observer,
                ):
                    return self._interrupted_result(active)
            else:
                visual_emitted = await self._emit_visual_if_consented(
                    runtime,
                    active,
                    {
                        "summary": effective_visual_summary,
                        "valid_for_ms": 10_000,
                    },
                    observer=observer,
                )
                if not visual_emitted and not await self._is_current(
                    runtime,
                    active,
                ):
                    return self._interrupted_result(active)
                if not visual_emitted:
                    effective_visual_summary = ""
                    if not await self._emit_current(
                        runtime,
                        active,
                        "vision.cancelled",
                        {"reason": "camera_consent_revoked"},
                        observer=observer,
                    ):
                        return self._interrupted_result(active)

        graph_result: dict[str, Any] = await self._graph.ainvoke(
            {
                "session_id": turn.session_id,
                "transcript": transcript,
                "visual_summary": effective_visual_summary,
                "turn_id": turn.turn_id,
                "cancel_token": turn.cancel_token,
                "care_loop_state": runtime.care_loop_state.model_copy(deep=True),
                "intervention_state": runtime.intervention_state.model_copy(
                    deep=True
                ),
                "visited": [],
            }
        )
        risk = cast(RiskAssessment, graph_result["risk"])
        response = cast(AgentResponse, graph_result["response"])
        avatar_plan = cast(AvatarResponsePlan, graph_result["avatar_plan"])
        care_loop_state = cast(CareLoopState, graph_result["care_loop_state"])
        intervention_state = cast(
            InterventionRuntimeState,
            graph_result["intervention_state"],
        )
        if not await self._commit_policy_state_current(
            runtime,
            active,
            care_loop_state,
            intervention_state,
        ):
            return self._interrupted_result(active, risk=risk, response=response)
        raw_metrics = graph_result.get("provider_metrics", {})
        agent_metrics: dict[str, object] = {}
        if isinstance(raw_metrics, dict):
            for key in (
                "provider",
                "total_duration_ns",
                "load_duration_ns",
                "prompt_eval_count",
                "eval_count",
            ):
                value = raw_metrics.get(key)
                if isinstance(value, str | int | float):
                    agent_metrics[key] = value
        if not await self._emit_current(
            runtime,
            active,
            "provider.agent.metrics",
            agent_metrics,
            observer=observer,
        ):
            return self._interrupted_result(active, risk=risk, response=response)
        if not await self._emit_current(
            runtime,
            active,
            "risk.updated",
            risk.model_dump(mode="json"),
            observer=observer,
        ):
            return self._interrupted_result(active, risk=risk, response=response)
        for event_type, raw_payload in (
            ("agent.decision.completed", graph_result.get("agent_trace")),
            ("care.loop.transitioned", graph_result.get("care_loop_trace")),
            (
                "intervention.policy.completed",
                graph_result.get("intervention_policy_audit"),
            ),
            (
                "evidence.context.completed",
                graph_result.get("evidence_context_audit"),
            ),
            (
                "evidence.response.completed",
                graph_result.get("evidence_response_audit"),
            ),
            (
                "capability.policy.completed",
                graph_result.get("capability_audit"),
            ),
        ):
            payload = (
                raw_payload.model_dump(mode="json")
                if isinstance(raw_payload, BaseModel)
                else {}
            )
            if not await self._emit_current(
                runtime,
                active,
                event_type,
                payload,
                observer=observer,
            ):
                return self._interrupted_result(active, risk=risk, response=response)
        if not await self._emit_current(
            runtime,
            active,
            "retrieval.completed",
            {
                "evidence_ids": response.evidence_ids,
                "reviewed_only": True,
                "memory_user_confirmed_only": True,
                "memory_ids": [
                    str(item["memory_id"])
                    for item in graph_result.get("context", {}).get(
                        "long_term_memory", []
                    )
                    if isinstance(item, dict) and "memory_id" in item
                ],
                "memory_retrieval_degraded": bool(
                    graph_result.get("context", {}).get(
                        "memory_retrieval_degraded", False
                    )
                ),
            },
            observer=observer,
        ):
            return self._interrupted_result(active, risk=risk, response=response)
        if not await self._emit_current(
            runtime,
            active,
            "avatar.plan.ready",
            avatar_plan.model_dump(mode="json"),
            observer=observer,
        ):
            return self._interrupted_result(active, risk=risk, response=response)
        if not await self._emit_current(
            runtime,
            active,
            "assistant.response.ready",
            response.model_dump(mode="json"),
            observer=observer,
        ):
            return self._interrupted_result(active, risk=risk, response=response)

        if not active.synthesize_audio:
            delivery_mode = "text"
            if not await self._emit_current(
                runtime,
                active,
                "response.displayed",
                {"text": response.display_text},
                observer=observer,
            ):
                return self._interrupted_result(active, risk=risk, response=response)
        else:
            tts_started_ns = time.perf_counter_ns()
            first_chunk_ns: int | None = None
            audio_chunk_count = 0
            try:
                async for chunk in self._audio.synthesize(
                    response.spoken_text,
                    turn=turn,
                    speech_rate=avatar_plan.speech_rate,
                ):
                    if first_chunk_ns is None:
                        first_chunk_ns = time.perf_counter_ns()
                    emitted = await self._emit_current(
                        runtime,
                        active,
                        "tts.audio.chunk",
                        {
                            "sequence": chunk.sequence,
                            "pts_ms": chunk.pts_ms,
                            "duration_ms": chunk.duration_ms,
                        },
                        observer=observer,
                    )
                    if not emitted:
                        return self._interrupted_result(
                            active,
                            risk=risk,
                            response=response,
                        )
                    if not await self._deliver_audio_current(
                        runtime,
                        active,
                        chunk,
                        observer=observer,
                    ):
                        return self._interrupted_result(
                            active,
                            risk=risk,
                            response=response,
                        )
                    audio_chunk_count += 1
            except AudioUnavailable:
                delivery_mode = "text"
                if not await self._emit_current(
                    runtime,
                    active,
                    "tts.degraded",
                    {"fallback": "text"},
                    observer=observer,
                ):
                    return self._interrupted_result(active, risk=risk, response=response)
                if not await self._emit_current(
                    runtime,
                    active,
                    "response.displayed",
                    {"text": response.display_text},
                    observer=observer,
                ):
                    return self._interrupted_result(active, risk=risk, response=response)
            else:
                if not await self._emit_current(
                    runtime,
                    active,
                    "provider.tts.metrics",
                    {
                        "first_chunk_duration_ns": (
                            0
                            if first_chunk_ns is None
                            else first_chunk_ns - tts_started_ns
                        ),
                        "total_duration_ns": time.perf_counter_ns()
                        - tts_started_ns,
                        "chunk_count": audio_chunk_count,
                    },
                    observer=observer,
                ):
                    return self._interrupted_result(
                        active,
                        risk=risk,
                        response=response,
                    )
                try:
                    avatar_result = await self._avatar.render(
                        turn,
                        plan=avatar_plan,
                        audio_chunk_count=audio_chunk_count,
                    )
                except AvatarUnavailable:
                    delivery_mode = "voice_text"
                    if not await self._emit_current(
                        runtime,
                        active,
                        "avatar.degraded",
                        {"fallback": "voice_text"},
                        observer=observer,
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
                            observer=observer,
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
                    observer=observer,
                ):
                    return self._interrupted_result(
                        active,
                        risk=risk,
                        response=response,
                    )

        if not await self._complete_active_turn(
            runtime,
            active,
            {"delivery_mode": delivery_mode},
            observer=observer,
        ):
            return self._interrupted_result(active, risk=risk, response=response)
        return TurnResult(
            session_id=turn.session_id,
            turn_id=turn.turn_id,
            trace_id=active.trace_id,
            status=TurnStatus.COMPLETED,
            risk_level=risk.level,
            response=response,
            delivery_mode=delivery_mode,
        )

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
            runtime.care_loop_state = CareLoopState()
            runtime.intervention_state = InterventionRuntimeState()
            await self.store.append_payload(
                session_id=session_id,
                event_type="session.ended",
                payload={"reason": "user_ended"},
            )
        await self._avatar.end_session(session_id)
        if self._session_end_callback is not None:
            await self._session_end_callback(session_id)
        return runtime.descriptor.model_copy(deep=True)

    async def _runtime(self, session_id: str) -> SessionRuntime:
        async with self._sessions_lock:
            runtime = self._sessions.get(session_id)
        if runtime is None:
            raise SessionNotFoundError(session_id)
        return runtime

    async def _begin_turn(
        self,
        runtime: SessionRuntime,
        *,
        synthesize_audio: bool = True,
    ) -> ActiveTurn:
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
                synthesize_audio=synthesize_audio,
            )
            runtime.active_turn = active
            return active

    async def _emit_current(
        self,
        runtime: SessionRuntime,
        active: ActiveTurn,
        event_type: str,
        payload: dict[str, object],
        *,
        observer: TurnObserver,
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
            if not await self._coordinator.tokens.is_current(
                active.handle.turn_id,
                active.handle.cancel_token,
            ):
                return False
            await observer.on_event(active.handle, event_type, payload)
            return True

    async def _commit_policy_state_current(
        self,
        runtime: SessionRuntime,
        active: ActiveTurn,
        care_loop_state: CareLoopState,
        intervention_state: InterventionRuntimeState,
    ) -> bool:
        async with runtime.event_lock:
            if runtime.active_turn is not active:
                return False
            if not await self._coordinator.tokens.is_current(
                active.handle.turn_id,
                active.handle.cancel_token,
            ):
                return False
            runtime.care_loop_state = care_loop_state.model_copy(deep=True)
            runtime.intervention_state = intervention_state.model_copy(deep=True)
            return True

    async def _observe_current(
        self,
        runtime: SessionRuntime,
        active: ActiveTurn,
        event_type: str,
        payload: dict[str, object],
        *,
        observer: TurnObserver,
    ) -> bool:
        async with runtime.event_lock:
            if runtime.active_turn is not active:
                return False
            if not await self._coordinator.tokens.is_current(
                active.handle.turn_id,
                active.handle.cancel_token,
            ):
                return False
            await observer.on_event(active.handle, event_type, payload)
            return True

    async def _deliver_audio_current(
        self,
        runtime: SessionRuntime,
        active: ActiveTurn,
        chunk: PcmChunk,
        *,
        observer: TurnObserver,
    ) -> bool:
        async with runtime.event_lock:
            if runtime.active_turn is not active:
                return False
            if not await self._coordinator.tokens.is_current(
                active.handle.turn_id,
                active.handle.cancel_token,
            ):
                return False
            await observer.on_audio(active.handle, chunk)
            return True

    async def _complete_active_turn(
        self,
        runtime: SessionRuntime,
        active: ActiveTurn,
        payload: dict[str, object],
        *,
        observer: TurnObserver,
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
                event_type="turn.completed",
                payload=payload,
            )
            try:
                if await self._coordinator.tokens.is_current(
                    active.handle.turn_id,
                    active.handle.cancel_token,
                ):
                    await observer.on_event(
                        active.handle,
                        "turn.completed",
                        payload,
                    )
            finally:
                runtime.active_turn = None
                await self._coordinator.complete_turn(active.handle)
            return True

    async def _emit_visual_if_consented(
        self,
        runtime: SessionRuntime,
        active: ActiveTurn,
        payload: dict[str, object],
        *,
        observer: TurnObserver,
    ) -> bool:
        async with runtime.event_lock:
            if runtime.active_turn is not active:
                return False
            if not await self._coordinator.tokens.is_current(
                active.handle.turn_id,
                active.handle.cancel_token,
            ):
                return False
            event = await self.store.append_payload_if_consent_granted(
                session_id=active.handle.session_id,
                consent_kind=ConsentKind.CAMERA.value,
                turn_id=active.handle.turn_id,
                trace_id=active.trace_id,
                cancel_token=active.handle.cancel_token,
                event_type="vision.observation.ready",
                payload=payload,
            )
            if event is None:
                return False
            if not await self._coordinator.tokens.is_current(
                active.handle.turn_id,
                active.handle.cancel_token,
            ):
                return False
            await observer.on_event(
                active.handle,
                "vision.observation.ready",
                payload,
            )
            return True

    async def _is_current(
        self,
        runtime: SessionRuntime,
        active: ActiveTurn,
    ) -> bool:
        async with runtime.event_lock:
            return (
                runtime.active_turn is active
                and await self._coordinator.tokens.is_current(
                    active.handle.turn_id,
                    active.handle.cancel_token,
                )
            )

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
