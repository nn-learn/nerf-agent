import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from typing import Literal
from urllib.parse import urlsplit

from fastapi import WebSocket
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.websockets import WebSocketDisconnect

from app.providers.faster_whisper import FasterWhisperProvider
from app.realtime.models import PcmChunk, TurnHandle
from app.realtime.session import (
    SessionManager,
    SessionNotFoundError,
)
from app.realtime.utterance import VadUtteranceSegmenter
from app.realtime.vad import WebRtcVad
from app.security.auth import (
    websocket_origin_allowed,
    websocket_transport_allowed,
)
from app.settings import Settings

MAX_TEXT_MESSAGE_BYTES = 2048
MAX_BINARY_FRAME_BYTES = 640
TURN_TASK_CANCEL_TIMEOUT_SECONDS = 0.25
POLICY_VIOLATION = 1008
MESSAGE_TOO_BIG = 1009
INTERNAL_ERROR = 1011


class _ClientMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AuthMessage(_ClientMessage):
    type: Literal["auth"]
    access_token: str = Field(min_length=32, max_length=256)


class AudioStartMessage(_ClientMessage):
    type: Literal["audio.start"]
    sample_rate: Literal[16000]
    channels: Literal[1]
    encoding: Literal["pcm_s16le"]


class AudioStopMessage(_ClientMessage):
    type: Literal["audio.stop"]


class InterruptMessage(_ClientMessage):
    type: Literal["turn.interrupt"]
    reason: str = Field(default="user_barge_in", min_length=1, max_length=64)


class SessionEndMessage(_ClientMessage):
    type: Literal["session.end"]


ClientMessage = (
    AuthMessage
    | AudioStartMessage
    | AudioStopMessage
    | InterruptMessage
    | SessionEndMessage
)
ClientMessageModel = type[
    AuthMessage
    | AudioStartMessage
    | AudioStopMessage
    | InterruptMessage
    | SessionEndMessage
]
VadFactory = Callable[[], WebRtcVad]
SendJson = Callable[[dict[str, object]], Awaitable[None]]
SendBytes = Callable[[bytes], Awaitable[None]]

_MESSAGE_MODELS: dict[str, ClientMessageModel] = {
    "auth": AuthMessage,
    "audio.start": AudioStartMessage,
    "audio.stop": AudioStopMessage,
    "turn.interrupt": InterruptMessage,
    "session.end": SessionEndMessage,
}
_EVENT_FIELDS: dict[str, tuple[str, ...]] = {
    "transcript.partial": ("kind", "text", "start_ms", "end_ms"),
    "transcript.final": ("text", "input_mode"),
    "transcript.empty": ("input_mode",),
    "risk.updated": (
        "level",
        "reasons",
        "confidence",
        "evidence_event_ids",
    ),
    "assistant.response.ready": (
        "spoken_text",
        "display_text",
        "support_mode",
        "risk_level",
        "evidence_ids",
        "memory_ids",
        "visual_observation_ids",
        "action_proposals",
        "memory_candidates",
        "avatar_style",
    ),
    "avatar.plan.ready": (
        "policy_version",
        "style",
        "speech_rate",
        "initial_pause_ms",
        "sentence_pause_ms",
        "gesture_intensity",
        "gaze_mode",
        "facial_affect",
        "interruptible",
        "max_segment_seconds",
        "reason_codes",
    ),
    "playback.interrupted": ("reason",),
    "playback.started": ("delivery_mode",),
    "response.displayed": ("text",),
    "vision.degraded": ("fallback",),
    "vision.cancelled": ("reason",),
    "tts.degraded": ("fallback",),
    "avatar.degraded": ("fallback",),
    "turn.completed": ("delivery_mode",),
}


class ClientMessageError(ValueError):
    pass


class ClientMessageTooLarge(ClientMessageError):
    pass


def _consume_task_result(task: asyncio.Task[object]) -> None:
    with suppress(asyncio.CancelledError, Exception):
        task.result()


async def _cancel_task_bounded(task: asyncio.Task[object]) -> None:
    if task.done():
        _consume_task_result(task)
        return
    task.cancel()
    done, _ = await asyncio.wait(
        {task},
        timeout=TURN_TASK_CANCEL_TIMEOUT_SECONDS,
    )
    if task in done:
        _consume_task_result(task)
        return
    task.add_done_callback(_consume_task_result)


def parse_client_message(text: str) -> ClientMessage:
    if len(text.encode("utf-8")) > MAX_TEXT_MESSAGE_BYTES:
        raise ClientMessageTooLarge
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as error:
        raise ClientMessageError("invalid JSON") from error
    if not isinstance(raw, dict):
        raise ClientMessageError("client message must be an object")
    message_type = raw.get("type")
    if not isinstance(message_type, str):
        raise ClientMessageError("client message type is required")
    model = _MESSAGE_MODELS.get(message_type)
    if model is None:
        raise ClientMessageError("unknown client message type")
    try:
        return model.model_validate(raw)
    except ValidationError as error:
        raise ClientMessageError("invalid client message") from error


class WebSocketTurnObserver:
    def __init__(
        self,
        *,
        send_json: SendJson,
        send_bytes: SendBytes,
    ) -> None:
        self._send_json = send_json
        self._send_bytes = send_bytes
        self._send_lock = asyncio.Lock()
        self._closed = False
        self._stream_turn_id: str | None = None

    @property
    def closed(self) -> bool:
        return self._closed

    async def on_event(
        self,
        turn: TurnHandle,
        event_type: str,
        payload: dict[str, object],
    ) -> None:
        fields = _EVENT_FIELDS.get(event_type)
        if fields is None:
            return
        async with self._send_lock:
            if self._closed:
                return
            if event_type == "turn.completed":
                await self._end_audio_locked(turn.turn_id)
            message: dict[str, object] = {
                "type": event_type,
                "turn_id": turn.turn_id,
            }
            message.update(
                {
                    key: payload[key]
                    for key in fields
                    if key in payload
                }
            )
            await self._send_json(message)

    async def on_audio(
        self,
        turn: TurnHandle,
        chunk: PcmChunk,
    ) -> None:
        async with self._send_lock:
            if self._closed:
                return
            if self._stream_turn_id != turn.turn_id:
                await self._send_json(
                    {
                        "type": "audio.start",
                        "stream_id": f"audio_{turn.turn_id}",
                        "turn_id": turn.turn_id,
                        "sample_rate": 16000,
                        "channels": 1,
                        "encoding": "pcm_s16le",
                    }
                )
                self._stream_turn_id = turn.turn_id
            await self._send_bytes(chunk.pcm_s16le)

    async def send_protocol(self, payload: dict[str, object]) -> None:
        async with self._send_lock:
            if not self._closed:
                await self._send_json(payload)

    async def send_interruption(
        self,
        *,
        turn_id: str | None,
        reason: str,
    ) -> None:
        async with self._send_lock:
            if self._closed:
                return
            if turn_id is not None:
                await self._end_audio_locked(turn_id)
            await self._send_json(
                {
                    "type": "playback.interrupted",
                    "turn_id": turn_id,
                    "reason": reason,
                }
            )

    async def close(self) -> None:
        async with self._send_lock:
            self._closed = True
            self._stream_turn_id = None

    async def _end_audio_locked(self, turn_id: str) -> None:
        if self._stream_turn_id != turn_id:
            return
        await self._send_json(
            {
                "type": "audio.end",
                "stream_id": f"audio_{turn_id}",
                "turn_id": turn_id,
            }
        )
        self._stream_turn_id = None


async def handle_realtime_websocket(
    websocket: WebSocket,
    *,
    session_id: str,
    settings: Settings,
    manager: SessionManager,
    transcriber: FasterWhisperProvider | None,
    vad_factory: VadFactory,
) -> None:
    origin = websocket.headers.get("origin")
    request_host = urlsplit(str(websocket.url)).hostname
    if not websocket_origin_allowed(
        origin,
        configured_origin=settings.web_origin,
    ):
        await websocket.close(
            code=POLICY_VIOLATION,
            reason="origin not allowed",
        )
        return
    if not websocket_transport_allowed(
        scheme=websocket.url.scheme,
        request_host=request_host,
        configured_origin=settings.web_origin,
    ):
        await websocket.close(
            code=POLICY_VIOLATION,
            reason="secure WebSocket required",
        )
        return

    await websocket.accept()
    if transcriber is None:
        await websocket.send_json(
            {"type": "error", "code": "PROVIDER_MODE_UNAVAILABLE"}
        )
        await websocket.close(
            code=INTERNAL_ERROR,
            reason="voice provider unavailable",
        )
        return

    authenticated = False
    observer: WebSocketTurnObserver | None = None
    segmenter: VadUtteranceSegmenter | None = None
    turn_task: asyncio.Task[object] | None = None
    turn_obsolete: asyncio.Event | None = None
    pending_utterance: tuple[PcmChunk, ...] | None = None
    accepting_turns = True
    capture_active = False
    sequence = 0

    async def send_json(payload: dict[str, object]) -> None:
        await websocket.send_json(payload)

    async def send_bytes(payload: bytes) -> None:
        await websocket.send_bytes(payload)

    async def close_for_vad_error(
        current_observer: WebSocketTurnObserver,
        *,
        code: str,
        reason: str,
    ) -> None:
        with suppress(RuntimeError, WebSocketDisconnect):
            await current_observer.send_protocol(
                {
                    "type": "error",
                    "code": code,
                    "recoverable": False,
                }
            )
        with suppress(RuntimeError):
            await websocket.close(
                code=INTERNAL_ERROR,
                reason=reason,
            )

    async def run_audio_turn(
        utterance: tuple[PcmChunk, ...],
        current_observer: WebSocketTurnObserver,
        obsolete: asyncio.Event,
    ) -> object:
        nonlocal accepting_turns, pending_utterance

        async def chunks() -> AsyncIterator[PcmChunk]:
            for chunk in utterance:
                yield chunk

        try:
            return await manager.process_audio_turn(
                session_id,
                chunks=chunks(),
                transcriber=transcriber,
                observer=current_observer,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            if obsolete.is_set():
                raise
            accepting_turns = False
            pending_utterance = None
            await current_observer.send_protocol(
                {"type": "error", "code": "PROVIDER_ERROR"}
            )
            with suppress(RuntimeError):
                await websocket.close(
                    code=INTERNAL_ERROR,
                    reason="voice provider failed",
            )
            raise

    def on_turn_done(completed: asyncio.Task[object]) -> None:
        nonlocal pending_utterance, turn_obsolete, turn_task
        _consume_task_result(completed)
        if turn_task is not completed:
            return
        turn_task = None
        turn_obsolete = None
        if (
            not accepting_turns
            or pending_utterance is None
            or observer is None
        ):
            return
        next_utterance = pending_utterance
        pending_utterance = None
        launch_turn(next_utterance, observer)

    def launch_turn(
        utterance: tuple[PcmChunk, ...],
        current_observer: WebSocketTurnObserver,
    ) -> None:
        nonlocal turn_obsolete, turn_task
        turn_obsolete = asyncio.Event()
        turn_task = asyncio.create_task(
            run_audio_turn(
                utterance,
                current_observer,
                turn_obsolete,
            )
        )
        turn_task.add_done_callback(on_turn_done)

    def start_or_queue_turn(
        utterance: tuple[PcmChunk, ...],
        current_observer: WebSocketTurnObserver,
    ) -> None:
        nonlocal pending_utterance
        if not accepting_turns:
            return
        if turn_task is None:
            launch_turn(utterance, current_observer)
            return
        pending_utterance = utterance

    async def interrupt_current_turn(
        current_observer: WebSocketTurnObserver,
        *,
        reason: str,
    ) -> None:
        active_task = turn_task
        active_obsolete = turn_obsolete
        if active_obsolete is not None:
            active_obsolete.set()
        outcome = await manager.interrupt(
            session_id,
            reason=reason,
        )
        if active_task is not None and not active_task.done():
            active_task.cancel()
        if outcome.interrupted:
            await current_observer.send_interruption(
                turn_id=outcome.turn_id,
                reason=reason,
            )

    try:
        try:
            first = await asyncio.wait_for(
                websocket.receive(),
                timeout=settings.websocket_auth_timeout_seconds,
            )
        except TimeoutError:
            await websocket.close(
                code=POLICY_VIOLATION,
                reason="authentication timeout",
            )
            return
        if first["type"] == "websocket.disconnect":
            return
        first_text = first.get("text")
        if not isinstance(first_text, str):
            await websocket.close(
                code=POLICY_VIOLATION,
                reason="authentication required before media",
            )
            return
        try:
            auth = parse_client_message(first_text)
        except ClientMessageTooLarge:
            await websocket.close(
                code=MESSAGE_TOO_BIG,
                reason="message too large",
            )
            return
        except ClientMessageError:
            await websocket.close(
                code=POLICY_VIOLATION,
                reason="invalid authentication",
            )
            return
        if not isinstance(auth, AuthMessage):
            await websocket.close(
                code=POLICY_VIOLATION,
                reason="authentication must be first",
            )
            return
        try:
            has_access = await manager.has_access(
                session_id,
                auth.access_token,
            )
        except SessionNotFoundError:
            has_access = False
        if not has_access:
            await websocket.close(
                code=POLICY_VIOLATION,
                reason="session access denied",
            )
            return

        authenticated = True
        observer = WebSocketTurnObserver(
            send_json=send_json,
            send_bytes=send_bytes,
        )
        try:
            segmenter = VadUtteranceSegmenter(
                vad=vad_factory(),
                end_silence_frames=settings.utterance_end_silence_ms // 20,
                max_utterance_frames=settings.utterance_max_ms // 20,
            )
        except Exception:
            await close_for_vad_error(
                observer,
                code="VAD_UNAVAILABLE",
                reason="VAD initialization failed",
            )
            return
        await observer.send_protocol(
            {"type": "session.ready", "session_id": session_id}
        )

        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            binary = message.get("bytes")
            if isinstance(binary, bytes):
                if len(binary) > MAX_BINARY_FRAME_BYTES:
                    await websocket.close(
                        code=MESSAGE_TOO_BIG,
                        reason="binary frame too large",
                    )
                    break
                if not capture_active:
                    await observer.send_protocol(
                        {"type": "error", "code": "AUDIO_NOT_STARTED"}
                    )
                    continue
                if len(binary) != MAX_BINARY_FRAME_BYTES:
                    await observer.send_protocol(
                        {"type": "error", "code": "INVALID_AUDIO_FRAME"}
                    )
                    continue
                chunk = PcmChunk(
                    sequence=sequence,
                    pts_ms=sequence * 20,
                    pcm_s16le=binary,
                )
                sequence += 1
                try:
                    result = segmenter.push(chunk)
                except Exception:
                    accepting_turns = False
                    pending_utterance = None
                    await close_for_vad_error(
                        observer,
                        code="VAD_PROCESSING_FAILED",
                        reason="VAD processing failed",
                    )
                    break
                if (
                    result.speech_started
                    and turn_task is not None
                    and not turn_task.done()
                ):
                    await interrupt_current_turn(
                        observer,
                        reason="user_barge_in",
                    )
                if result.utterance is not None:
                    start_or_queue_turn(result.utterance, observer)
                continue

            text = message.get("text")
            if not isinstance(text, str):
                await observer.send_protocol(
                    {"type": "error", "code": "INVALID_CLIENT_MESSAGE"}
                )
                continue
            try:
                control = parse_client_message(text)
            except ClientMessageTooLarge:
                await websocket.close(
                    code=MESSAGE_TOO_BIG,
                    reason="message too large",
                )
                break
            except ClientMessageError:
                await observer.send_protocol(
                    {"type": "error", "code": "INVALID_CLIENT_MESSAGE"}
                )
                continue

            if isinstance(control, AudioStartMessage):
                if capture_active:
                    await observer.send_protocol(
                        {"type": "error", "code": "AUDIO_ALREADY_STARTED"}
                    )
                    continue
                capture_active = True
                segmenter.reset()
            elif isinstance(control, AudioStopMessage):
                if not capture_active:
                    await observer.send_protocol(
                        {"type": "error", "code": "AUDIO_NOT_STARTED"}
                    )
                    continue
                capture_active = False
                utterance = segmenter.flush()
                if utterance is not None:
                    start_or_queue_turn(utterance, observer)
            elif isinstance(control, InterruptMessage):
                await interrupt_current_turn(
                    observer,
                    reason=control.reason,
                )
            elif isinstance(control, SessionEndMessage):
                accepting_turns = False
                pending_utterance = None
                capture_active = False
                segmenter.reset()
                await manager.end_session(session_id)
                break
            else:
                await observer.send_protocol(
                    {"type": "error", "code": "INVALID_CLIENT_MESSAGE"}
                )
    except WebSocketDisconnect:
        pass
    finally:
        accepting_turns = False
        pending_utterance = None
        capture_active = False
        if segmenter is not None:
            segmenter.reset()
        if observer is not None:
            await observer.close()
        if authenticated:
            with suppress(SessionNotFoundError):
                await manager.interrupt(
                    session_id,
                    reason="websocket_disconnected",
                )
        active_task = turn_task
        if active_task is not None:
            active_obsolete = turn_obsolete
            if active_obsolete is not None:
                active_obsolete.set()
            await _cancel_task_bounded(active_task)
