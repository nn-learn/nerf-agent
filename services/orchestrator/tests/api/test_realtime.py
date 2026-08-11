import asyncio
import json
import threading
import time
import warnings
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from starlette.exceptions import StarletteDeprecationWarning
from starlette.websockets import WebSocketDisconnect

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=(
            r"Using `httpx` with `starlette\.testclient` is deprecated; "
            r"install `httpx2` instead\."
        ),
        category=StarletteDeprecationWarning,
        module=r"fastapi\.testclient",
    )
    from fastapi.testclient import TestClient

from app.api.realtime import create_realtime_router
from app.events.store import EventStore
from app.providers.faster_whisper import FasterWhisperProvider
from app.realtime.models import (
    PcmChunk,
    TranscriptEvent,
    TranscriptKind,
    TurnHandle,
)
from app.realtime.session import InterruptOutcome, SessionManager
from app.realtime.websocket import WebSocketTurnObserver
from app.settings import Settings

VALID_TOKEN = "pst_test_token_with_more_than_32_characters"
AUTH_MESSAGE = {"type": "auth", "access_token": VALID_TOKEN}
AUDIO_START = {
    "type": "audio.start",
    "sample_rate": 16000,
    "channels": 1,
    "encoding": "pcm_s16le",
}


class AlwaysSpeechVad:
    def __init__(self) -> None:
        self.calls = 0

    def is_speech(self, chunk: PcmChunk) -> bool:
        _ = chunk
        self.calls += 1
        return True


class FailingPushVad:
    def is_speech(self, chunk: PcmChunk) -> bool:
        _ = chunk
        raise RuntimeError("VAD processing failed")


class FakeTranscriber(FasterWhisperProvider):
    def __init__(self, text: str = "I need some support") -> None:
        self.text = text

    async def transcribe(
        self,
        chunks: AsyncIterator[PcmChunk],
        *,
        turn_id: str,
        cancel_token: str,
    ) -> AsyncIterator[TranscriptEvent]:
        _ = (turn_id, cancel_token)
        end_ms = 0
        async for chunk in chunks:
            end_ms = chunk.pts_ms + chunk.duration_ms
        yield TranscriptEvent(
            kind=TranscriptKind.FINAL,
            text=self.text,
            start_ms=0,
            end_ms=end_ms,
        )


class RecordingManager:
    def __init__(self, *, block_turns: bool = False) -> None:
        self.block_turns = block_turns
        self.turns: list[tuple[PcmChunk, ...]] = []
        self.interrupt_reasons: list[str] = []
        self.observers: list[WebSocketTurnObserver] = []
        self.ended_sessions: list[str] = []
        self.concurrent_turns = 0
        self.max_concurrent_turns = 0
        self._release: asyncio.Event | None = None

    async def has_access(self, session_id: str, access_token: str) -> bool:
        return session_id == "session_test" and access_token == VALID_TOKEN

    async def process_audio_turn(
        self,
        session_id: str,
        *,
        chunks: AsyncIterator[PcmChunk],
        transcriber: FasterWhisperProvider,
        observer: WebSocketTurnObserver,
    ) -> object:
        _ = transcriber
        captured = tuple([chunk async for chunk in chunks])
        self.turns.append(captured)
        self.observers.append(observer)
        self.concurrent_turns += 1
        self.max_concurrent_turns = max(
            self.max_concurrent_turns,
            self.concurrent_turns,
        )
        turn = TurnHandle(
            session_id=session_id,
            turn_id=f"turn_{len(self.turns)}",
            cancel_token=f"ct_{len(self.turns)}",
        )
        try:
            await observer.on_event(
                turn,
                "transcript.partial",
                {
                    "kind": "PARTIAL",
                    "text": f"heard {len(captured)}",
                    "start_ms": 0,
                    "end_ms": len(captured) * 20,
                },
            )
            if self.block_turns:
                self._release = asyncio.Event()
                await self._release.wait()
            await observer.on_event(
                turn,
                "turn.completed",
                {"delivery_mode": "text"},
            )
            return object()
        finally:
            self.concurrent_turns -= 1

    async def interrupt(self, session_id: str, *, reason: str) -> InterruptOutcome:
        self.interrupt_reasons.append(reason)
        if self._release is not None:
            self._release.set()
        return InterruptOutcome(
            session_id=session_id,
            interrupted=self.concurrent_turns > 0,
            turn_id="turn_active" if self.concurrent_turns > 0 else None,
            cancelled_token="ct_active" if self.concurrent_turns > 0 else None,
            reason=reason,
        )

    async def end_session(self, session_id: str) -> object:
        self.ended_sessions.append(session_id)
        return object()


class CancellationResistantManager(RecordingManager):
    def __init__(self) -> None:
        super().__init__()
        self.release_stale_turn = threading.Event()
        self.swallowed_cancellation = threading.Event()

    async def process_audio_turn(
        self,
        session_id: str,
        *,
        chunks: AsyncIterator[PcmChunk],
        transcriber: FasterWhisperProvider,
        observer: WebSocketTurnObserver,
    ) -> object:
        _ = transcriber
        captured = tuple([chunk async for chunk in chunks])
        self.turns.append(captured)
        self.observers.append(observer)
        self.concurrent_turns += 1
        self.max_concurrent_turns = max(
            self.max_concurrent_turns,
            self.concurrent_turns,
        )
        call_number = len(self.turns)
        turn = TurnHandle(
            session_id=session_id,
            turn_id=f"turn_{call_number}",
            cancel_token=f"ct_{call_number}",
        )
        try:
            await observer.on_event(
                turn,
                "transcript.partial",
                {
                    "kind": "PARTIAL",
                    "text": f"heard {len(captured)}",
                    "start_ms": 0,
                    "end_ms": len(captured) * 20,
                },
            )
            if call_number == 1:
                while not self.release_stale_turn.is_set():
                    try:
                        await asyncio.sleep(0.01)
                    except asyncio.CancelledError:
                        self.swallowed_cancellation.set()
                return object()
            await observer.on_event(
                turn,
                "turn.completed",
                {"delivery_mode": "text"},
            )
            return object()
        finally:
            self.concurrent_turns -= 1


def make_socket_app(
    tmp_path: Path,
    *,
    manager: Any | None = None,
    transcriber: FasterWhisperProvider | None = None,
    web_origin: str = "http://localhost:5173",
    auth_timeout: float = 0.2,
    vad_factory: Callable[[], AlwaysSpeechVad] = AlwaysSpeechVad,
) -> tuple[FastAPI, Any, Settings]:
    settings = Settings(
        provider_mode="mock",
        event_database_path=tmp_path / "events.sqlite3",
        web_origin=web_origin,
        websocket_auth_timeout_seconds=auth_timeout,
    )
    current_manager = manager or RecordingManager()
    app = FastAPI()
    app.include_router(
        create_realtime_router(
            settings=settings,
            manager=current_manager,
            transcriber=transcriber or FakeTranscriber(),
            vad_factory=vad_factory,
        )
    )
    return app, current_manager, settings


def connect(
    client: TestClient,
    *,
    url: str = "ws://localhost/api/sessions/session_test/realtime",
    origin: str = "http://localhost:5173",
) -> Any:
    return client.websocket_connect(url, headers={"Origin": origin})


def authenticate(socket: Any) -> None:
    socket.send_json(AUTH_MESSAGE)
    assert socket.receive_json() == {
        "type": "session.ready",
        "session_id": "session_test",
    }


def assert_policy_close(connecting: Any, *, code: int = 1008) -> None:
    with pytest.raises(WebSocketDisconnect) as closed, connecting as socket:
        socket.receive_json()
    assert closed.value.code == code


def test_wrong_origin_closes_with_policy_violation(tmp_path: Path) -> None:
    """Catches browser media access from an unconfigured origin."""
    app, _, _ = make_socket_app(tmp_path)

    with TestClient(app, base_url="http://localhost") as client:
        assert_policy_close(connect(client, origin="https://evil.example"))


def test_advertised_loopback_alias_can_authenticate(tmp_path: Path) -> None:
    """Catches the 127.0.0.1 demo page being rejected by WebSocket Origin."""
    app, _, _ = make_socket_app(tmp_path)

    with TestClient(app, base_url="http://localhost") as client, connect(
        client,
        origin="http://127.0.0.1:5173",
    ) as socket:
        authenticate(socket)


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:not-a-port",
        "http://[::1",
    ],
)
def test_malformed_origin_closes_without_handler_error(
    tmp_path: Path,
    origin: str,
) -> None:
    """Catches malformed unauthenticated Origin values escaping as tracebacks."""
    app, _, _ = make_socket_app(tmp_path)

    with TestClient(app, base_url="http://localhost") as client:
        assert_policy_close(connect(client, origin=origin))


def test_non_loopback_ws_requires_transport_security(tmp_path: Path) -> None:
    """Catches microphone PCM crossing a non-loopback plaintext WebSocket."""
    app, _, _ = make_socket_app(
        tmp_path,
        web_origin="https://care.example.com",
    )

    with TestClient(app, base_url="https://api.example.com") as client:
        assert_policy_close(
            connect(
                client,
                url="ws://api.example.com/api/sessions/session_test/realtime",
                origin="https://care.example.com",
            )
        )


def test_binary_audio_before_auth_closes_connection(tmp_path: Path) -> None:
    """Catches unauthenticated clients streaming binary microphone data."""
    app, _, _ = make_socket_app(tmp_path)

    with TestClient(app, base_url="http://localhost") as client, connect(client) as socket:
        socket.send_bytes(b"\x00" * 640)
        with pytest.raises(WebSocketDisconnect) as closed:
            socket.receive_json()

    assert closed.value.code == 1008


def test_invalid_access_token_closes_connection(tmp_path: Path) -> None:
    """Catches a structurally valid token controlling another session."""
    app, _, _ = make_socket_app(tmp_path)

    with TestClient(app, base_url="http://localhost") as client, connect(client) as socket:
        socket.send_json(
            {"type": "auth", "access_token": "x" * 40}
        )
        with pytest.raises(WebSocketDisconnect) as closed:
            socket.receive_json()

    assert closed.value.code == 1008


def test_auth_timeout_closes_connection(tmp_path: Path) -> None:
    """Catches idle unauthenticated sockets retaining server resources."""
    app, _, _ = make_socket_app(tmp_path, auth_timeout=0.01)

    with (
        TestClient(app, base_url="http://localhost") as client,
        connect(client) as socket,
        pytest.raises(WebSocketDisconnect) as closed,
    ):
        socket.receive_json()

    assert closed.value.code == 1008


def test_valid_auth_receives_session_ready(tmp_path: Path) -> None:
    """Catches media protocol startup before session ownership is verified."""
    app, _, _ = make_socket_app(tmp_path)

    with TestClient(app, base_url="http://localhost") as client, connect(client) as socket:
        authenticate(socket)


def test_vad_factory_failure_sends_error_and_closes_1011(
    tmp_path: Path,
) -> None:
    """Catches VAD initialization failures escaping the socket handler."""

    def failing_vad_factory() -> AlwaysSpeechVad:
        raise RuntimeError("VAD initialization failed")

    app, _, _ = make_socket_app(
        tmp_path,
        vad_factory=failing_vad_factory,
    )

    with TestClient(app, base_url="http://localhost") as client, connect(client) as socket:
        socket.send_json(AUTH_MESSAGE)
        assert socket.receive_json() == {
            "type": "error",
            "code": "VAD_UNAVAILABLE",
            "recoverable": False,
        }
        with pytest.raises(WebSocketDisconnect) as closed:
            socket.receive_json()

    assert closed.value.code == 1011


def test_vad_push_failure_sends_error_and_closes_1011(
    tmp_path: Path,
) -> None:
    """Catches VAD runtime failures escaping without a terminal protocol error."""
    app, _, _ = make_socket_app(
        tmp_path,
        vad_factory=FailingPushVad,
    )

    with TestClient(app, base_url="http://localhost") as client, connect(client) as socket:
        authenticate(socket)
        socket.send_json(AUDIO_START)
        socket.send_bytes(b"\x00" * 640)
        assert socket.receive_json() == {
            "type": "error",
            "code": "VAD_PROCESSING_FAILED",
            "recoverable": False,
        }
        with pytest.raises(WebSocketDisconnect) as closed:
            socket.receive_json()

    assert closed.value.code == 1011


def test_invalid_binary_frame_returns_protocol_error(tmp_path: Path) -> None:
    """Catches malformed PCM being passed to VAD or transcription."""
    app, manager, _ = make_socket_app(tmp_path)

    with TestClient(app, base_url="http://localhost") as client, connect(client) as socket:
        authenticate(socket)
        socket.send_json(AUDIO_START)
        socket.send_bytes(b"\x00" * 639)

        assert socket.receive_json() == {
            "type": "error",
            "code": "INVALID_AUDIO_FRAME",
        }

    assert manager.turns == []


def test_oversized_binary_frame_closes_with_message_too_big(
    tmp_path: Path,
) -> None:
    """Catches oversized binary media remaining on a reusable connection."""
    app, manager, _ = make_socket_app(tmp_path)

    with TestClient(app, base_url="http://localhost") as client, connect(client) as socket:
        authenticate(socket)
        socket.send_json(AUDIO_START)
        socket.send_bytes(b"\x00" * 641)

        with pytest.raises(WebSocketDisconnect) as closed:
            socket.receive_json()

    assert closed.value.code == 1009
    assert manager.turns == []


def test_continuous_speech_is_split_at_one_thousand_frames(tmp_path: Path) -> None:
    """Catches a WebSocket path bypassing the segmenter's 20-second bound."""
    app, manager, _ = make_socket_app(tmp_path)

    with TestClient(app, base_url="http://localhost") as client, connect(client) as socket:
        authenticate(socket)
        socket.send_json(AUDIO_START)
        for _ in range(1000):
            socket.send_bytes(b"\x01\x00" * 320)

        assert socket.receive_json()["type"] == "transcript.partial"

    assert len(manager.turns) == 1
    assert len(manager.turns[0]) == 1000


def test_barge_in_interrupts_and_local_turns_remain_serial(tmp_path: Path) -> None:
    """Catches a second utterance racing the active local model turn."""
    manager = RecordingManager(block_turns=True)
    app, _, _ = make_socket_app(tmp_path, manager=manager)

    with TestClient(app, base_url="http://localhost") as client, connect(client) as socket:
        authenticate(socket)
        socket.send_json(AUDIO_START)
        socket.send_bytes(b"\x01\x00" * 320)
        socket.send_json({"type": "audio.stop"})
        assert socket.receive_json()["type"] == "transcript.partial"

        socket.send_json(AUDIO_START)
        socket.send_bytes(b"\x02\x00" * 320)
        assert socket.receive_json() == {
            "type": "playback.interrupted",
            "turn_id": "turn_active",
            "reason": "user_barge_in",
        }
        socket.send_json({"type": "audio.stop"})
        assert socket.receive_json()["type"] == "transcript.partial"

    assert "user_barge_in" in manager.interrupt_reasons
    assert manager.max_concurrent_turns == 1


def test_disconnect_interrupts_turn_and_closes_observer(tmp_path: Path) -> None:
    """Catches disconnect leaving capture, model work, or socket sends alive."""
    manager = RecordingManager(block_turns=True)
    app, _, _ = make_socket_app(tmp_path, manager=manager)

    with TestClient(app, base_url="http://localhost") as client, connect(client) as socket:
        authenticate(socket)
        socket.send_json(AUDIO_START)
        socket.send_bytes(b"\x01\x00" * 320)
        socket.send_json({"type": "audio.stop"})
        assert socket.receive_json()["type"] == "transcript.partial"

    assert manager.interrupt_reasons[-1] == "websocket_disconnected"
    assert len(manager.observers) == 1
    assert manager.observers[0].closed


def test_cancellation_resistant_turn_does_not_block_next_utterance(
    tmp_path: Path,
) -> None:
    """Catches a provider swallowing cancellation and freezing socket receive."""
    manager = CancellationResistantManager()
    app, _, _ = make_socket_app(tmp_path, manager=manager)
    safety_release = threading.Timer(
        2,
        manager.release_stale_turn.set,
    )

    try:
        with TestClient(app, base_url="http://localhost") as client, connect(client) as socket:
            authenticate(socket)
            socket.send_json(AUDIO_START)
            socket.send_bytes(b"\x01\x00" * 320)
            socket.send_json({"type": "audio.stop"})
            assert socket.receive_json()["type"] == "transcript.partial"

            socket.send_json(AUDIO_START)
            socket.send_bytes(b"\x02\x00" * 320)
            assert socket.receive_json()["type"] == "playback.interrupted"
            safety_release.start()
            started = time.perf_counter()
            socket.send_json({"type": "audio.stop"})
            socket.send_bytes(b"\x03\x00" * 320)
            second_partial_received = False
            for _ in range(5):
                payload = socket.receive_json()
                if payload["type"] == "transcript.partial":
                    second_partial_received = True
                if payload == {
                    "type": "error",
                    "code": "AUDIO_NOT_STARTED",
                }:
                    break
            else:
                pytest.fail("receive loop did not process pending-turn input")
            elapsed = time.perf_counter() - started
            manager.release_stale_turn.set()
            if not second_partial_received:
                for _ in range(5):
                    if socket.receive_json()["type"] == "transcript.partial":
                        second_partial_received = True
                        break
    finally:
        safety_release.cancel()
        manager.release_stale_turn.set()

    assert elapsed < 1
    assert second_partial_received
    assert manager.swallowed_cancellation.is_set()
    assert len(manager.turns) == 2
    assert manager.max_concurrent_turns == 1


def test_disconnect_cleanup_is_bounded_when_provider_ignores_cancel(
    tmp_path: Path,
) -> None:
    """Catches disconnect awaiting a cancellation-resistant task forever."""
    manager = CancellationResistantManager()
    app, _, _ = make_socket_app(tmp_path, manager=manager)
    safety_release = threading.Timer(
        2,
        manager.release_stale_turn.set,
    )

    try:
        with TestClient(app, base_url="http://localhost") as client:
            safety_release.start()
            started = time.perf_counter()
            with connect(client) as socket:
                authenticate(socket)
                socket.send_json(AUDIO_START)
                socket.send_bytes(b"\x01\x00" * 320)
                socket.send_json({"type": "audio.stop"})
                assert socket.receive_json()["type"] == "transcript.partial"
            elapsed = time.perf_counter() - started
            manager.release_stale_turn.set()
    finally:
        safety_release.cancel()
        manager.release_stale_turn.set()

    assert elapsed < 1
    assert manager.swallowed_cancellation.is_set()
    assert manager.interrupt_reasons[-1] == "websocket_disconnected"


def test_audio_turn_event_store_payloads_contain_no_bytes(tmp_path: Path) -> None:
    """Catches the socket path leaking raw PCM into the audit store."""
    store = EventStore(tmp_path / "events.sqlite3")
    manager = SessionManager(store=store)
    session = asyncio.run(
        manager.create_session(
            camera_consent=False,
            requested_id="session_private",
        )
    )
    settings = Settings(
        provider_mode="local",
        event_database_path=store.database_path,
        web_origin="http://localhost:5173",
    )
    app = FastAPI()
    app.include_router(
        create_realtime_router(
            settings=settings,
            manager=manager,
            transcriber=FakeTranscriber(),
            vad_factory=AlwaysSpeechVad,
        )
    )

    with TestClient(app, base_url="http://localhost") as client, client.websocket_connect(
        "ws://localhost/api/sessions/session_private/realtime",
        headers={"Origin": settings.web_origin},
    ) as socket:
        socket.send_json(
            {"type": "auth", "access_token": session.access_token}
        )
        assert socket.receive_json()["type"] == "session.ready"
        socket.send_json(AUDIO_START)
        socket.send_bytes(b"\x03\x00" * 320)
        socket.send_json({"type": "audio.stop"})
        for _ in range(30):
            message = socket.receive()
            if message.get("text") is None:
                continue
            payload = json.loads(message["text"])
            if payload["type"] == "turn.completed":
                break
        else:
            pytest.fail("turn.completed was not delivered")

    events = asyncio.run(store.list_session(session.session_id))
    assert all(not _contains_bytes(event.payload) for event in events)
    assert all("pcm_s16le" not in event.payload for event in events)


def test_observer_sends_audio_metadata_binary_and_end_under_one_stream() -> None:
    """Catches binary TTS lacking stable stream metadata or post-close sends."""
    sent: list[dict[str, object] | bytes] = []

    async def send_json(payload: dict[str, object]) -> None:
        sent.append(payload)

    async def send_bytes(payload: bytes) -> None:
        sent.append(payload)

    async def exercise() -> None:
        observer = WebSocketTurnObserver(
            send_json=send_json,
            send_bytes=send_bytes,
        )
        turn = TurnHandle(
            session_id="session_test",
            turn_id="turn_test",
            cancel_token="ct_test",
        )
        await observer.on_audio(
            turn,
            PcmChunk(
                sequence=0,
                pts_ms=0,
                pcm_s16le=b"\x04\x00" * 320,
            ),
        )
        await observer.on_event(
            turn,
            "turn.completed",
            {"delivery_mode": "voice_text"},
        )
        await observer.close()
        await observer.on_event(
            turn,
            "transcript.final",
            {"text": "late", "input_mode": "voice"},
        )

    asyncio.run(exercise())

    assert sent == [
        {
            "type": "audio.start",
            "stream_id": "audio_turn_test",
            "turn_id": "turn_test",
            "sample_rate": 16000,
            "channels": 1,
            "encoding": "pcm_s16le",
        },
        b"\x04\x00" * 320,
        {
            "type": "audio.end",
            "stream_id": "audio_turn_test",
            "turn_id": "turn_test",
        },
        {
            "type": "turn.completed",
            "turn_id": "turn_test",
            "delivery_mode": "voice_text",
        },
    ]


def test_observer_maps_empty_transcript_without_raw_event_internals() -> None:
    """Catches transcript.empty being dropped or leaking non-protocol fields."""
    sent: list[dict[str, object]] = []

    async def send_json(payload: dict[str, object]) -> None:
        sent.append(payload)

    async def send_bytes(payload: bytes) -> None:
        _ = payload

    async def exercise() -> None:
        observer = WebSocketTurnObserver(
            send_json=send_json,
            send_bytes=send_bytes,
        )
        await observer.on_event(
            TurnHandle(
                session_id="session_test",
                turn_id="turn_empty",
                cancel_token="ct_secret",
            ),
            "transcript.empty",
            {
                "input_mode": "voice",
                "pcm_s16le": b"must-not-leak",
            },
        )

    asyncio.run(exercise())

    assert sent == [
        {
            "type": "transcript.empty",
            "turn_id": "turn_empty",
            "input_mode": "voice",
        }
    ]


def test_provider_mode_without_transcriber_rejects_fake_voice(
    tmp_path: Path,
) -> None:
    """Catches mock provider mode silently accepting synthetic voice."""
    app = FastAPI()
    settings = Settings(
        provider_mode="mock",
        event_database_path=tmp_path / "unavailable.sqlite3",
        web_origin="http://localhost:5173",
    )
    app.include_router(
        create_realtime_router(
            settings=settings,
            manager=RecordingManager(),
            transcriber=None,
            vad_factory=AlwaysSpeechVad,
        )
    )

    with TestClient(app, base_url="http://localhost") as client, connect(client) as socket:
        assert socket.receive_json() == {
            "type": "error",
            "code": "PROVIDER_MODE_UNAVAILABLE",
        }
        with pytest.raises(WebSocketDisconnect) as closed:
            socket.receive_json()

    assert closed.value.code == 1011


def _contains_bytes(value: object) -> bool:
    if isinstance(value, bytes | bytearray | memoryview):
        return True
    if isinstance(value, dict):
        return any(_contains_bytes(child) for child in value.values())
    if isinstance(value, list | tuple):
        return any(_contains_bytes(child) for child in value)
    return False
