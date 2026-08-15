import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from app.events.store import EventStore
from app.providers.edge_tts import EdgeTtsProvider
from app.providers.faster_whisper import FasterWhisperProvider
from app.providers.mock import MockAgentProvider
from app.providers.protocols import AgentPlan
from app.providers.runtime import EdgeTtsAudioBridge
from app.realtime.models import PcmChunk, TranscriptEvent, TranscriptKind, TurnHandle
from app.realtime.session import MockAudioBridge, SessionManager, TurnStatus
from app.realtime.turn_coordinator import TurnCoordinator
from app.safety.models import RiskAssessment


class RecordingObserver:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []
        self.audio: list[PcmChunk] = []
        self.timeline: list[str] = []

    async def on_event(
        self,
        turn: TurnHandle,
        event_type: str,
        payload: dict[str, object],
    ) -> None:
        _ = turn
        self.events.append((event_type, payload))
        self.timeline.append(event_type)

    async def on_audio(
        self,
        turn: TurnHandle,
        chunk: PcmChunk,
    ) -> None:
        _ = turn
        self.audio.append(chunk)
        self.timeline.append("audio.binary")


class MetricsAgentProvider(MockAgentProvider):
    async def plan_reply(
        self,
        transcript: str,
        risk: RiskAssessment,
        context: dict[str, object],
        *,
        turn_id: str,
        cancel_token: str,
    ) -> AgentPlan:
        plan = await super().plan_reply(
            transcript,
            risk,
            context,
            turn_id=turn_id,
            cancel_token=cancel_token,
        )
        return AgentPlan(
            response=plan.response,
            provider_metrics={
                "provider": "ollama",
                "total_duration_ns": 2_000_000_000,
                "load_duration_ns": 500_000_000,
                "prompt_eval_count": 120,
                "eval_count": 48,
                "transcript": transcript,
                "prompt": "secret prompt",
                "response": plan.response.display_text,
                "access_token": "pst_secret",
            },
        )


class FakeTranscriber(FasterWhisperProvider):
    def __init__(self, events: list[TranscriptEvent]) -> None:
        self.events = events
        self.calls: list[tuple[str, str]] = []

    async def transcribe(
        self,
        chunks: AsyncIterator[PcmChunk],
        *,
        turn_id: str,
        cancel_token: str,
    ) -> AsyncIterator[TranscriptEvent]:
        self.calls.append((turn_id, cancel_token))
        async for _ in chunks:
            pass
        for event in self.events:
            yield event


async def one_pcm_chunk() -> AsyncIterator[PcmChunk]:
    yield PcmChunk(
        sequence=0,
        pts_ms=0,
        pcm_s16le=b"\x00\x00" * 320,
    )


def assert_no_sensitive_metrics(payload: dict[str, object]) -> None:
    assert set(payload).isdisjoint(
        {"transcript", "prompt", "response", "access_token", "pcm_s16le"}
    )
    assert not any(
        isinstance(value, (bytes, bytearray, memoryview))
        for value in payload.values()
    )


@pytest.mark.asyncio
async def test_text_turn_delivers_metadata_before_binary_and_filters_metrics(
    tmp_path: Path,
) -> None:
    """Catches binary audit persistence, metric leakage, or audio preceding response."""
    store = EventStore(tmp_path / "events.sqlite3")
    observer = RecordingObserver()
    manager = SessionManager(
        store=store,
        agent_provider=MetricsAgentProvider(),
    )
    session = await manager.create_session(camera_consent=False)

    result = await manager.process_text_turn(
        session.session_id,
        text="最近压力很大",
        observer=observer,
    )

    assert result.status is TurnStatus.COMPLETED
    assert len(observer.audio) == 1
    assert observer.timeline.index("assistant.response.ready") < observer.timeline.index(
        "audio.binary"
    )

    stored = await store.list_session(session.session_id)
    turn_events = [event for event in stored if event.turn_id == result.turn_id]
    audio_events = [event for event in turn_events if event.type == "tts.audio.chunk"]
    assert [event.payload for event in audio_events] == [
        {"sequence": 0, "pts_ms": 0, "duration_ms": 20}
    ]
    assert all("pcm_s16le" not in event.payload for event in turn_events)
    assert not any(
        isinstance(value, (bytes, bytearray, memoryview))
        for event in turn_events
        for value in event.payload.values()
    )

    agent_metrics = next(
        event.payload for event in turn_events if event.type == "provider.agent.metrics"
    )
    assert agent_metrics == {
        "provider": "ollama",
        "total_duration_ns": 2_000_000_000,
        "load_duration_ns": 500_000_000,
        "prompt_eval_count": 120,
        "eval_count": 48,
    }
    tts_metrics = next(
        event.payload for event in turn_events if event.type == "provider.tts.metrics"
    )
    assert set(tts_metrics) == {
        "first_chunk_duration_ns",
        "total_duration_ns",
        "chunk_count",
    }
    assert tts_metrics["chunk_count"] == 1
    assert_no_sensitive_metrics(agent_metrics)
    assert_no_sensitive_metrics(tts_metrics)


@pytest.mark.asyncio
async def test_audio_turn_observes_partial_transcript_without_persisting_it(
    tmp_path: Path,
) -> None:
    """Catches voice input auditing partial speech or starting STT outside the active turn."""
    store = EventStore(tmp_path / "events.sqlite3")
    observer = RecordingObserver()
    transcriber = FakeTranscriber(
        [
            TranscriptEvent(
                kind=TranscriptKind.PARTIAL,
                text="最近",
                start_ms=0,
                end_ms=200,
            ),
            TranscriptEvent(
                kind=TranscriptKind.FINAL,
                text="最近压力很大",
                start_ms=0,
                end_ms=900,
            ),
        ]
    )
    manager = SessionManager(store=store)
    session = await manager.create_session(camera_consent=False)

    result = await manager.process_audio_turn(
        session.session_id,
        chunks=one_pcm_chunk(),
        transcriber=transcriber,
        observer=observer,
    )

    assert result.status is TurnStatus.COMPLETED
    assert transcriber.calls == [(result.turn_id, transcriber.calls[0][1])]
    assert [event for event, _ in observer.events].count("transcript.partial") == 1
    stored = await store.list_session(session.session_id)
    turn_events = [event for event in stored if event.turn_id == result.turn_id]
    assert "transcript.partial" not in [event.type for event in turn_events]
    assert next(
        event.payload for event in turn_events if event.type == "transcript.final"
    ) == {"text": "最近压力很大", "input_mode": "voice"}
    assert set(
        next(
            event.payload
            for event in turn_events
            if event.type == "provider.stt.metrics"
        )
    ) == {"duration_ns", "model", "device"}


@pytest.mark.asyncio
async def test_audio_turn_with_no_final_transcript_completes_without_response(
    tmp_path: Path,
) -> None:
    """Catches empty STT output entering the graph or leaving the turn occupied."""
    store = EventStore(tmp_path / "events.sqlite3")
    manager = SessionManager(store=store)
    session = await manager.create_session(camera_consent=False)

    result = await manager.process_audio_turn(
        session.session_id,
        chunks=one_pcm_chunk(),
        transcriber=FakeTranscriber([]),
    )

    assert result.status is TurnStatus.COMPLETED
    assert result.response is None
    assert result.delivery_mode == "text"
    events = await store.list_session(session.session_id)
    assert "transcript.empty" in [event.type for event in events]

    follow_up = await manager.process_text_turn(
        session.session_id,
        text="改用文字",
        synthesize_audio=False,
    )
    assert follow_up.status is TurnStatus.COMPLETED


@pytest.mark.asyncio
async def test_text_only_turn_skips_tts_and_avatar(tmp_path: Path) -> None:
    """Catches the HTTP keyboard fallback synthesizing audio it cannot deliver."""
    store = EventStore(tmp_path / "events.sqlite3")
    observer = RecordingObserver()
    manager = SessionManager(
        store=store,
        audio_bridge=MockAudioBridge(available=False),
    )
    session = await manager.create_session(camera_consent=False)

    result = await manager.process_text_turn(
        session.session_id,
        text="只显示文字",
        synthesize_audio=False,
        observer=observer,
    )

    assert result.status is TurnStatus.COMPLETED
    assert result.delivery_mode == "text"
    assert observer.audio == []
    stored = await store.list_session(session.session_id)
    event_types = [event.type for event in stored]
    assert "response.displayed" in event_types
    assert "tts.audio.chunk" not in event_types
    assert "provider.tts.metrics" not in event_types
    assert "avatar.frame.ready" not in event_types


@pytest.mark.asyncio
async def test_observer_failure_keeps_audit_event_and_cancels_delivery(
    tmp_path: Path,
) -> None:
    """Catches observer failure rolling back audit history or allowing stale audio."""

    class FailingObserver(RecordingObserver):
        async def on_event(
            self,
            turn: TurnHandle,
            event_type: str,
            payload: dict[str, object],
        ) -> None:
            await super().on_event(turn, event_type, payload)
            if event_type == "assistant.response.ready":
                raise ConnectionError("socket closed")

    store = EventStore(tmp_path / "events.sqlite3")
    observer = FailingObserver()
    manager = SessionManager(store=store)
    session = await manager.create_session(camera_consent=False)

    with pytest.raises(ConnectionError, match="socket closed"):
        await manager.process_text_turn(
            session.session_id,
            text="最近压力很大",
            observer=observer,
        )

    stored = await store.list_session(session.session_id)
    event_types = [event.type for event in stored]
    assert "assistant.response.ready" in event_types
    assert "tts.audio.chunk" not in event_types


@pytest.mark.asyncio
async def test_interrupt_and_binary_delivery_share_one_ordered_gate(
    tmp_path: Path,
) -> None:
    """Catches an interrupted turn completing a stale observer audio send."""

    class BlockingAudioObserver(RecordingObserver):
        def __init__(self) -> None:
            super().__init__()
            self.audio_started = asyncio.Event()
            self.audio_release = asyncio.Event()

        async def on_audio(
            self,
            turn: TurnHandle,
            chunk: PcmChunk,
        ) -> None:
            self.audio_started.set()
            await self.audio_release.wait()
            await super().on_audio(turn, chunk)

    store = EventStore(tmp_path / "events.sqlite3")
    observer = BlockingAudioObserver()
    manager = SessionManager(store=store)
    session = await manager.create_session(camera_consent=False)
    running_turn = asyncio.create_task(
        manager.process_text_turn(
            session.session_id,
            text="最近压力很大",
            observer=observer,
        )
    )
    await asyncio.wait_for(observer.audio_started.wait(), timeout=1)

    interrupt = asyncio.create_task(
        manager.interrupt(session.session_id, reason="user_speech")
    )
    await asyncio.sleep(0)

    interrupt_blocked_by_audio_delivery = not interrupt.done()
    observer.audio_release.set()
    outcome = await asyncio.wait_for(interrupt, timeout=1)
    result = await asyncio.wait_for(running_turn, timeout=1)

    assert interrupt_blocked_by_audio_delivery
    assert outcome.interrupted
    assert result.status is TurnStatus.INTERRUPTED
    assert len(observer.audio) == 1


@pytest.mark.asyncio
async def test_completed_turn_cannot_also_be_interrupted(
    tmp_path: Path,
) -> None:
    """Catches completion releasing its lock before clearing the active turn."""

    class BlockingCompletionObserver(RecordingObserver):
        def __init__(self) -> None:
            super().__init__()
            self.completion_started = asyncio.Event()
            self.completion_release = asyncio.Event()

        async def on_event(
            self,
            turn: TurnHandle,
            event_type: str,
            payload: dict[str, object],
        ) -> None:
            await super().on_event(turn, event_type, payload)
            if event_type == "turn.completed":
                self.completion_started.set()
                await self.completion_release.wait()

    store = EventStore(tmp_path / "events.sqlite3")
    observer = BlockingCompletionObserver()
    manager = SessionManager(store=store)
    session = await manager.create_session(camera_consent=False)
    running_turn = asyncio.create_task(
        manager.process_text_turn(
            session.session_id,
            text="最近压力很大",
            observer=observer,
        )
    )
    await asyncio.wait_for(observer.completion_started.wait(), timeout=1)
    interrupt = asyncio.create_task(
        manager.interrupt(session.session_id, reason="user_speech")
    )
    await asyncio.sleep(0)
    assert not interrupt.done()

    observer.completion_release.set()
    result = await asyncio.wait_for(running_turn, timeout=1)
    outcome = await asyncio.wait_for(interrupt, timeout=1)

    assert result.status is TurnStatus.COMPLETED
    assert not outcome.interrupted
    stored = await store.list_session(session.session_id)
    turn_events = [event.type for event in stored if event.turn_id == result.turn_id]
    assert turn_events.count("turn.completed") == 1
    assert "playback.interrupted" not in turn_events


@pytest.mark.parametrize("error_type", [RuntimeError, OSError])
@pytest.mark.asyncio
async def test_edge_tts_runtime_failures_degrade_to_displayed_text(
    tmp_path: Path,
    error_type: type[Exception],
) -> None:
    """Catches real TTS/FFmpeg failures bypassing SessionManager fallback."""

    async def failing_pcm_source(
        text: str,
        voice: str,
        rate: str,
    ) -> AsyncIterator[bytes]:
        _ = (text, voice, rate)
        raise error_type("TTS subprocess failed")
        yield b""

    store = EventStore(tmp_path / "events.sqlite3")
    coordinator = TurnCoordinator()
    provider = EdgeTtsProvider(
        registry=coordinator.tokens,
        pcm_source=failing_pcm_source,
    )
    manager = SessionManager(
        store=store,
        coordinator=coordinator,
        audio_bridge=EdgeTtsAudioBridge(provider),
    )
    session = await manager.create_session(camera_consent=False)

    result = await manager.process_text_turn(
        session.session_id,
        text="请改用文字",
    )

    assert result.status is TurnStatus.COMPLETED
    assert result.delivery_mode == "text"
    stored = await store.list_session(session.session_id)
    turn_events = [event.type for event in stored if event.turn_id == result.turn_id]
    assert "tts.degraded" in turn_events
    assert "response.displayed" in turn_events
    assert "tts.audio.chunk" not in turn_events


@pytest.mark.asyncio
async def test_edge_tts_bridge_preserves_cancellation() -> None:
    """Catches cancellation being rewritten as ordinary TTS degradation."""

    async def cancelled_pcm_source(
        text: str,
        voice: str,
        rate: str,
    ) -> AsyncIterator[bytes]:
        _ = (text, voice, rate)
        raise asyncio.CancelledError
        yield b""

    coordinator = TurnCoordinator()
    turn = await coordinator.start_turn("session_1")
    bridge = EdgeTtsAudioBridge(
        EdgeTtsProvider(
            registry=coordinator.tokens,
            pcm_source=cancelled_pcm_source,
        )
    )

    with pytest.raises(asyncio.CancelledError):
        _ = [chunk async for chunk in bridge.synthesize("text", turn=turn)]
