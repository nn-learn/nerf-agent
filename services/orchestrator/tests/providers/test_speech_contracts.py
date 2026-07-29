from collections.abc import AsyncIterator

import pytest

from app.contracts.session import CancellationRegistry
from app.providers.edge_tts import EdgeTtsProvider, PcmFramer
from app.providers.faster_whisper import FasterWhisperProvider
from app.realtime.models import PcmChunk, TranscriptKind
from app.realtime.sentences import split_spoken_sentences
from app.realtime.vad import WebRtcVad


@pytest.mark.asyncio
async def test_faster_whisper_keeps_original_small_cpu_int8_configuration() -> None:
    """Catches silent replacement or accidental GPU allocation by the STT adapter."""
    created: dict[str, object] = {}

    class Segment:
        text = " 我最近睡得不太好。"
        start = 0.0
        end = 1.2

    class FakeModel:
        def transcribe(self, audio, **options):
            created["audio_size"] = audio.size
            created["options"] = options
            return iter([Segment()]), object()

    def factory(model_name: str, **options: object) -> FakeModel:
        created["model_name"] = model_name
        created.update(options)
        return FakeModel()

    registry = CancellationRegistry()
    token = await registry.issue("turn_1")
    provider = FasterWhisperProvider(
        registry=registry,
        model_factory=factory,
    )

    async def chunks() -> AsyncIterator[PcmChunk]:
        yield PcmChunk(
            sequence=0,
            pts_ms=0,
            pcm_s16le=b"\x00\x00" * 320,
        )

    events = [
        event
        async for event in provider.transcribe(
            chunks(),
            turn_id="turn_1",
            cancel_token=token,
        )
    ]

    expected_configuration = {
        "model_name": "small",
        "device": "cpu",
        "compute_type": "int8",
    }
    assert expected_configuration.items() <= created.items()
    assert events[-1].kind is TranscriptKind.FINAL
    assert events[-1].text == "我最近睡得不太好。"


def test_pcm_framer_emits_exact_20ms_16khz_mono_chunks() -> None:
    """Catches incompatible audio sizes reaching LiveKit or RAD-NeRF."""
    framer = PcmFramer()

    first = framer.feed(b"\x01\x00" * 500, start_pts_ms=2_000)
    second = framer.feed(b"\x02\x00" * 140, start_pts_ms=2_000)

    chunks = [*first, *second]
    assert len(chunks) == 2
    assert all(len(chunk.pcm_s16le) == 640 for chunk in chunks)
    assert [chunk.pts_ms for chunk in chunks] == [2_000, 2_020]


def test_sentence_splitter_preserves_urls_decimals_and_citations() -> None:
    """Catches TTS segmentation that corrupts facts or spoken references."""
    text = "先看 3.14 这个数，不要拆开。资料在 https://example.com/a.b。[证据1] 然后继续。"

    sentences = split_spoken_sentences(text)

    assert "3.14" in "".join(sentences)
    assert "https://example.com/a.b" in "".join(sentences)
    assert "[证据1]" in "".join(sentences)


@pytest.mark.asyncio
async def test_edge_tts_keeps_xiaoxiao_voice_and_stops_on_cancellation() -> None:
    """Catches voice drift and stale synthesized audio after barge-in."""
    captured: dict[str, str] = {}

    async def pcm_source(text: str, voice: str) -> AsyncIterator[bytes]:
        captured.update(text=text, voice=voice)
        yield b"\x01\x00" * 320
        yield b"\x02\x00" * 320

    registry = CancellationRegistry()
    token = await registry.issue("turn_1")
    provider = EdgeTtsProvider(
        registry=registry,
        pcm_source=pcm_source,
    )
    stream = provider.synthesize(
        "我们先慢慢呼吸。",
        turn_id="turn_1",
        cancel_token=token,
        start_pts_ms=2_000,
    )

    first = await anext(stream)
    await registry.cancel(token)
    remaining = [chunk async for chunk in stream]

    assert captured["voice"] == "zh-CN-XiaoxiaoNeural"
    assert first.pts_ms == 2_000
    assert remaining == []


def test_webrtc_vad_receives_original_twenty_ms_audio_contract() -> None:
    """Catches VAD format changes that disagree with STT and Avatar audio."""
    received: dict[str, object] = {}

    class FakeVad:
        def is_speech(self, frame: bytes, sample_rate: int) -> bool:
            received.update(size=len(frame), sample_rate=sample_rate)
            return True

    vad = WebRtcVad(vad_factory=lambda mode: FakeVad(), aggressiveness=2)
    result = vad.is_speech(
        PcmChunk(
            sequence=0,
            pts_ms=0,
            pcm_s16le=b"\x00\x00" * 320,
        )
    )

    assert result
    assert received == {"size": 640, "sample_rate": 16_000}
