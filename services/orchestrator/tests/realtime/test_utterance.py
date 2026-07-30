from collections.abc import Iterable

import pytest

from app.realtime.models import PcmChunk
from app.realtime.utterance import VadUtteranceSegmenter


class SequenceVad:
    def __init__(self, decisions: Iterable[bool]) -> None:
        self._decisions = iter(decisions)

    def is_speech(self, chunk: PcmChunk) -> bool:
        _ = chunk
        return next(self._decisions)


def make_chunk(sequence: int, *, size: int = 640) -> PcmChunk:
    return PcmChunk.model_construct(
        sequence=sequence,
        pts_ms=sequence * 20,
        pcm_s16le=bytes([sequence % 256]) * size,
        sample_rate=16000,
        channels=1,
        sample_width_bytes=2,
        duration_ms=20,
    )


def test_segmenter_includes_fifteen_idle_frames_before_speech() -> None:
    """Catches speech onset dropping the configured conversational pre-roll."""
    segmenter = VadUtteranceSegmenter(
        vad=SequenceVad([False] * 20 + [True] + [False] * 30)
    )

    results = [segmenter.push(make_chunk(index)) for index in range(51)]

    started = [result for result in results if result.speech_started]
    emitted = [result.utterance for result in results if result.utterance is not None]
    assert len(started) == 1
    assert emitted[0] is not None
    assert [chunk.sequence for chunk in emitted[0][:16]] == list(range(5, 21))


def test_thirty_silent_frames_finish_the_active_utterance() -> None:
    """Catches an utterance waiting beyond the bounded 600 ms end silence."""
    segmenter = VadUtteranceSegmenter(
        vad=SequenceVad([True] + [False] * 30)
    )

    results = [segmenter.push(make_chunk(index)) for index in range(31)]

    assert all(result.utterance is None for result in results[:-1])
    assert results[-1].utterance is not None
    assert len(results[-1].utterance) == 31


def test_one_thousand_total_frames_force_an_utterance_split() -> None:
    """Catches continuous speech growing an unbounded in-memory utterance."""
    segmenter = VadUtteranceSegmenter(vad=SequenceVad([True] * 1000))

    results = [segmenter.push(make_chunk(index)) for index in range(1000)]

    assert all(result.utterance is None for result in results[:-1])
    assert results[-1].utterance is not None
    assert len(results[-1].utterance) == 1000


def test_non_twenty_millisecond_frame_is_rejected() -> None:
    """Catches malformed binary media bypassing the 20 ms PCM contract."""
    segmenter = VadUtteranceSegmenter(vad=SequenceVad([True]))

    with pytest.raises(ValueError, match="640 bytes"):
        segmenter.push(make_chunk(0, size=639))


def test_speech_started_is_emitted_only_on_idle_to_speech_transition() -> None:
    """Catches repeated barge-in interrupts for every voiced frame."""
    segmenter = VadUtteranceSegmenter(
        vad=SequenceVad([False, True, True, False, True])
    )

    results = [segmenter.push(make_chunk(index)) for index in range(5)]

    assert [result.speech_started for result in results] == [
        False,
        True,
        False,
        False,
        False,
    ]


def test_flush_emits_only_an_active_speech_utterance() -> None:
    """Catches audio.stop turning idle pre-roll silence into a model turn."""
    idle = VadUtteranceSegmenter(vad=SequenceVad([False] * 3))
    for index in range(3):
        idle.push(make_chunk(index))

    active = VadUtteranceSegmenter(vad=SequenceVad([False, True]))
    active.push(make_chunk(10))
    active.push(make_chunk(11))

    assert idle.flush() is None
    flushed = active.flush()
    assert flushed is not None
    assert [chunk.sequence for chunk in flushed] == [10, 11]
    assert active.flush() is None


def test_reset_clears_idle_and_active_buffered_bytes() -> None:
    """Catches a later capture inheriting media from a disconnected capture."""
    segmenter = VadUtteranceSegmenter(
        vad=SequenceVad([False, True, True, False, True])
    )
    segmenter.push(make_chunk(0))
    segmenter.push(make_chunk(1))
    segmenter.push(make_chunk(2))

    segmenter.reset()
    segmenter.push(make_chunk(3))
    result = segmenter.push(make_chunk(4))

    assert result.speech_started
    flushed = segmenter.flush()
    assert flushed is not None
    assert [chunk.sequence for chunk in flushed] == [3, 4]
