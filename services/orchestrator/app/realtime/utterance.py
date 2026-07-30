from collections import deque
from dataclasses import dataclass

from app.realtime.models import PcmChunk
from app.realtime.vad import WebRtcVad


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    speech_started: bool = False
    utterance: tuple[PcmChunk, ...] | None = None


class VadUtteranceSegmenter:
    def __init__(
        self,
        *,
        vad: WebRtcVad,
        pre_roll_frames: int = 15,
        end_silence_frames: int = 30,
        max_utterance_frames: int = 1000,
    ) -> None:
        if pre_roll_frames < 0:
            raise ValueError("pre-roll frame count must not be negative")
        if end_silence_frames < 1 or max_utterance_frames < 1:
            raise ValueError("utterance frame limits must be positive")
        self._vad = vad
        self._pre_roll: deque[PcmChunk] = deque(maxlen=pre_roll_frames)
        self._active: list[PcmChunk] | None = None
        self._silence_frames = 0
        self._max_utterance_frames = max_utterance_frames
        self._end_silence_frames = end_silence_frames

    def push(self, chunk: PcmChunk) -> SegmentationResult:
        if len(chunk.pcm_s16le) != 640:
            raise ValueError("PCM frame must contain exactly 640 bytes")
        speech = self._vad.is_speech(chunk)
        if self._active is None:
            if not speech:
                self._pre_roll.append(chunk)
                return SegmentationResult()
            self._active = [*self._pre_roll, chunk]
            self._pre_roll.clear()
            self._silence_frames = 0
            if len(self._active) >= self._max_utterance_frames:
                return SegmentationResult(
                    speech_started=True,
                    utterance=self._finish(),
                )
            return SegmentationResult(speech_started=True)

        self._active.append(chunk)
        self._silence_frames = 0 if speech else self._silence_frames + 1
        if (
            self._silence_frames >= self._end_silence_frames
            or len(self._active) >= self._max_utterance_frames
        ):
            return SegmentationResult(utterance=self._finish())
        return SegmentationResult()

    def flush(self) -> tuple[PcmChunk, ...] | None:
        if self._active is None:
            self._pre_roll.clear()
            return None
        return self._finish()

    def reset(self) -> None:
        self._pre_roll.clear()
        self._active = None
        self._silence_frames = 0

    def _finish(self) -> tuple[PcmChunk, ...]:
        active = self._active
        if active is None:
            raise RuntimeError("cannot finish an idle utterance")
        utterance = tuple(active)
        self.reset()
        return utterance
