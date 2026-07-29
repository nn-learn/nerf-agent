from collections.abc import Callable
from typing import Protocol, cast

from app.realtime.models import PcmChunk


class VadEngine(Protocol):
    def is_speech(self, frame: bytes, sample_rate: int) -> bool: ...


VadFactory = Callable[[int], VadEngine]


def _default_vad_factory(mode: int) -> VadEngine:
    import webrtcvad  # type: ignore[import-not-found]

    return cast(VadEngine, webrtcvad.Vad(mode))


class WebRtcVad:
    def __init__(
        self,
        *,
        aggressiveness: int = 2,
        vad_factory: VadFactory = _default_vad_factory,
    ) -> None:
        if aggressiveness not in {0, 1, 2, 3}:
            raise ValueError("VAD aggressiveness must be between 0 and 3")
        self._engine = vad_factory(aggressiveness)

    def is_speech(self, chunk: PcmChunk) -> bool:
        return self._engine.is_speech(
            chunk.pcm_s16le,
            chunk.sample_rate,
        )
