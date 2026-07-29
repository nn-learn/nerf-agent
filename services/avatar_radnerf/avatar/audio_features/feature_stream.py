from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AudioFeatureConfig:
    sample_rate: int = 16_000
    frame_ms: int = 20
    wav2vec_model: str = "cpierse/wav2vec2-large-xlsr-53-esperanto"


class AudioFeatureStream:
    """Bounded PCM bridge; model inference remains in the Python 3.10 worker."""

    def __init__(
        self,
        config: AudioFeatureConfig | None = None,
        *,
        max_chunks: int = 50,
    ) -> None:
        self.config = config or AudioFeatureConfig()
        self._chunks: deque[bytes] = deque(maxlen=max_chunks)

    @property
    def queue_size(self) -> int:
        return len(self._chunks)

    def push(self, pcm_s16le: bytes) -> None:
        expected = (
            self.config.sample_rate
            * 2
            * self.config.frame_ms
            // 1_000
        )
        if len(pcm_s16le) != expected:
            raise ValueError(f"expected {expected} PCM bytes")
        self._chunks.append(pcm_s16le)

    def clear(self) -> None:
        self._chunks.clear()
