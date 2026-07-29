import asyncio
from collections.abc import AsyncIterator, Callable, Iterable
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from app.contracts.session import CancellationRegistry
from app.realtime.models import PcmChunk, TranscriptEvent, TranscriptKind


class WhisperSegment(Protocol):
    text: str
    start: float
    end: float


class WhisperModel(Protocol):
    def transcribe(
        self,
        audio: NDArray[np.float32],
        **options: object,
    ) -> tuple[Iterable[WhisperSegment], object]: ...


WhisperModelFactory = Callable[..., WhisperModel]


def _default_factory(model_name: str, **options: object) -> WhisperModel:
    from faster_whisper import WhisperModel as RuntimeWhisperModel  # type: ignore[import-not-found]

    return cast(WhisperModel, RuntimeWhisperModel(model_name, **options))


class FasterWhisperProvider:
    def __init__(
        self,
        *,
        registry: CancellationRegistry,
        model_factory: WhisperModelFactory = _default_factory,
    ) -> None:
        self._registry = registry
        self._model = model_factory(
            "small",
            device="cpu",
            compute_type="int8",
        )

    async def transcribe(
        self,
        chunks: AsyncIterator[PcmChunk],
        *,
        turn_id: str,
        cancel_token: str,
    ) -> AsyncIterator[TranscriptEvent]:
        raw = bytearray()
        async for chunk in chunks:
            if not await self._registry.is_current(turn_id, cancel_token):
                return
            raw.extend(chunk.pcm_s16le)
        if not raw or not await self._registry.is_current(turn_id, cancel_token):
            return

        audio = (
            np.frombuffer(bytes(raw), dtype="<i2").astype(np.float32)
            / 32768.0
        )

        def run() -> list[WhisperSegment]:
            segments, _ = self._model.transcribe(
                audio,
                language="zh",
                vad_filter=False,
                beam_size=5,
            )
            return list(segments)

        segments = await asyncio.to_thread(run)
        if not await self._registry.is_current(turn_id, cancel_token):
            return
        texts: list[str] = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            texts.append(text)
            yield TranscriptEvent(
                kind=TranscriptKind.PARTIAL,
                text="".join(texts),
                start_ms=max(0, round(segment.start * 1_000)),
                end_ms=max(0, round(segment.end * 1_000)),
            )
        if texts:
            yield TranscriptEvent(
                kind=TranscriptKind.FINAL,
                text="".join(texts),
                start_ms=0,
                end_ms=max(0, round(segments[-1].end * 1_000)),
            )
