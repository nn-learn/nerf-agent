import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

from app.contracts.session import CancellationRegistry
from app.realtime.models import PcmChunk

AudioSink = Callable[[PcmChunk], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class AudioPublishResult:
    avatar_ready_before_audio: bool
    published_chunks: int


class AudioPublisher:
    def __init__(
        self,
        *,
        registry: CancellationRegistry,
        first_frame_timeout_seconds: float = 0.7,
    ) -> None:
        if first_frame_timeout_seconds <= 0:
            raise ValueError("Avatar first-frame timeout must be positive")
        self._registry = registry
        self._timeout = first_frame_timeout_seconds

    async def publish(
        self,
        chunks: AsyncIterator[PcmChunk],
        *,
        avatar_ready: asyncio.Event,
        sink: AudioSink,
        turn_id: str,
        cancel_token: str,
    ) -> AudioPublishResult:
        ready = False
        try:
            await asyncio.wait_for(
                avatar_ready.wait(),
                timeout=self._timeout,
            )
            ready = True
        except TimeoutError:
            ready = False

        published = 0
        async for chunk in chunks:
            if not await self._registry.is_current(turn_id, cancel_token):
                break
            await sink(chunk)
            published += 1
        return AudioPublishResult(
            avatar_ready_before_audio=ready,
            published_chunks=published,
        )
