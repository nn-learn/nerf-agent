from collections.abc import Awaitable, Callable
from dataclasses import dataclass

FrameSink = Callable[[bytes, int], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class PublishedFrame:
    rgb: bytes
    pts_ms: int


class AvatarVideoPublisher:
    """Small boundary that keeps LiveKit SDK details out of the render engine."""

    track_name = "avatar-video"

    def __init__(self, sink: FrameSink) -> None:
        self._sink = sink

    async def publish(self, frame: PublishedFrame) -> None:
        await self._sink(frame.rgb, frame.pts_ms)
