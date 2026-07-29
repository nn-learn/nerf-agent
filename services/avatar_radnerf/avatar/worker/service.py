import asyncio
from dataclasses import dataclass
from typing import Literal, TypeVar

QueueItem = TypeVar("QueueItem")

@dataclass(frozen=True, slots=True)
class AvatarAudio:
    turn_id: str
    cancel_token: str
    pcm_s16le: bytes


@dataclass(frozen=True, slots=True)
class AvatarFrame:
    turn_id: str
    cancel_token: str
    rgb: bytes
    pts_ms: int


class AvatarWorker:
    def __init__(self) -> None:
        self._session_id: str | None = None
        self._cancelled: set[tuple[str, str]] = set()
        self._audio: asyncio.Queue[AvatarAudio] = asyncio.Queue()
        self._frames: asyncio.Queue[AvatarFrame] = asyncio.Queue()
        self._state: Literal[
            "offline",
            "neutral_listening",
            "speaking",
        ] = "offline"
        self._lock = asyncio.Lock()

    @property
    def audio_queue_size(self) -> int:
        return self._audio.qsize()

    @property
    def frame_queue_size(self) -> int:
        return self._frames.qsize()

    @property
    def state(self) -> str:
        return self._state

    async def start_session(self, session_id: str) -> None:
        async with self._lock:
            self._session_id = session_id
            self._cancelled.clear()
            self._drain(self._audio)
            self._drain(self._frames)
            self._state = "neutral_listening"

    async def enqueue_audio(
        self,
        *,
        turn_id: str,
        cancel_token: str,
        pcm_s16le: bytes,
    ) -> bool:
        if len(pcm_s16le) != 640:
            raise ValueError("Avatar audio must be one 20 ms PCM chunk")
        async with self._lock:
            if (
                self._session_id is None
                or (turn_id, cancel_token) in self._cancelled
            ):
                return False
            await self._audio.put(
                AvatarAudio(
                    turn_id=turn_id,
                    cancel_token=cancel_token,
                    pcm_s16le=pcm_s16le,
                )
            )
            self._state = "speaking"
            return True

    async def enqueue_frame(
        self,
        *,
        turn_id: str,
        cancel_token: str,
        rgb: bytes,
        pts_ms: int,
    ) -> bool:
        async with self._lock:
            if (
                self._session_id is None
                or (turn_id, cancel_token) in self._cancelled
            ):
                return False
            await self._frames.put(
                AvatarFrame(
                    turn_id=turn_id,
                    cancel_token=cancel_token,
                    rgb=rgb,
                    pts_ms=pts_ms,
                )
            )
            return True

    async def cancel_turn(
        self,
        *,
        turn_id: str,
        cancel_token: str,
    ) -> None:
        async with self._lock:
            self._cancelled.add((turn_id, cancel_token))
            self._drain(self._audio)
            self._drain(self._frames)
            self._state = "neutral_listening"

    @staticmethod
    def _drain(queue: asyncio.Queue[QueueItem]) -> None:
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            queue.task_done()
