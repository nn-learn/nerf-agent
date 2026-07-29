import asyncio
from typing import TypeVar
from uuid import uuid4

from app.contracts.session import CancellationRegistry
from app.realtime.models import (
    InterruptionResult,
    PcmChunk,
    TurnHandle,
)

QueueItem = TypeVar("QueueItem")


class TurnCoordinator:
    def __init__(self) -> None:
        self.tokens = CancellationRegistry()
        self._active: dict[str, TurnHandle] = {}
        self._audio_queue: asyncio.Queue[tuple[TurnHandle, PcmChunk]] = (
            asyncio.Queue()
        )
        self._avatar_queue: asyncio.Queue[tuple[TurnHandle, str]] = (
            asyncio.Queue()
        )
        self._lock = asyncio.Lock()

    @property
    def audio_queue_size(self) -> int:
        return self._audio_queue.qsize()

    @property
    def avatar_queue_size(self) -> int:
        return self._avatar_queue.qsize()

    async def start_turn(self, session_id: str) -> TurnHandle:
        turn_id = f"turn_{uuid4().hex}"
        token = await self.tokens.issue(turn_id)
        handle = TurnHandle(
            session_id=session_id,
            turn_id=turn_id,
            cancel_token=token,
        )
        async with self._lock:
            self._active[session_id] = handle
        return handle

    async def enqueue_audio(
        self,
        turn: TurnHandle,
        chunk: PcmChunk,
    ) -> bool:
        async with self._lock:
            if not await self.tokens.is_current(
                turn.turn_id,
                turn.cancel_token,
            ):
                return False
            await self._audio_queue.put((turn, chunk))
            return True

    async def enqueue_avatar_frame(
        self,
        turn: TurnHandle,
        *,
        frame_id: str,
    ) -> bool:
        async with self._lock:
            if not await self.tokens.is_current(
                turn.turn_id,
                turn.cancel_token,
            ):
                return False
            await self._avatar_queue.put((turn, frame_id))
            return True

    async def interrupt(
        self,
        session_id: str,
        *,
        reason: str,
    ) -> InterruptionResult:
        async with self._lock:
            turn = self._active.pop(session_id)
            await self.tokens.cancel(turn.cancel_token)
            self._drain(self._audio_queue)
            self._drain(self._avatar_queue)
            return InterruptionResult(
                session_id=session_id,
                turn_id=turn.turn_id,
                cancelled_token=turn.cancel_token,
                reason=reason,
            )

    @staticmethod
    def _drain(queue: asyncio.Queue[QueueItem]) -> None:
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            queue.task_done()
