import asyncio
from contextlib import suppress
from typing import Literal

from app.memory.identity import MemorySubjectStore
from app.memory.service import MemoryIngestionService

MemoryWorkerState = Literal["queued", "processing", "failed"]


class MemoryIngestionWorker:
    """Single-consumer off-path queue that never blocks a realtime turn."""

    def __init__(
        self,
        *,
        service: MemoryIngestionService,
        subjects: MemorySubjectStore,
        max_queue_size: int = 128,
    ) -> None:
        self._service = service
        self._subjects = subjects
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=max_queue_size)
        self._queued: set[str] = set()
        self._active_session_id: str | None = None
        self._session_errors: dict[str, str] = {}
        self._task: asyncio.Task[None] | None = None
        self.completed_count = 0
        self.failed_count = 0
        self.overflow_count = 0
        self.last_error: str | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self, *, drain: bool = True) -> None:
        task = self._task
        if task is None:
            return
        if drain:
            await self._queue.join()
        self._task = None
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    async def enqueue(self, session_id: str) -> None:
        if session_id in self._queued:
            return
        self._session_errors.pop(session_id, None)
        self._queued.add(session_id)
        try:
            self._queue.put_nowait(session_id)
        except asyncio.QueueFull:
            # Session shutdown must remain realtime-safe. The durable event log and
            # ingestion cursor let startup recovery retry overflowed sessions.
            self._queued.discard(session_id)
            self.overflow_count += 1

    async def join(self) -> None:
        await self._queue.join()

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    def session_state(self, session_id: str) -> MemoryWorkerState | None:
        if self._active_session_id == session_id:
            return "processing"
        if session_id in self._queued:
            return "queued"
        if session_id in self._session_errors:
            return "failed"
        return None

    def session_error(self, session_id: str) -> str | None:
        return self._session_errors.get(session_id)

    async def _run(self) -> None:
        while True:
            session_id = await self._queue.get()
            self._active_session_id = session_id
            try:
                user_id = self._subjects.user_for_session(session_id)
                await asyncio.to_thread(
                    self._service.ingest_session,
                    user_id=user_id,
                    session_id=session_id,
                    consent_granted=False,
                )
            except Exception as error:
                self.failed_count += 1
                self.last_error = f"{type(error).__name__}: {error}"
                self._session_errors[session_id] = self.last_error
            else:
                self.completed_count += 1
                self.last_error = None
                self._session_errors.pop(session_id, None)
            finally:
                self._active_session_id = None
                self._queued.discard(session_id)
                self._queue.task_done()
