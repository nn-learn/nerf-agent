from app.memory.consolidation import MemoryConsolidator, MemoryProfileRepository
from app.memory.episodes import MemoryEpisodeRepository
from app.memory.pipeline import MemoryPipeline, MemoryPipelineResult
from app.memory.reader import EventMessageReader
from app.memory.repository import MemoryRepository


class MemoryIngestionService:
    """Incrementally turns persisted session events into governed memory proposals."""

    def __init__(
        self,
        *,
        reader: EventMessageReader,
        pipeline: MemoryPipeline,
        repository: MemoryRepository,
        profiles: MemoryProfileRepository | None = None,
        consolidator: MemoryConsolidator | None = None,
        episodes: MemoryEpisodeRepository | None = None,
    ) -> None:
        self._reader = reader
        self._pipeline = pipeline
        self._repository = repository
        self._profiles = profiles
        self._consolidator = consolidator
        self._episodes = episodes

    def ingest_session(
        self,
        *,
        user_id: str,
        session_id: str,
        consent_granted: bool,
        now_ms: int | None = None,
    ) -> MemoryPipelineResult:
        cursor = self._repository.get_ingestion_cursor(
            user_id=user_id,
            session_id=session_id,
        )
        messages = self._reader.read_session(
            session_id,
            after_sequence=cursor,
        )
        result = self._pipeline.run(
            user_id=user_id,
            messages=messages,
            consent_granted=consent_granted,
            now_ms=now_ms,
        )
        if self._profiles is not None:
            for item in result.stored_items:
                self._profiles.record_observation(
                    item,
                    session_id=session_id,
                    now_ms=now_ms,
                )
        if self._consolidator is not None:
            self._consolidator.rebuild_user(user_id, now_ms=now_ms)
        if self._episodes is not None:
            self._episodes.rebuild_session(
                user_id=user_id,
                session_id=session_id,
                memory_repository=self._repository,
                now_ms=now_ms,
            )
        if messages:
            self._repository.update_ingestion_cursor(
                user_id=user_id,
                session_id=session_id,
                last_sequence=max(message.sequence for message in messages),
                now_ms=now_ms,
            )
        return result
