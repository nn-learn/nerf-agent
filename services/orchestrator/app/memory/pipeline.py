import time
from collections.abc import Sequence

from pydantic import BaseModel, Field

from app.memory.extraction import (
    ExtractionTelemetry,
    InstrumentedMemoryExtractor,
    MemoryExtractor,
)
from app.memory.models import (
    MemoryCandidate,
    MemoryDecision,
    MemoryItem,
    MemoryMessage,
    MemoryState,
)
from app.memory.policy import MemoryPolicy
from app.memory.repository import MemoryRepository
from app.memory.windowing import AdaptiveMessageWindower


class MemoryPipelineMetrics(BaseModel):
    message_count: int = Field(ge=0)
    window_count: int = Field(ge=0)
    extraction_batch_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    active_count: int = Field(ge=0)
    awaiting_consent_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    message_tokens: int = Field(ge=0)
    context_tokens: int = Field(ge=0)
    elapsed_ms: float = Field(ge=0)
    model_request_count: int = Field(default=0, ge=0)
    model_cache_hit_count: int = Field(default=0, ge=0)
    model_prompt_tokens: int = Field(default=0, ge=0)
    model_completion_tokens: int = Field(default=0, ge=0)
    model_total_duration_ms: float = Field(default=0, ge=0)
    model_load_duration_ms: float = Field(default=0, ge=0)
    invalid_model_candidate_count: int = Field(default=0, ge=0)
    rule_fallback_candidate_count: int = Field(default=0, ge=0)

    @property
    def context_token_ratio(self) -> float:
        if self.message_tokens == 0:
            return 0.0
        return self.context_tokens / self.message_tokens


class MemoryPipelineResult(BaseModel):
    stored_items: list[MemoryItem]
    rejected_candidates: list[MemoryCandidate]
    metrics: MemoryPipelineMetrics


class MemoryPipeline:
    """Message-to-governed-memory V1 with bounded extraction batches."""

    def __init__(
        self,
        *,
        repository: MemoryRepository,
        extractor: MemoryExtractor,
        policy: MemoryPolicy | None = None,
        windower: AdaptiveMessageWindower | None = None,
        extraction_batch_tokens: int = 4096,
        extraction_batch_windows: int = 8,
    ) -> None:
        self._repository = repository
        self._extractor = extractor
        self._policy = policy or MemoryPolicy()
        self._windower = windower or AdaptiveMessageWindower()
        self._batch_tokens = extraction_batch_tokens
        self._batch_windows = extraction_batch_windows

    def run(
        self,
        *,
        user_id: str,
        messages: Sequence[MemoryMessage],
        consent_granted: bool,
        persist: bool = True,
        now_ms: int | None = None,
    ) -> MemoryPipelineResult:
        started = time.perf_counter()
        telemetry_before = self._telemetry()
        windows = self._windower.partition(messages)
        batches = self._windower.pack_batches(
            windows,
            max_prompt_tokens=self._batch_tokens,
            max_windows=self._batch_windows,
        )
        candidates: list[MemoryCandidate] = []
        for batch in batches:
            for extracted in self._extractor.extract_batch(batch.windows):
                candidates.extend(extracted)

        to_store: list[tuple[MemoryCandidate, MemoryState]] = []
        rejected: list[MemoryCandidate] = []
        active_count = 0
        awaiting_count = 0
        for candidate in candidates:
            decision = self._policy.evaluate(
                candidate,
                consent_granted=consent_granted,
            )
            if decision is MemoryDecision.REJECT:
                rejected.append(candidate)
            elif decision is MemoryDecision.ACCEPT:
                to_store.append(
                    (
                        candidate.model_copy(update={"user_confirmed": True}),
                        MemoryState.ACTIVE,
                    )
                )
                active_count += 1
            else:
                to_store.append((candidate, MemoryState.AWAITING_CONSENT))
                awaiting_count += 1

        stored = (
            self._repository.store_candidates(
                user_id=user_id,
                candidates=to_store,
                now_ms=now_ms,
            )
            if persist and to_store
            else []
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        extraction_telemetry = self._telemetry().delta(telemetry_before)
        return MemoryPipelineResult(
            stored_items=stored,
            rejected_candidates=rejected,
            metrics=MemoryPipelineMetrics(
                message_count=len(messages),
                window_count=len(windows),
                extraction_batch_count=len(batches),
                candidate_count=len(candidates),
                active_count=active_count,
                awaiting_consent_count=awaiting_count,
                rejected_count=len(rejected),
                message_tokens=sum(window.estimated_tokens for window in windows),
                context_tokens=sum(window.context_tokens for window in windows),
                elapsed_ms=elapsed_ms,
                model_request_count=extraction_telemetry.request_count,
                model_cache_hit_count=extraction_telemetry.cache_hit_count,
                model_prompt_tokens=extraction_telemetry.prompt_eval_count,
                model_completion_tokens=extraction_telemetry.eval_count,
                model_total_duration_ms=extraction_telemetry.total_duration_ns / 1_000_000,
                model_load_duration_ms=extraction_telemetry.load_duration_ns / 1_000_000,
                invalid_model_candidate_count=(
                    extraction_telemetry.invalid_candidate_count
                ),
                rule_fallback_candidate_count=(
                    extraction_telemetry.fallback_candidate_count
                ),
            ),
        )

    def _telemetry(self) -> ExtractionTelemetry:
        if isinstance(self._extractor, InstrumentedMemoryExtractor):
            return self._extractor.telemetry()
        return ExtractionTelemetry()
