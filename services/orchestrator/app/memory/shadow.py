import asyncio
import json
import sqlite3
import time
from collections import Counter, deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from functools import partial
from pathlib import Path
from threading import Lock
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field

from app.memory.retrieval import RetrievedMemory


class MemoryRetriever(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        user_id: str,
        purpose_scope: str = "personalization",
        k: int = 5,
        as_of_ms: int | None = None,
    ) -> list[RetrievedMemory]: ...


class LazyMemoryRetriever:
    """Construct the CPU embedding model only after an authorized shadow run."""

    def __init__(self, factory: Callable[[], MemoryRetriever]) -> None:
        self._factory = factory
        self._retriever: MemoryRetriever | None = None
        self._lock = Lock()

    @property
    def ready(self) -> bool:
        return self._retriever is not None

    def prepare(self) -> None:
        self._get_retriever()

    def retrieve(
        self,
        query: str,
        *,
        user_id: str,
        purpose_scope: str = "personalization",
        k: int = 5,
        as_of_ms: int | None = None,
    ) -> list[RetrievedMemory]:
        retriever = self._get_retriever()
        return retriever.retrieve(
            query,
            user_id=user_id,
            purpose_scope=purpose_scope,
            k=k,
            as_of_ms=as_of_ms,
        )

    def _get_retriever(self) -> MemoryRetriever:
        retriever = self._retriever
        if retriever is not None:
            return retriever
        with self._lock:
            if self._retriever is None:
                self._retriever = self._factory()
            return self._retriever


class MemoryResearchConsent(BaseModel):
    user_id: str
    granted: bool
    policy_version: str
    updated_at_ms: int = Field(ge=0)


class ShadowRun(BaseModel):
    shadow_run_id: str
    user_id: str
    session_id: str
    turn_id: str
    consent_policy_version: str
    strategy_version: str
    status: str
    baseline_latency_ms: float = Field(ge=0)
    shadow_latency_ms: float | None = Field(default=None, ge=0)
    overlap_at_5: float | None = Field(default=None, ge=0, le=1)
    rank_biased_overlap: float | None = Field(default=None, ge=0, le=1)
    baseline_count: int = Field(ge=0)
    shadow_count: int | None = Field(default=None, ge=0)
    error_code: str | None = None
    created_at_ms: int = Field(ge=0)
    completed_at_ms: int | None = Field(default=None, ge=0)
    expires_at_ms: int = Field(ge=0)


class ShadowAggregateReport(BaseModel):
    run_count: int = Field(ge=0)
    completed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    timeout_count: int = Field(ge=0)
    circuit_open_count: int = Field(ge=0)
    completion_rate: float = Field(ge=0, le=1)
    reliable: bool
    minimum_reliable_runs: int = Field(gt=0)
    warnings: list[str]
    mean_overlap_at_5: float | None = Field(default=None, ge=0, le=1)
    mean_rank_biased_overlap: float | None = Field(default=None, ge=0, le=1)
    baseline_latency_ms_p50: float | None = Field(default=None, ge=0)
    baseline_latency_ms_p95: float | None = Field(default=None, ge=0)
    shadow_latency_ms_p50: float | None = Field(default=None, ge=0)
    shadow_latency_ms_p95: float | None = Field(default=None, ge=0)
    strategy_versions: dict[str, int]


class ShadowRuntimeStatus(BaseModel):
    policy_version: str
    strategy_version: str
    model_state: str
    circuit_open: bool
    pending_count: int = Field(ge=0)
    max_concurrency: int = Field(gt=0)
    max_pending: int = Field(gt=0)


class MemoryShadowRepository:
    """Consent and metadata-only shadow ledger; query and memory text are absent."""

    def __init__(
        self,
        database_path: Path,
        *,
        retention_days: int = 30,
        minimum_reliable_runs: int = 100,
    ) -> None:
        if not 1 <= retention_days <= 90:
            raise ValueError("retention_days must be between 1 and 90")
        if minimum_reliable_runs < 30:
            raise ValueError("minimum_reliable_runs must be at least 30")
        self.database_path = database_path
        self._retention_ms = retention_days * 24 * 60 * 60 * 1000
        self._minimum_reliable_runs = minimum_reliable_runs

    def set_consent(
        self,
        *,
        user_id: str,
        granted: bool,
        policy_version: str,
        now_ms: int | None = None,
    ) -> MemoryResearchConsent:
        timestamp = now_ms if now_ms is not None else int(time.time() * 1000)
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.execute(
                """
                INSERT INTO memory_research_consents(
                    user_id, granted, policy_version, updated_at_ms
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    granted=excluded.granted,
                    policy_version=excluded.policy_version,
                    updated_at_ms=excluded.updated_at_ms
                """,
                (user_id, int(granted), policy_version, timestamp),
            )
            if not granted:
                connection.execute("PRAGMA secure_delete = ON")
                connection.execute(
                    """
                    DELETE FROM memory_shadow_rankings
                    WHERE shadow_run_id IN (
                        SELECT shadow_run_id FROM memory_shadow_runs
                        WHERE user_id = ?
                    )
                    """,
                    (user_id,),
                )
                connection.execute(
                    "DELETE FROM memory_shadow_runs WHERE user_id = ?",
                    (user_id,),
                )
        return MemoryResearchConsent(
            user_id=user_id,
            granted=granted,
            policy_version=policy_version,
            updated_at_ms=timestamp,
        )

    def consent_for_user(self, user_id: str) -> MemoryResearchConsent | None:
        with closing(sqlite3.connect(self.database_path)) as connection:
            row = connection.execute(
                """
                SELECT user_id, granted, policy_version, updated_at_ms
                FROM memory_research_consents WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        return MemoryResearchConsent(
            user_id=str(row[0]),
            granted=bool(row[1]),
            policy_version=str(row[2]),
            updated_at_ms=int(row[3]),
        )

    def is_granted(self, user_id: str, *, policy_version: str) -> bool:
        consent = self.consent_for_user(user_id)
        return bool(
            consent is not None
            and consent.granted
            and consent.policy_version == policy_version
        )

    def create_run(
        self,
        *,
        user_id: str,
        session_id: str,
        turn_id: str,
        policy_version: str,
        strategy_version: str,
        baseline: list[RetrievedMemory],
        baseline_latency_ms: float,
        now_ms: int | None = None,
    ) -> str | None:
        timestamp = now_ms if now_ms is not None else int(time.time() * 1000)
        expires_at_ms = timestamp + self._retention_ms
        self.purge_expired(now_ms=timestamp)
        run_id = f"shadow_{uuid4().hex}"
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            consent = connection.execute(
                """
                SELECT granted, policy_version FROM memory_research_consents
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
            if consent is None or not bool(consent[0]) or str(consent[1]) != policy_version:
                return None
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO memory_shadow_runs(
                    shadow_run_id, user_id, session_id, turn_id,
                    consent_policy_version, strategy_version,
                    status, baseline_latency_ms, baseline_count, created_at_ms
                    , expires_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, 'PENDING', ?, ?, ?, ?)
                """,
                (
                    run_id,
                    user_id,
                    session_id,
                    turn_id,
                    policy_version,
                    strategy_version,
                    baseline_latency_ms,
                    len(baseline),
                    timestamp,
                    expires_at_ms,
                ),
            )
            if cursor.rowcount == 0:
                return None
            self._insert_rankings(connection, run_id, "BASELINE", baseline)
        return run_id

    def complete_run(
        self,
        run_id: str,
        *,
        shadow: list[RetrievedMemory],
        latency_ms: float,
        completed_at_ms: int | None = None,
    ) -> bool:
        timestamp = (
            completed_at_ms if completed_at_ms is not None else int(time.time() * 1000)
        )
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            baseline_rows = connection.execute(
                """
                SELECT memory_id FROM memory_shadow_rankings
                WHERE shadow_run_id = ? AND arm = 'BASELINE'
                ORDER BY rank ASC
                """,
                (run_id,),
            ).fetchall()
            baseline_ids = [str(row[0]) for row in baseline_rows]
            shadow_ids = [item.memory_id for item in shadow]
            overlap = _overlap_at_k(baseline_ids, shadow_ids, k=5)
            rbo = _rank_biased_overlap(baseline_ids, shadow_ids)
            updated = connection.execute(
                """
                UPDATE memory_shadow_runs
                SET status='COMPLETED', shadow_latency_ms=?, overlap_at_5=?,
                    rank_biased_overlap=?, shadow_count=?, completed_at_ms=?
                WHERE shadow_run_id = ? AND status = 'PENDING'
                  AND EXISTS (
                      SELECT 1 FROM memory_research_consents consent
                        WHERE consent.user_id = memory_shadow_runs.user_id
                        AND consent.granted = 1
                        AND consent.policy_version =
                            memory_shadow_runs.consent_policy_version
                  )
                """,
                (latency_ms, overlap, rbo, len(shadow), timestamp, run_id),
            )
            if updated.rowcount == 0:
                return False
            self._insert_rankings(connection, run_id, "SHADOW", shadow)
        return True

    def fail_run(
        self,
        run_id: str,
        *,
        error_code: str,
        latency_ms: float | None = None,
        completed_at_ms: int | None = None,
    ) -> bool:
        timestamp = (
            completed_at_ms if completed_at_ms is not None else int(time.time() * 1000)
        )
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            updated = connection.execute(
                """
                UPDATE memory_shadow_runs SET status='FAILED', error_code=?,
                    shadow_latency_ms=?, completed_at_ms=?
                WHERE shadow_run_id = ? AND status = 'PENDING'
                """,
                (error_code, latency_ms, timestamp, run_id),
            )
        return updated.rowcount > 0

    def list_runs(self, user_id: str, *, limit: int = 100) -> list[ShadowRun]:
        self.purge_expired()
        with closing(sqlite3.connect(self.database_path)) as connection:
            rows = connection.execute(
                """
                SELECT shadow_run_id, user_id, session_id, turn_id,
                       consent_policy_version, strategy_version,
                       status, baseline_latency_ms, shadow_latency_ms, overlap_at_5,
                       rank_biased_overlap, baseline_count, shadow_count, error_code,
                       created_at_ms, completed_at_ms, expires_at_ms
                FROM memory_shadow_runs WHERE user_id = ?
                ORDER BY created_at_ms DESC LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [self._row_to_run(row) for row in rows]

    def aggregate(self, user_id: str) -> ShadowAggregateReport:
        runs = self.list_runs(user_id, limit=10_000)
        completed = [run for run in runs if run.status == "COMPLETED"]
        latencies = sorted(
            run.shadow_latency_ms for run in runs if run.shadow_latency_ms is not None
        )
        overlaps = [run.overlap_at_5 for run in completed if run.overlap_at_5 is not None]
        rbos = [
            run.rank_biased_overlap
            for run in completed
            if run.rank_biased_overlap is not None
        ]
        versions = Counter(run.strategy_version for run in runs)
        baseline_latencies = sorted(run.baseline_latency_ms for run in runs)
        reliable = len(completed) >= self._minimum_reliable_runs
        return ShadowAggregateReport(
            run_count=len(runs),
            completed_count=len(completed),
            failed_count=sum(run.status == "FAILED" for run in runs),
            timeout_count=sum(
                run.error_code in {"TIMEOUT", "INITIALIZATION_TIMEOUT"}
                for run in runs
            ),
            circuit_open_count=sum(run.error_code == "CIRCUIT_OPEN" for run in runs),
            completion_rate=len(completed) / len(runs) if runs else 0.0,
            reliable=reliable,
            minimum_reliable_runs=self._minimum_reliable_runs,
            warnings=(
                []
                if reliable
                else [
                    "INSUFFICIENT_COMPLETED_RUNS: shadow divergence and latency "
                    "are observational only"
                ]
            ),
            mean_overlap_at_5=_mean(overlaps),
            mean_rank_biased_overlap=_mean(rbos),
            baseline_latency_ms_p50=_percentile(baseline_latencies, 0.50),
            baseline_latency_ms_p95=_percentile(baseline_latencies, 0.95),
            shadow_latency_ms_p50=_percentile(latencies, 0.50),
            shadow_latency_ms_p95=_percentile(latencies, 0.95),
            strategy_versions=dict(sorted(versions.items())),
        )

    def purge_expired(self, *, now_ms: int | None = None) -> int:
        timestamp = now_ms if now_ms is not None else int(time.time() * 1000)
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.execute("PRAGMA secure_delete = ON")
            run_ids = connection.execute(
                "SELECT shadow_run_id FROM memory_shadow_runs WHERE expires_at_ms <= ?",
                (timestamp,),
            ).fetchall()
            for (run_id,) in run_ids:
                connection.execute(
                    "DELETE FROM memory_shadow_rankings WHERE shadow_run_id = ?",
                    (run_id,),
                )
            cursor = connection.execute(
                "DELETE FROM memory_shadow_runs WHERE expires_at_ms <= ?",
                (timestamp,),
            )
        return cursor.rowcount

    @staticmethod
    def _insert_rankings(
        connection: sqlite3.Connection,
        run_id: str,
        arm: str,
        items: list[RetrievedMemory],
    ) -> None:
        connection.executemany(
            """
            INSERT INTO memory_shadow_rankings(
                shadow_run_id, arm, rank, memory_id, score,
                relevance_score, reason_codes_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    arm,
                    rank,
                    item.memory_id,
                    item.score,
                    item.relevance_score,
                    json.dumps(item.reason_codes, ensure_ascii=False),
                )
                for rank, item in enumerate(items, 1)
            ],
        )

    @staticmethod
    def _row_to_run(row: tuple[Any, ...]) -> ShadowRun:
        return ShadowRun(
            shadow_run_id=str(row[0]),
            user_id=str(row[1]),
            session_id=str(row[2]),
            turn_id=str(row[3]),
            consent_policy_version=str(row[4]),
            strategy_version=str(row[5]),
            status=str(row[6]),
            baseline_latency_ms=float(row[7]),
            shadow_latency_ms=float(row[8]) if row[8] is not None else None,
            overlap_at_5=float(row[9]) if row[9] is not None else None,
            rank_biased_overlap=float(row[10]) if row[10] is not None else None,
            baseline_count=int(row[11]),
            shadow_count=int(row[12]) if row[12] is not None else None,
            error_code=str(row[13]) if row[13] is not None else None,
            created_at_ms=int(row[14]),
            completed_at_ms=int(row[15]) if row[15] is not None else None,
            expires_at_ms=int(row[16]),
        )


class MemoryShadowRunner:
    def __init__(
        self,
        *,
        repository: MemoryShadowRepository,
        retriever: MemoryRetriever,
        strategy_version: str,
        policy_version: str = "memory-shadow-research-v1",
        timeout_seconds: float = 15,
        initialization_timeout_seconds: float = 120,
        failure_threshold: int = 3,
        cooldown_seconds: float = 60,
        max_concurrency: int = 1,
        max_pending: int = 16,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if initialization_timeout_seconds <= 0:
            raise ValueError("initialization_timeout_seconds must be positive")
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if cooldown_seconds <= 0:
            raise ValueError("cooldown_seconds must be positive")
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        if max_pending <= 0:
            raise ValueError("max_pending must be positive")
        self._repository = repository
        self._retriever = retriever
        self.strategy_version = strategy_version
        self.policy_version = policy_version
        self._timeout_seconds = timeout_seconds
        self._initialization_timeout_seconds = initialization_timeout_seconds
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._max_concurrency = max_concurrency
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrency,
            thread_name_prefix="memory-shadow",
        )
        self._max_pending = max_pending
        self._tasks: set[asyncio.Task[None]] = set()
        self._task_users: dict[asyncio.Task[None], str] = {}
        self._prewarm_task: asyncio.Task[None] | None = None
        self._model_state = "ready" if not hasattr(retriever, "prepare") else "cold"
        self._failure_times: deque[float] = deque(maxlen=failure_threshold)
        self._circuit_open_until = 0.0
        self._lock = Lock()

    def submit(
        self,
        *,
        user_id: str,
        session_id: str,
        turn_id: str,
        query: str,
        baseline: list[RetrievedMemory],
        baseline_latency_ms: float,
    ) -> bool:
        if not self._repository.is_granted(
            user_id,
            policy_version=self.policy_version,
        ):
            return False
        run_id = self._repository.create_run(
            user_id=user_id,
            session_id=session_id,
            turn_id=turn_id,
            policy_version=self.policy_version,
            strategy_version=self.strategy_version,
            baseline=baseline,
            baseline_latency_ms=baseline_latency_ms,
        )
        if run_id is None:
            return False
        if len(self._tasks) >= self._max_pending:
            self._repository.fail_run(run_id, error_code="QUEUE_FULL")
            return False
        if self._is_circuit_open():
            self._repository.fail_run(run_id, error_code="CIRCUIT_OPEN")
            return False
        task = asyncio.create_task(
            self._run(
                run_id=run_id,
                user_id=user_id,
                query=query,
            )
        )
        self._tasks.add(task)
        self._task_users[task] = user_id
        task.add_done_callback(self._forget_task)
        return True

    def cancel_user(self, user_id: str) -> None:
        """Discard in-flight work after consent withdrawal; no result may persist."""
        for task, task_user_id in tuple(self._task_users.items()):
            if task_user_id == user_id:
                task.cancel()

    def start_prewarm(self) -> None:
        """Start model initialization after consent, without blocking the request."""
        if not hasattr(self._retriever, "prepare"):
            self._model_state = "ready"
            return
        if self._model_state in {"warming", "ready", "failed"}:
            return
        if self._prewarm_task is None:
            self._model_state = "warming"
            self._prewarm_task = asyncio.create_task(self._prewarm())
            self._prewarm_task.add_done_callback(
                self._consume_prewarm_exception
            )

    def status(self) -> ShadowRuntimeStatus:
        return ShadowRuntimeStatus(
            policy_version=self.policy_version,
            strategy_version=self.strategy_version,
            model_state=self._model_state,
            circuit_open=self._is_circuit_open(),
            pending_count=len(self._tasks),
            max_concurrency=self._max_concurrency,
            max_pending=self._max_pending,
        )

    async def drain(self) -> None:
        while self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)

    async def close(self) -> None:
        await self.drain()
        if self._prewarm_task is not None:
            await asyncio.gather(self._prewarm_task, return_exceptions=True)
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _forget_task(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        self._task_users.pop(task, None)

    @staticmethod
    def _consume_prewarm_exception(task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            task.exception()

    async def _run(self, *, run_id: str, user_id: str, query: str) -> None:
        try:
            await self._ensure_prepared()
        except TimeoutError:
            self._repository.fail_run(
                run_id,
                error_code="INITIALIZATION_TIMEOUT",
            )
            self._record_failure()
            return
        except Exception:
            self._repository.fail_run(
                run_id,
                error_code="INITIALIZATION_ERROR",
            )
            self._record_failure()
            return

        started = time.perf_counter()
        try:
            loop = asyncio.get_running_loop()
            shadow = await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor,
                    partial(
                        self._retriever.retrieve,
                        query,
                        user_id=user_id,
                        k=5,
                    ),
                ),
                timeout=self._timeout_seconds,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            self._repository.complete_run(
                run_id,
                shadow=shadow,
                latency_ms=latency_ms,
            )
            self._record_success()
        except TimeoutError:
            latency_ms = (time.perf_counter() - started) * 1000
            self._repository.fail_run(
                run_id,
                error_code="TIMEOUT",
                latency_ms=latency_ms,
            )
            self._record_failure()
        except Exception:
            latency_ms = (time.perf_counter() - started) * 1000
            self._repository.fail_run(
                run_id,
                error_code="RETRIEVER_ERROR",
                latency_ms=latency_ms,
            )
            self._record_failure()

    async def _ensure_prepared(self) -> None:
        if not hasattr(self._retriever, "prepare"):
            return
        if self._model_state == "failed":
            raise RuntimeError("shadow model initialization previously failed")
        self.start_prewarm()
        prewarm_task = self._prewarm_task
        if prewarm_task is not None:
            await asyncio.shield(prewarm_task)

    async def _prewarm(self) -> None:
        prepare = getattr(self._retriever, "prepare", None)
        if not callable(prepare):
            self._model_state = "ready"
            return
        loop = asyncio.get_running_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(self._executor, prepare),
                timeout=self._initialization_timeout_seconds,
            )
            self._model_state = "ready"
        except TimeoutError:
            self._model_state = "failed"
            raise
        except Exception:
            self._model_state = "failed"
            raise

    def _record_success(self) -> None:
        with self._lock:
            self._failure_times.clear()

    def _record_failure(self) -> None:
        now = time.monotonic()
        with self._lock:
            self._failure_times.append(now)
            if len(self._failure_times) >= self._failure_threshold:
                self._circuit_open_until = now + self._cooldown_seconds

    def _is_circuit_open(self) -> bool:
        with self._lock:
            return time.monotonic() < self._circuit_open_until


def _overlap_at_k(left: list[str], right: list[str], *, k: int) -> float:
    if not left and not right:
        return 1.0
    denominator = max(1, min(k, max(len(left), len(right))))
    return len(set(left[:k]) & set(right[:k])) / denominator


def _rank_biased_overlap(left: list[str], right: list[str], *, persistence: float = 0.9) -> float:
    depth = max(len(left), len(right))
    if depth == 0:
        return 1.0
    score = 0.0
    for index in range(1, depth + 1):
        agreement = len(set(left[:index]) & set(right[:index])) / index
        score += (1 - persistence) * persistence ** (index - 1) * agreement
    score += persistence**depth * len(set(left) & set(right)) / depth
    return min(1.0, score)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(len(values) - 1, lower + 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight
