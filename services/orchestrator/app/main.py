import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast
from urllib.parse import urlsplit, urlunsplit

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.clinician import create_clinician_router
from app.api.consents import create_consent_router
from app.api.livekit import create_livekit_router
from app.api.memory import create_memory_router
from app.api.realtime import create_realtime_router
from app.api.sessions import create_sessions_router
from app.events.store import EventStore
from app.memory.consolidation import MemoryConsolidator, MemoryProfileRepository
from app.memory.episodes import (
    EpisodeAugmentedMemoryRetriever,
    FreshnessAwareMemoryRetriever,
    MemoryEpisodeRepository,
    MemoryRetrieverPort,
)
from app.memory.identity import MemorySubjectStore
from app.memory.models import MemoryRecall
from app.memory.pipeline import MemoryPipeline
from app.memory.reader import EventMessageReader
from app.memory.repository import MemoryRepository
from app.memory.retrieval import (
    GovernedHybridMemoryRetriever,
    GovernedMemoryRetriever,
    GovernedProfileRetriever,
    LayeredMemoryRetriever,
)
from app.memory.runtime import build_memory_claim_normalizer, build_memory_extractor
from app.memory.service import MemoryIngestionService
from app.memory.shadow import (
    LazyMemoryRetriever,
    MemoryRetriever,
    MemoryShadowRepository,
    MemoryShadowRunner,
)
from app.memory.worker import MemoryIngestionWorker
from app.providers.runtime import RuntimeProviders, build_runtime_providers
from app.rag.bge import BgeM3EmbeddingProvider
from app.realtime.session import SessionManager
from app.realtime.turn_coordinator import TurnCoordinator
from app.security.middleware import (
    InMemoryRateLimitMiddleware,
    SecurityHeadersMiddleware,
)
from app.settings import Settings


def _browser_origins(configured_origin: str) -> list[str]:
    origins = [configured_origin]
    parsed = urlsplit(configured_origin)
    alias = {
        "localhost": "127.0.0.1",
        "127.0.0.1": "localhost",
    }.get(parsed.hostname or "")
    if alias is None:
        return origins
    port = f":{parsed.port}" if parsed.port is not None else ""
    origins.append(
        urlunsplit((parsed.scheme, f"{alias}{port}", "", "", ""))
    )
    return origins


def create_app(
    settings: Settings | None = None,
    *,
    runtime_providers: RuntimeProviders | None = None,
    coordinator: TurnCoordinator | None = None,
    memory_shadow_retriever: MemoryRetriever | None = None,
) -> FastAPI:
    current = settings or Settings()
    owns_providers = runtime_providers is None
    if runtime_providers is None:
        active_coordinator = coordinator or TurnCoordinator()
        providers = build_runtime_providers(
            current,
            registry=active_coordinator.tokens,
        )
    else:
        if coordinator is None:
            raise ValueError(
                "coordinator is required with injected runtime providers"
            )
        if runtime_providers.registry is not coordinator.tokens:
            raise ValueError(
                "runtime provider registry must be coordinator.tokens"
            )
        active_coordinator = coordinator
        providers = runtime_providers

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        _ = application
        providers.start_prewarm()
        memory_worker.start()
        for pending_session_id in memory_subjects.pending_ended_sessions():
            await memory_worker.enqueue(pending_session_id)
        try:
            yield
        finally:
            if memory_shadow_runner is not None:
                await memory_shadow_runner.close()
            await memory_worker.stop()
            close_extractor = getattr(memory_extractor, "close", None)
            if callable(close_extractor):
                close_extractor()
            close_claim_normalizer = getattr(memory_claim_normalizer, "close", None)
            if callable(close_claim_normalizer):
                close_claim_normalizer()
            if owns_providers:
                await providers.aclose()

    app = FastAPI(
        title="PsyAvatar Care Orchestrator",
        lifespan=lifespan,
    )
    event_store = EventStore(current.event_database_path)
    memory_repository = MemoryRepository(current.event_database_path)
    memory_repository.initialize()
    memory_subjects = MemorySubjectStore(current.event_database_path)
    memory_profiles = MemoryProfileRepository(current.event_database_path)
    memory_claim_normalizer = build_memory_claim_normalizer(current)
    memory_consolidator = MemoryConsolidator(
        memory_profiles,
        normalizer=memory_claim_normalizer,
    )
    layered_memory_retriever = LayeredMemoryRetriever(
        GovernedMemoryRetriever(memory_repository),
        GovernedProfileRetriever(memory_profiles),
    )
    memory_episodes = MemoryEpisodeRepository(
        current.event_database_path,
        max_members=current.memory_episode_max_members,
        max_summary_chars=current.memory_episode_max_chars,
        time_gap_ms=current.memory_episode_time_gap_minutes * 60 * 1000,
    )
    governed_memory_retriever: MemoryRetrieverPort = layered_memory_retriever
    if current.memory_episode_summary_enabled:
        governed_memory_retriever = EpisodeAugmentedMemoryRetriever(
            governed_memory_retriever,
            episodes=memory_episodes,
            memories=memory_repository,
        )
    if current.memory_freshness_rerank_enabled:
        governed_memory_retriever = FreshnessAwareMemoryRetriever(
            governed_memory_retriever,
            minimum_factor=current.memory_freshness_minimum_factor,
        )
    memory_shadow_repository = MemoryShadowRepository(
        current.event_database_path,
        retention_days=current.memory_shadow_retention_days,
        minimum_reliable_runs=current.memory_shadow_min_reliable_runs,
    )
    memory_shadow_repository.purge_expired()
    if current.memory_shadow_enabled:
        shadow_retriever = memory_shadow_retriever or LazyMemoryRetriever(
            lambda: GovernedHybridMemoryRetriever(
                memory_repository,
                embedding_provider=BgeM3EmbeddingProvider(),
                semantic_min_relevance=(
                    current.memory_shadow_semantic_min_relevance
                ),
            )
        )
        memory_shadow_runner: MemoryShadowRunner | None = MemoryShadowRunner(
            repository=memory_shadow_repository,
            retriever=shadow_retriever,
            strategy_version=current.memory_shadow_strategy_version,
            policy_version=current.memory_shadow_policy_version,
            timeout_seconds=current.memory_shadow_timeout_seconds,
            initialization_timeout_seconds=(
                current.memory_shadow_initialization_timeout_seconds
            ),
            failure_threshold=current.memory_shadow_failure_threshold,
            cooldown_seconds=current.memory_shadow_cooldown_seconds,
            max_concurrency=current.memory_shadow_max_concurrency,
            max_pending=current.memory_shadow_max_pending,
        )
    else:
        memory_shadow_runner = None
    memory_extractor = build_memory_extractor(current)
    memory_ingestion = MemoryIngestionService(
        reader=EventMessageReader(current.event_database_path),
        pipeline=MemoryPipeline(
            repository=memory_repository,
            extractor=memory_extractor,
            extraction_batch_tokens=current.memory_extraction_batch_tokens,
            extraction_batch_windows=current.memory_extraction_batch_windows,
        ),
        repository=memory_repository,
        profiles=memory_profiles,
        consolidator=memory_consolidator,
        episodes=memory_episodes,
    )
    memory_worker = MemoryIngestionWorker(
        service=memory_ingestion,
        subjects=memory_subjects,
        max_queue_size=current.memory_ingestion_queue_size,
    )

    async def load_memory_context(
        session_id: str,
        turn_id: str,
        query: str,
    ) -> list[dict[str, object]]:
        user_id = memory_subjects.user_for_session(session_id)
        baseline_started = time.perf_counter()
        items = await asyncio.to_thread(
            governed_memory_retriever.retrieve,
            query,
            user_id=user_id,
        )
        baseline_latency_ms = (time.perf_counter() - baseline_started) * 1000
        now_ms = int(time.time() * 1000)
        await asyncio.to_thread(
            memory_repository.record_retrievals,
            user_id=user_id,
            session_id=session_id,
            turn_id=turn_id,
            retrievals=[
                MemoryRecall(
                    memory_id=item.memory_id,
                    session_id=session_id,
                    turn_id=turn_id,
                    score=item.score,
                    relevance_score=item.relevance_score,
                    reason_codes=item.reason_codes,
                    used_at_ms=now_ms,
                )
                for item in items
            ],
        )
        if memory_shadow_runner is not None:
            memory_shadow_runner.submit(
                user_id=user_id,
                session_id=session_id,
                turn_id=turn_id,
                query=query,
                baseline=items,
                baseline_latency_ms=baseline_latency_ms,
            )
        return governed_memory_retriever.to_model_context(items)

    session_manager = SessionManager(
        store=event_store,
        coordinator=active_coordinator,
        agent_provider=providers.agent_provider,
        audio_bridge=providers.audio_bridge,
        provider_mode=providers.provider_mode,
        session_end_callback=memory_worker.enqueue,
        memory_context_loader=load_memory_context,
    )
    app.state.event_store = event_store
    app.state.session_manager = session_manager
    app.state.runtime_providers = providers
    app.state.memory_repository = memory_repository
    app.state.memory_subjects = memory_subjects
    app.state.memory_worker = memory_worker
    app.state.memory_profiles = memory_profiles
    app.state.memory_consolidator = memory_consolidator
    app.state.memory_episodes = memory_episodes
    app.state.memory_claim_normalizer = memory_claim_normalizer
    app.state.memory_shadow_repository = memory_shadow_repository
    app.state.memory_shadow_runner = memory_shadow_runner
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_browser_origins(current.web_origin),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Content-Type", "Authorization"],
    )
    app.add_middleware(
        InMemoryRateLimitMiddleware,
        requests_per_minute=current.request_rate_limit_per_minute,
    )
    app.add_middleware(SecurityHeadersMiddleware)
    app.include_router(create_livekit_router(current, session_manager))
    app.include_router(
        create_realtime_router(
            settings=current,
            manager=session_manager,
            transcriber=providers.transcriber,
        )
    )
    def bind_memory_subject(session_id: str, token: str | None) -> str | None:
        _, issued = memory_subjects.issue_or_resolve(
            session_id=session_id,
            supplied_token=token,
        )
        return issued

    app.include_router(
        create_sessions_router(
            session_manager,
            bind_memory_subject=bind_memory_subject,
            validate_memory_subject=memory_subjects.validate_existing,
        )
    )
    app.include_router(create_consent_router(event_store, session_manager))
    app.include_router(
        create_memory_router(
            manager=session_manager,
            repository=memory_repository,
            subjects=memory_subjects,
            store=event_store,
            worker=memory_worker,
            profiles=memory_profiles,
            consolidator=memory_consolidator,
            episodes=memory_episodes,
            shadow_repository=memory_shadow_repository,
            shadow_runner=memory_shadow_runner,
            shadow_policy_version=current.memory_shadow_policy_version,
            shadow_strategy_version=current.memory_shadow_strategy_version,
        )
    )
    app.include_router(create_clinician_router(event_store))

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "live"}

    @app.get("/health/ready")
    async def ready() -> JSONResponse:
        readiness = await providers.readiness()
        provider_mode = str(readiness["provider_mode"])
        if provider_mode != "local":
            return JSONResponse(
                {
                    "status": "ready",
                    "provider_mode": provider_mode,
                }
            )

        reason = readiness.get("reason")
        if isinstance(reason, str):
            return JSONResponse(
                {
                    "status": "unavailable",
                    "provider_mode": "local",
                    "reason": reason,
                },
                status_code=503,
            )

        raw_ollama = readiness.get("ollama")
        raw_stt = readiness.get("stt")
        raw_tts = readiness.get("tts")
        if not all(
            isinstance(item, dict)
            for item in (raw_ollama, raw_stt, raw_tts)
        ):
            return JSONResponse(
                {
                    "status": "unavailable",
                    "provider_mode": "local",
                    "reason": "PROVIDER_STATUS_UNAVAILABLE",
                },
                status_code=503,
            )
        ollama = cast(dict[str, object], raw_ollama)
        stt = cast(dict[str, object], raw_stt)
        tts = cast(dict[str, object], raw_tts)
        ollama_status: dict[str, object] = {
            "ready": ollama.get("ready") is True,
            "model": ollama.get("model"),
        }
        ollama_reason = ollama.get("reason")
        if isinstance(ollama_reason, str):
            ollama_status["reason"] = ollama_reason
        payload: dict[str, object] = {
            "status": (
                "ready" if readiness.get("ready") is True else "unavailable"
            ),
            "provider_mode": "local",
            "providers": {
                "ollama": ollama_status,
                "stt": {
                    "ready": stt.get("ready") is True,
                    "model": stt.get("model"),
                    "device": "cpu",
                },
                "tts": {
                    "ready": tts.get("ready") is True,
                    "voice": tts.get("voice"),
                },
            },
        }
        return JSONResponse(
            payload,
            status_code=(
                200 if readiness.get("ready") is True else 503
            ),
        )

    return app


app = create_app()
