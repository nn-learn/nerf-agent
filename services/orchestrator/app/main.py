from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.clinician import create_clinician_router
from app.api.consents import create_consent_router
from app.api.livekit import create_livekit_router
from app.api.realtime import create_realtime_router
from app.api.sessions import create_sessions_router
from app.events.store import EventStore
from app.providers.runtime import RuntimeProviders, build_runtime_providers
from app.realtime.session import SessionManager
from app.realtime.turn_coordinator import TurnCoordinator
from app.security.middleware import (
    InMemoryRateLimitMiddleware,
    SecurityHeadersMiddleware,
)
from app.settings import Settings


def create_app(
    settings: Settings | None = None,
    *,
    runtime_providers: RuntimeProviders | None = None,
    coordinator: TurnCoordinator | None = None,
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
        try:
            yield
        finally:
            if owns_providers:
                await providers.aclose()

    app = FastAPI(
        title="PsyAvatar Care Orchestrator",
        lifespan=lifespan,
    )
    event_store = EventStore(current.event_database_path)
    session_manager = SessionManager(
        store=event_store,
        coordinator=active_coordinator,
        agent_provider=providers.agent_provider,
        audio_bridge=providers.audio_bridge,
        provider_mode=providers.provider_mode,
    )
    app.state.event_store = event_store
    app.state.session_manager = session_manager
    app.state.runtime_providers = providers
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[current.web_origin],
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
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
    app.include_router(create_sessions_router(session_manager))
    app.include_router(create_consent_router(event_store, session_manager))
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
