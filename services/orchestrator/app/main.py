from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.clinician import create_clinician_router
from app.api.consents import create_consent_router
from app.api.livekit import create_livekit_router
from app.api.sessions import create_sessions_router
from app.events.store import EventStore
from app.realtime.session import SessionManager
from app.security.middleware import (
    InMemoryRateLimitMiddleware,
    SecurityHeadersMiddleware,
)
from app.settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    current = settings or Settings()
    app = FastAPI(title="PsyAvatar Care Orchestrator")
    event_store = EventStore(current.event_database_path)
    session_manager = SessionManager(store=event_store, provider_mode="mock")
    app.state.event_store = event_store
    app.state.session_manager = session_manager
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
    app.include_router(create_sessions_router(session_manager))
    app.include_router(create_consent_router(event_store, session_manager))
    app.include_router(create_clinician_router(event_store))

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "live"}

    @app.get("/health/ready")
    async def ready() -> dict[str, str | bool]:
        return {
            "status": "ready",
            "provider_mode": current.provider_mode,
            "qwen_configured": bool(current.dashscope_api_key),
        }

    return app


app = create_app()
