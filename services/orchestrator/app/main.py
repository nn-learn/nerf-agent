from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.consents import create_consent_router
from app.api.livekit import create_livekit_router
from app.events.store import EventStore
from app.settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    current = settings or Settings()
    app = FastAPI(title="PsyAvatar Care Orchestrator")
    event_store = EventStore(current.event_database_path)
    app.state.event_store = event_store
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[current.web_origin],
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "Authorization"],
    )
    app.include_router(create_livekit_router(current))
    app.include_router(create_consent_router(event_store))

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
