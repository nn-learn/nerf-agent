from fastapi import FastAPI

from app.settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    current = settings or Settings()
    app = FastAPI(title="PsyAvatar Care Orchestrator")

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
