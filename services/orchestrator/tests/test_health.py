import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.settings import Settings


def test_local_settings_have_ollama_runtime_defaults() -> None:
    """Catches local mode defaults drifting from the supported Ollama runtime."""
    settings = Settings(provider_mode="local")

    assert settings.ollama_base_url == "http://127.0.0.1:11434"
    assert settings.text_model == "qwen3.6:latest"
    assert settings.ollama_keep_alive == "30m"
    assert settings.ollama_timeout_seconds == 180
    assert settings.ollama_num_ctx == 4096
    assert settings.ollama_num_predict == 320


@pytest.mark.asyncio
async def test_health_has_no_secret_values() -> None:
    """Catches health endpoints that leak credentials or omit provider readiness."""
    transport = ASGITransport(app=create_app(Settings(provider_mode="mock")))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "provider_mode": "mock",
        "qwen_configured": False,
    }
    assert "api_key" not in response.text.lower()


@pytest.mark.asyncio
async def test_liveness_does_not_depend_on_provider_readiness() -> None:
    """Catches liveness routes that disappear or depend on cloud credentials."""
    transport = ASGITransport(app=create_app(Settings(provider_mode="real")))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "live"}
