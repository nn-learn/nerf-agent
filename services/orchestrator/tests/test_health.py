import asyncio

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.providers.runtime import build_runtime_providers
from app.realtime.turn_coordinator import TurnCoordinator
from app.settings import Settings


class FakeWhisperModel:
    def transcribe(self, audio, **options):
        _ = (audio, options)
        return [], object()


def fake_whisper_factory(model_name: str, **options: object) -> FakeWhisperModel:
    _ = (model_name, options)
    return FakeWhisperModel()


class FakeEmbeddingProvider:
    dimensions = 1024

    def embed(self, text: str) -> list[float]:
        _ = text
        return [1.0] + [0.0] * 1023


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


@pytest.mark.asyncio
async def test_local_ready_health_reports_exact_provider_status(tmp_path) -> None:
    """Catches configured local adapters being hidden behind static health output."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={"done": True})
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.11.4"})
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={"models": [{"name": "qwen3.6:latest"}]},
            )
        raise AssertionError(request.url.path)

    settings = Settings(
        provider_mode="local",
        event_database_path=tmp_path / "events.sqlite3",
    )
    coordinator = TurnCoordinator()
    async with AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as ollama_client:
        providers = build_runtime_providers(
            settings,
            registry=coordinator.tokens,
            ollama_client=ollama_client,
            embedding_provider=FakeEmbeddingProvider(),
            whisper_model_factory=fake_whisper_factory,
        )
        app = create_app(
            settings,
            runtime_providers=providers,
            coordinator=coordinator,
        )
        async with app.router.lifespan_context(app):
            assert providers.prewarm_task is not None
            await providers.prewarm_task
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport,
                base_url="http://test",
            ) as client:
                response = await client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "provider_mode": "local",
        "providers": {
            "ollama": {"ready": True, "model": "qwen3.6:latest"},
            "stt": {"ready": True, "model": "small", "device": "cpu"},
            "tts": {"ready": True, "voice": "zh-CN-XiaoxiaoNeural"},
        },
    }


@pytest.mark.asyncio
async def test_local_ready_health_is_503_while_prewarming(tmp_path) -> None:
    """Catches readiness claiming local service before model prewarm completes."""
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/generate":
            entered.set()
            await release.wait()
            return httpx.Response(200, json={"done": True})
        raise AssertionError(request.url.path)

    settings = Settings(
        provider_mode="local",
        event_database_path=tmp_path / "events.sqlite3",
    )
    coordinator = TurnCoordinator()
    async with AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as ollama_client:
        providers = build_runtime_providers(
            settings,
            registry=coordinator.tokens,
            ollama_client=ollama_client,
            embedding_provider=FakeEmbeddingProvider(),
            whisper_model_factory=fake_whisper_factory,
        )
        app = create_app(
            settings,
            runtime_providers=providers,
            coordinator=coordinator,
        )
        async with app.router.lifespan_context(app):
            await asyncio.wait_for(entered.wait(), timeout=1)
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport,
                base_url="http://test",
            ) as client:
                response = await client.get("/health/ready")
            release.set()
            assert providers.prewarm_task is not None
            await providers.prewarm_task

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "provider_mode": "local",
        "reason": "OLLAMA_WARMING",
    }
