import asyncio
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast

import httpx
import pytest

from app.contracts.session import CancellationRegistry
from app.main import create_app
from app.providers.faster_whisper import FasterWhisperProvider
from app.providers.local_agent import LocalAgentProvider
from app.providers.mock import MockAgentProvider
from app.providers.runtime import (
    EdgeTtsAudioBridge,
    LazyReviewedRetriever,
    build_runtime_providers,
)
from app.rag.retriever import EvidenceBundle, HybridRetriever
from app.realtime.models import PcmChunk
from app.realtime.turn_coordinator import TurnCoordinator
from app.settings import Settings


class FakeEmbeddingProvider:
    dimensions = 1024

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return [1.0] + [0.0] * 1023


class FakeWhisperModel:
    def transcribe(self, audio, **options):
        _ = (audio, options)
        return [], object()


def fake_whisper_factory(
    model_name: str,
    **options: object,
) -> FakeWhisperModel:
    assert model_name == "small"
    assert options == {"device": "cpu", "compute_type": "int8"}
    return FakeWhisperModel()


class BlockingInitializationRetriever(LazyReviewedRetriever):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.initialization_count = 0
        self.initialization_started = threading.Event()
        self.initialization_release = threading.Event()

    def _initialize(self) -> HybridRetriever:
        self.initialization_count += 1
        self.initialization_started.set()
        if not self.initialization_release.wait(timeout=2):
            raise TimeoutError("test did not release retriever initialization")
        return cast(HybridRetriever, CompletedRetriever())


class CompletedRetriever:
    def retrieve(
        self,
        query: str,
        *,
        risk_level: object,
        k: int,
    ) -> EvidenceBundle:
        _ = (risk_level, k)
        return EvidenceBundle(
            query=query,
            items=[],
            has_sufficient_evidence=False,
        )


def ollama_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/api/version":
        return httpx.Response(200, json={"version": "0.11.4"})
    if request.url.path == "/api/tags":
        return httpx.Response(
            200,
            json={"models": [{"name": "qwen3.6:latest"}]},
        )
    if request.url.path == "/api/generate":
        return httpx.Response(200, json={"done": True})
    raise AssertionError(f"unexpected Ollama path: {request.url.path}")


@pytest.mark.asyncio
async def test_build_runtime_providers_keeps_modes_explicit_and_shares_registry(
    tmp_path: Path,
) -> None:
    """Catches local adapters using unrelated cancellation state or real becoming local."""
    registry = CancellationRegistry()
    embedding = FakeEmbeddingProvider()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(ollama_handler)
    ) as client:
        mock = build_runtime_providers(
            Settings(provider_mode="mock"),
            registry=registry,
            ollama_client=client,
        )
        local = build_runtime_providers(
            Settings(
                provider_mode="local",
                knowledge_manifest_path=tmp_path / "manifest.yaml",
            ),
            registry=registry,
            ollama_client=client,
            embedding_provider=embedding,
            whisper_model_factory=fake_whisper_factory,
        )
        real = build_runtime_providers(
            Settings(provider_mode="real"),
            registry=registry,
            ollama_client=client,
        )

        assert mock.transcriber is None
        assert isinstance(mock.agent_provider, MockAgentProvider)
        assert mock.provider_mode == "mock"

        assert isinstance(local.transcriber, FasterWhisperProvider)
        assert isinstance(local.agent_provider, LocalAgentProvider)
        assert isinstance(local.audio_bridge, EdgeTtsAudioBridge)
        assert local.provider_mode == "local"
        assert local.ollama is not None
        assert local.ollama._registry is registry
        assert local.transcriber._registry is registry
        assert local.audio_bridge._provider._registry is registry

        assert real.provider_mode == "real"
        assert real.transcriber is None
        assert real.ollama is None
        assert isinstance(real.agent_provider, MockAgentProvider)
        assert await real.readiness() == {
            "ready": True,
            "provider_mode": "real",
        }


@pytest.mark.asyncio
async def test_lazy_retriever_initializes_and_indexes_once_under_concurrency(
    tmp_path: Path,
) -> None:
    """Catches simultaneous first turns constructing or indexing BGE more than once."""
    source = tmp_path / "support.md"
    source.write_text("先识别当前最困扰的一件事。", encoding="utf-8")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "\n".join(
            [
                "documents:",
                "  - document_id: stress-1",
                "    title: 压力支持",
                "    version: 1.0.0",
                "    reviewer: reviewer-a",
                "    reviewed_at: 2025-01-01",
                "    expires_at: 2099-01-01",
                "    audience: [adult]",
                "    allowed_risk_levels: [GREEN]",
                "    source: support.md",
            ]
        ),
        encoding="utf-8",
    )
    embedding = FakeEmbeddingProvider()
    retriever = LazyReviewedRetriever(
        Settings(
            provider_mode="local",
            knowledge_manifest_path=manifest,
            rag_evidence_threshold=0.62,
        ),
        embedding_provider=embedding,
    )

    first, second = await asyncio.gather(
        retriever.retrieve("压力", risk_level="GREEN", k=1),
        retriever.retrieve("睡眠", risk_level="GREEN", k=1),
    )

    assert first.query == "压力"
    assert second.query == "睡眠"
    assert len(embedding.calls) == 3


@pytest.mark.asyncio
async def test_lazy_retriever_keeps_one_initialization_when_first_waiter_cancels(
    tmp_path: Path,
) -> None:
    """Catches caller cancellation starting a duplicate BGE construction."""
    retriever = BlockingInitializationRetriever(
        Settings(
            provider_mode="local",
            knowledge_manifest_path=tmp_path / "manifest.yaml",
        )
    )
    first = asyncio.create_task(
        retriever.retrieve("first", risk_level="GREEN", k=1)
    )
    started = await asyncio.to_thread(
        retriever.initialization_started.wait,
        1,
    )
    assert started

    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    second = asyncio.create_task(
        retriever.retrieve("second", risk_level="GREEN", k=1)
    )
    release = asyncio.get_running_loop().call_later(
        0.1,
        retriever.initialization_release.set,
    )
    try:
        result = await second
    finally:
        release.cancel()
        retriever.initialization_release.set()

    assert result.query == "second"
    assert retriever.initialization_count == 1


@pytest.mark.asyncio
async def test_runtime_readiness_tracks_prewarm_and_closes_ollama() -> None:
    """Catches liveness blocking on warmup or shutdown abandoning its owned client."""
    registry = CancellationRegistry()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/generate":
            entered.set()
            await release.wait()
            return httpx.Response(200, json={"done": True})
        return ollama_handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    providers = build_runtime_providers(
        Settings(provider_mode="local"),
        registry=registry,
        ollama_client=client,
        embedding_provider=FakeEmbeddingProvider(),
        whisper_model_factory=fake_whisper_factory,
    )
    providers.start_prewarm()
    await asyncio.wait_for(entered.wait(), timeout=1)

    assert await providers.readiness() == {
        "ready": False,
        "provider_mode": "local",
        "reason": "OLLAMA_WARMING",
    }

    release.set()
    assert providers.prewarm_task is not None
    await providers.prewarm_task
    readiness = await providers.readiness()
    assert readiness == {
        "ready": True,
        "provider_mode": "local",
        "ollama": {
            "ready": True,
            "model": "qwen3.6:latest",
            "reason": None,
        },
        "stt": {"ready": True, "model": "small"},
        "tts": {"ready": True, "voice": "zh-CN-XiaoxiaoNeural"},
    }

    await providers.aclose()
    await client.aclose()


@pytest.mark.asyncio
async def test_edge_tts_bridge_streams_provider_chunks() -> None:
    """Catches runtime assembly buffering Edge TTS instead of exposing its stream."""
    registry = CancellationRegistry()
    turn_id = "turn_1"
    token = await registry.issue(turn_id)

    async def pcm_source(text: str, voice: str) -> AsyncIterator[bytes]:
        assert text == "慢慢说"
        assert voice == "zh-CN-XiaoxiaoNeural"
        yield b"\x00\x00" * 320

    from app.providers.edge_tts import EdgeTtsProvider
    from app.realtime.models import TurnHandle

    provider = EdgeTtsProvider(registry=registry, pcm_source=pcm_source)
    bridge = EdgeTtsAudioBridge(provider)
    chunks = [
        chunk
        async for chunk in bridge.synthesize(
            "慢慢说",
            turn=TurnHandle(
                session_id="session_1",
                turn_id=turn_id,
                cancel_token=token,
            ),
        )
    ]

    assert chunks == [
        PcmChunk(
            sequence=0,
            pts_ms=0,
            pcm_s16le=b"\x00\x00" * 320,
        )
    ]


@pytest.mark.asyncio
async def test_create_app_rejects_injected_runtime_without_coordinator(
    tmp_path: Path,
) -> None:
    """Catches an injected bundle silently using a new cancellation registry."""
    client = httpx.AsyncClient(transport=httpx.MockTransport(ollama_handler))
    providers = build_runtime_providers(
        Settings(provider_mode="local"),
        registry=CancellationRegistry(),
        ollama_client=client,
        embedding_provider=FakeEmbeddingProvider(),
        whisper_model_factory=fake_whisper_factory,
    )
    try:
        with pytest.raises(
            ValueError,
            match="coordinator is required with injected runtime providers",
        ):
            create_app(
                Settings(
                    provider_mode="local",
                    event_database_path=tmp_path / "events.sqlite3",
                ),
                runtime_providers=providers,
            )
    finally:
        await providers.aclose()
        await client.aclose()


@pytest.mark.asyncio
async def test_create_app_validates_and_shares_injected_registry(
    tmp_path: Path,
) -> None:
    """Catches a supplied coordinator and providers using different tokens."""
    coordinator = TurnCoordinator()
    client = httpx.AsyncClient(transport=httpx.MockTransport(ollama_handler))
    providers = build_runtime_providers(
        Settings(provider_mode="local"),
        registry=coordinator.tokens,
        ollama_client=client,
        embedding_provider=FakeEmbeddingProvider(),
        whisper_model_factory=fake_whisper_factory,
    )
    try:
        with pytest.raises(
            ValueError,
            match="runtime provider registry must be coordinator.tokens",
        ):
            create_app(
                Settings(
                    provider_mode="local",
                    event_database_path=tmp_path / "bad-events.sqlite3",
                ),
                runtime_providers=providers,
                coordinator=TurnCoordinator(),
            )

        app = create_app(
            Settings(
                provider_mode="local",
                event_database_path=tmp_path / "events.sqlite3",
            ),
            runtime_providers=providers,
            coordinator=coordinator,
        )
        assert providers.registry is coordinator.tokens
        assert app.state.session_manager._coordinator is coordinator
    finally:
        await providers.aclose()
        await client.aclose()
