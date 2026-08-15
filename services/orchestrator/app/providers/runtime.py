import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass
from datetime import date

import httpx

from app.contracts.session import CancellationRegistry
from app.providers.edge_tts import EdgeTtsProvider
from app.providers.faster_whisper import (
    FasterWhisperProvider,
    WhisperModelFactory,
)
from app.providers.local_agent import LocalAgentProvider
from app.providers.mock import MockAgentProvider
from app.providers.ollama_text import OllamaTextProvider
from app.providers.protocols import AgentProvider
from app.providers.qwen_text import TextProviderResponseError
from app.rag.bge import BgeM3EmbeddingProvider
from app.rag.ingest import load_manifest
from app.rag.retriever import (
    EmbeddingProvider,
    EvidenceBundle,
    HybridRetriever,
)
from app.realtime.models import PcmChunk, TurnHandle
from app.realtime.session import AudioBridge, AudioUnavailable, MockAudioBridge
from app.safety.models import RiskLevel
from app.settings import Settings


class EdgeTtsAudioBridge:
    def __init__(self, provider: EdgeTtsProvider) -> None:
        self._provider = provider

    async def synthesize(
        self,
        text: str,
        *,
        turn: TurnHandle,
        speech_rate: float = 1.0,
    ) -> AsyncIterator[PcmChunk]:
        try:
            async for chunk in self._provider.synthesize(
                text,
                turn_id=turn.turn_id,
                cancel_token=turn.cancel_token,
                start_pts_ms=0,
                speech_rate=speech_rate,
            ):
                yield chunk
        except (RuntimeError, OSError) as error:
            raise AudioUnavailable(
                "local TTS provider is unavailable"
            ) from error


class LazyReviewedRetriever:
    def __init__(
        self,
        settings: Settings,
        *,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self._settings = settings
        self._embedding_provider = embedding_provider
        self._retriever: HybridRetriever | None = None
        self._initialization_task: asyncio.Task[HybridRetriever] | None = None
        self._initialization_lock = asyncio.Lock()

    async def retrieve(
        self,
        query: str,
        *,
        risk_level: RiskLevel,
        k: int,
    ) -> EvidenceBundle:
        retriever = await self._get_retriever()
        return await asyncio.to_thread(
            retriever.retrieve,
            query,
            risk_level=risk_level,
            k=k,
        )

    async def _get_retriever(self) -> HybridRetriever:
        if self._retriever is not None:
            return self._retriever
        async with self._initialization_lock:
            if self._retriever is not None:
                return self._retriever
            if self._initialization_task is None:
                self._initialization_task = asyncio.create_task(
                    asyncio.to_thread(self._initialize)
                )
                self._initialization_task.add_done_callback(
                    self._consume_initialization_exception
                )
            initialization_task = self._initialization_task

        try:
            retriever = await asyncio.shield(initialization_task)
        except asyncio.CancelledError:
            raise
        except Exception:
            async with self._initialization_lock:
                if self._initialization_task is initialization_task:
                    self._initialization_task = None
            raise

        async with self._initialization_lock:
            if self._retriever is None:
                self._retriever = retriever
            return self._retriever

    @staticmethod
    def _consume_initialization_exception(
        task: asyncio.Task[HybridRetriever],
    ) -> None:
        if not task.cancelled():
            task.exception()

    def _initialize(self) -> HybridRetriever:
        documents = load_manifest(self._settings.knowledge_manifest_path)
        retriever = HybridRetriever(
            embedding_provider=(
                self._embedding_provider or BgeM3EmbeddingProvider()
            ),
            evidence_threshold=self._settings.rag_evidence_threshold,
        )
        retriever.index(documents, as_of=date.today())
        return retriever


@dataclass(slots=True)
class RuntimeProviders:
    provider_mode: str
    registry: CancellationRegistry
    agent_provider: AgentProvider
    transcriber: FasterWhisperProvider | None
    audio_bridge: AudioBridge
    ollama: OllamaTextProvider | None
    tts_voice: str
    prewarm_task: asyncio.Task[None] | None = None

    def start_prewarm(self) -> None:
        if self.ollama is not None and self.prewarm_task is None:
            self.prewarm_task = asyncio.create_task(self.ollama.prewarm())

    async def readiness(self) -> dict[str, object]:
        if self.provider_mode != "local" or self.ollama is None:
            return {
                "ready": True,
                "provider_mode": self.provider_mode,
            }
        if self.prewarm_task is not None and not self.prewarm_task.done():
            return {
                "ready": False,
                "provider_mode": "local",
                "reason": "OLLAMA_WARMING",
            }
        if (
            self.prewarm_task is not None
            and not self.prewarm_task.cancelled()
            and self.prewarm_task.exception() is not None
        ):
            return {
                "ready": False,
                "provider_mode": "local",
                "reason": "OLLAMA_PREWARM_FAILED",
            }
        try:
            result = await self.ollama.readiness()
        except TextProviderResponseError:
            return {
                "ready": False,
                "provider_mode": "local",
                "reason": "OLLAMA_UNAVAILABLE",
            }
        return {
            "ready": result.ready,
            "provider_mode": "local",
            "ollama": {
                "ready": result.ready,
                "model": result.model,
                "reason": result.reason,
            },
            "stt": {"ready": self.transcriber is not None, "model": "small"},
            "tts": {"ready": True, "voice": self.tts_voice},
        }

    async def aclose(self) -> None:
        if self.prewarm_task is not None and not self.prewarm_task.done():
            self.prewarm_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.prewarm_task
        if self.ollama is not None:
            await self.ollama.aclose()


def build_runtime_providers(
    settings: Settings,
    *,
    registry: CancellationRegistry,
    ollama_client: httpx.AsyncClient | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    whisper_model_factory: WhisperModelFactory | None = None,
) -> RuntimeProviders:
    if settings.provider_mode != "local":
        return RuntimeProviders(
            provider_mode=settings.provider_mode,
            registry=registry,
            agent_provider=MockAgentProvider(),
            transcriber=None,
            audio_bridge=MockAudioBridge(),
            ollama=None,
            tts_voice=settings.tts_voice,
        )

    ollama = OllamaTextProvider(
        registry=registry,
        model=settings.text_model,
        base_url=settings.ollama_base_url,
        keep_alive=settings.ollama_keep_alive,
        timeout_seconds=settings.ollama_timeout_seconds,
        num_ctx=settings.ollama_num_ctx,
        num_predict=settings.ollama_num_predict,
        client=ollama_client,
    )
    retriever = LazyReviewedRetriever(
        settings,
        embedding_provider=embedding_provider,
    )
    transcriber = (
        FasterWhisperProvider(registry=registry)
        if whisper_model_factory is None
        else FasterWhisperProvider(
            registry=registry,
            model_factory=whisper_model_factory,
        )
    )
    edge_tts = EdgeTtsProvider(
        registry=registry,
        voice=settings.tts_voice,
    )
    return RuntimeProviders(
        provider_mode="local",
        registry=registry,
        agent_provider=LocalAgentProvider(
            text_provider=ollama,
            retriever=retriever,
            top_k=settings.rag_top_k,
        ),
        transcriber=transcriber,
        audio_bridge=EdgeTtsAudioBridge(edge_tts),
        ollama=ollama,
        tts_voice=settings.tts_voice,
    )
