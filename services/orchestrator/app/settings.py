from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderMode = Literal["mock", "local", "real"]
MemoryExtractorMode = Literal["rule", "ollama"]
MemoryClaimNormalizerMode = Literal["rule", "ollama"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PSYAVATAR_", env_file=".env")

    provider_mode: ProviderMode = "mock"
    ollama_base_url: str = "http://127.0.0.1:11434"
    text_model: str = "qwen3.6:latest"
    ollama_keep_alive: str = "30m"
    ollama_timeout_seconds: float = Field(default=180, gt=0, le=600)
    ollama_num_ctx: int = Field(default=4096, ge=1024, le=32768)
    ollama_num_predict: int = Field(default=320, ge=32, le=2048)
    memory_extractor_mode: MemoryExtractorMode = "rule"
    memory_ollama_num_ctx: int = Field(default=8192, ge=1024, le=32768)
    memory_ollama_num_predict: int = Field(default=1200, ge=64, le=4096)
    memory_extraction_cache_entries: int = Field(default=256, ge=0, le=10_000)
    memory_rule_fallback: bool = True
    memory_extraction_batch_tokens: int = Field(default=4096, ge=512, le=32768)
    memory_extraction_batch_windows: int = Field(default=8, ge=1, le=32)
    memory_ingestion_queue_size: int = Field(default=128, ge=1, le=10_000)
    memory_claim_normalizer_mode: MemoryClaimNormalizerMode = "rule"
    memory_claim_ollama_timeout_seconds: float = Field(default=60, gt=0, le=300)
    memory_claim_ollama_num_ctx: int = Field(default=4096, ge=1024, le=16384)
    memory_claim_ollama_num_predict: int = Field(default=800, ge=64, le=2048)
    memory_claim_ollama_cache_entries: int = Field(default=256, ge=0, le=10_000)
    memory_claim_ollama_batch_claims: int = Field(default=32, ge=2, le=64)
    memory_episode_summary_enabled: bool = True
    memory_episode_max_members: int = Field(default=6, ge=2, le=32)
    memory_episode_max_chars: int = Field(default=800, ge=200, le=4000)
    memory_episode_time_gap_minutes: int = Field(default=30, ge=5, le=1440)
    memory_freshness_rerank_enabled: bool = True
    memory_freshness_minimum_factor: float = Field(default=0.90, gt=0, le=1)
    memory_shadow_enabled: bool = False
    memory_shadow_policy_version: str = "memory-shadow-research-v1"
    memory_shadow_strategy_version: str = "hybrid-bge-m3-v1.6@0.50"
    memory_shadow_semantic_min_relevance: float = Field(default=0.50, ge=0, le=1)
    memory_shadow_timeout_seconds: float = Field(default=15, gt=0, le=120)
    memory_shadow_initialization_timeout_seconds: float = Field(
        default=120,
        gt=0,
        le=300,
    )
    memory_shadow_failure_threshold: int = Field(default=3, ge=1, le=100)
    memory_shadow_cooldown_seconds: float = Field(default=60, gt=0, le=3600)
    memory_shadow_max_concurrency: int = Field(default=1, ge=1, le=8)
    memory_shadow_max_pending: int = Field(default=16, ge=1, le=256)
    memory_shadow_retention_days: int = Field(default=30, ge=1, le=90)
    memory_shadow_min_reliable_runs: int = Field(default=100, ge=30, le=100_000)
    knowledge_manifest_path: Path = Path("../../knowledge/manifest.yaml")
    rag_top_k: int = Field(default=3, ge=1, le=8)
    rag_evidence_threshold: float = Field(default=0.62, ge=0, le=1)
    websocket_auth_timeout_seconds: float = Field(default=5, gt=0, le=30)
    utterance_end_silence_ms: int = Field(default=600, ge=200, le=2000)
    utterance_max_ms: int = Field(default=20_000, ge=1000, le=60_000)
    dashscope_api_key: SecretStr | None = None
    vision_model: str = "qwen3.6-flash"
    tts_voice: str = "zh-CN-XiaoxiaoNeural"
    stt_model: str = "small"
    livekit_url: str | None = None
    livekit_api_key: str | None = None
    livekit_api_secret: SecretStr | None = None
    web_origin: str = "http://localhost:5173"
    event_database_path: Path = Path("runtime/events.sqlite3")
    request_rate_limit_per_minute: int = Field(default=120, ge=1, le=10_000)
