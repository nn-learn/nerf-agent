from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderMode = Literal["mock", "local", "real"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PSYAVATAR_", env_file=".env")

    provider_mode: ProviderMode = "mock"
    ollama_base_url: str = "http://127.0.0.1:11434"
    text_model: str = "qwen3.6:latest"
    ollama_keep_alive: str = "30m"
    ollama_timeout_seconds: float = Field(default=180, gt=0, le=600)
    ollama_num_ctx: int = Field(default=4096, ge=1024, le=32768)
    ollama_num_predict: int = Field(default=320, ge=32, le=2048)
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
