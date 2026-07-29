from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PSYAVATAR_", env_file=".env")

    provider_mode: Literal["mock", "real"] = "mock"
    dashscope_api_key: SecretStr | None = None
    text_model: str = "qwen-max"
    vision_model: str = "qwen3.6-flash"
    tts_voice: str = "zh-CN-XiaoxiaoNeural"
    stt_model: str = "small"
    livekit_url: str | None = None
    livekit_api_key: str | None = None
    livekit_api_secret: SecretStr | None = None
    web_origin: str = "http://localhost:5173"
    event_database_path: Path = Path("runtime/events.sqlite3")
    request_rate_limit_per_minute: int = Field(default=120, ge=1, le=10_000)
