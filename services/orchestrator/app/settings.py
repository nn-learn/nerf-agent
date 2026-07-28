from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PSYAVATAR_", env_file=".env")

    provider_mode: Literal["mock", "real"] = "mock"
    dashscope_api_key: str | None = None
    text_model: str = "qwen-max"
    vision_model: str = "qwen3.6-flash"
    tts_voice: str = "zh-CN-XiaoxiaoNeural"
    stt_model: str = "small"

