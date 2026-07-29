from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TranscriptKind(StrEnum):
    PARTIAL = "PARTIAL"
    FINAL = "FINAL"


class TranscriptEvent(BaseModel):
    kind: TranscriptKind
    text: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)


class PcmChunk(BaseModel):
    sequence: int = Field(ge=0)
    pts_ms: int = Field(ge=0)
    pcm_s16le: bytes
    sample_rate: Literal[16000] = 16000
    channels: Literal[1] = 1
    sample_width_bytes: Literal[2] = 2
    duration_ms: Literal[20] = 20

    @field_validator("pcm_s16le")
    @classmethod
    def require_exact_twenty_milliseconds(cls, value: bytes) -> bytes:
        expected = 16_000 * 1 * 2 * 20 // 1_000
        if len(value) != expected:
            raise ValueError(f"PCM chunk must contain exactly {expected} bytes")
        return value


class TurnHandle(BaseModel):
    session_id: str
    turn_id: str
    cancel_token: str


class InterruptionResult(BaseModel):
    session_id: str
    turn_id: str
    cancelled_token: str
    reason: str
