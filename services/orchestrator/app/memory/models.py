from enum import StrEnum

from pydantic import BaseModel, Field


class MemoryKind(StrEnum):
    WORKING = "WORKING"
    SEMANTIC = "SEMANTIC"
    EPISODIC = "EPISODIC"
    SAFETY = "SAFETY"


class MemoryDecision(StrEnum):
    ACCEPT = "ACCEPT"
    REQUEST_CONSENT = "REQUEST_CONSENT"
    REJECT = "REJECT"


class MemoryState(StrEnum):
    CANDIDATE = "CANDIDATE"
    AWAITING_CONSENT = "AWAITING_CONSENT"
    ACTIVE = "ACTIVE"
    REVOKED = "REVOKED"
    EXPIRED = "EXPIRED"
    REJECTED = "REJECTED"


class MemoryCandidate(BaseModel):
    source: str = Field(min_length=1)
    contains_sensitive_content: bool
    text: str = Field(min_length=1)
    kind: MemoryKind
    source_turn_id: str = Field(min_length=1)


class MemoryItem(BaseModel):
    memory_id: str
    user_id: str
    candidate: MemoryCandidate
    state: MemoryState
