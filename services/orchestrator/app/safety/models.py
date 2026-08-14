from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class RiskLevel(StrEnum):
    GREEN = "GREEN"
    AMBER = "AMBER"
    RED = "RED"
    EMERGENCY = "EMERGENCY"


class RiskAssessment(BaseModel):
    level: RiskLevel
    reasons: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    evidence_event_ids: list[str] = Field(default_factory=list)


class AgentResponse(BaseModel):
    spoken_text: str
    display_text: str
    support_mode: Literal["listen", "educate", "exercise", "handoff"]
    risk_level: RiskLevel
    evidence_ids: list[str]
    memory_ids: list[str] = Field(default_factory=list)
    visual_observation_ids: list[str]
    action_proposals: list[dict[str, object]]
    memory_candidates: list[dict[str, object]]
    avatar_style: Literal[
        "neutral_listening",
        "warm",
        "concerned_calm",
        "gentle_guidance",
        "handoff_calm",
    ]

