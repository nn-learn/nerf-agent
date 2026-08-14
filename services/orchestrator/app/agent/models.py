from enum import StrEnum

from pydantic import BaseModel, Field

from app.safety.models import RiskLevel


class AgentIntent(StrEnum):
    GENERAL_SUPPORT = "GENERAL_SUPPORT"
    EMOTIONAL_DISCLOSURE = "EMOTIONAL_DISCLOSURE"
    PSYCHOEDUCATION = "PSYCHOEDUCATION"
    COPING_EXERCISE = "COPING_EXERCISE"
    MEMORY_RECALL = "MEMORY_RECALL"
    MEMORY_CONTROL = "MEMORY_CONTROL"
    CLINICIAN_HANDOFF = "CLINICIAN_HANDOFF"


class EvidenceRequirement(StrEnum):
    NONE = "NONE"
    REVIEWED_KNOWLEDGE = "REVIEWED_KNOWLEDGE"
    CONFIRMED_MEMORY = "CONFIRMED_MEMORY"


class MemoryAccessMode(StrEnum):
    FORBIDDEN = "FORBIDDEN"
    OPTIONAL = "OPTIONAL"
    REQUIRED = "REQUIRED"


class ResponseStrategy(StrEnum):
    MODEL_SUPPORT = "MODEL_SUPPORT"
    MODEL_EDUCATION = "MODEL_EDUCATION"
    MODEL_CAPABILITY_PROPOSAL = "MODEL_CAPABILITY_PROPOSAL"
    DETERMINISTIC_CRISIS = "DETERMINISTIC_CRISIS"
    DETERMINISTIC_ABSTAIN = "DETERMINISTIC_ABSTAIN"


class IntentAssessment(BaseModel):
    intent: AgentIntent
    confidence: float = Field(ge=0, le=1)
    reason_codes: list[str] = Field(default_factory=list)


class AgentDirective(BaseModel):
    policy_version: str = "agent-control-v3.0"
    intent: AgentIntent
    risk_level: RiskLevel
    response_strategy: ResponseStrategy
    evidence_requirement: EvidenceRequirement
    memory_access: MemoryAccessMode
    allowed_capabilities: list[str] = Field(default_factory=list)
    max_tool_proposals: int = Field(default=0, ge=0, le=3)
    reason_codes: list[str] = Field(default_factory=list)


class AgentDecisionTrace(BaseModel):
    """Content-free trace: safe to persist without transcript or retrieved text."""

    policy_version: str
    intent: AgentIntent
    risk_level: RiskLevel
    response_strategy: ResponseStrategy
    evidence_requirement: EvidenceRequirement
    memory_access: MemoryAccessMode
    allowed_capabilities: list[str]
    max_tool_proposals: int
    reviewed_evidence_available: bool
    confirmed_memory_available: bool
    retrieval_degraded: bool
    reason_codes: list[str]
