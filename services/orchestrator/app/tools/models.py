from enum import StrEnum

from pydantic import BaseModel, Field

from app.safety.models import RiskLevel


class ToolState(StrEnum):
    PROPOSED = "PROPOSED"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    APPROVED = "APPROVED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class CapabilityOrigin(StrEnum):
    LOCAL = "LOCAL"
    MCP = "MCP"


class CapabilityImpact(StrEnum):
    READ_ONLY = "READ_ONLY"
    LOCAL_REVERSIBLE = "LOCAL_REVERSIBLE"
    SENSITIVE_DATA_EXPORT = "SENSITIVE_DATA_EXPORT"
    DATA_DELETION = "DATA_DELETION"
    EXTERNAL_COMMUNICATION = "EXTERNAL_COMMUNICATION"
    CLINICAL_ESCALATION = "CLINICAL_ESCALATION"


class ApprovalRole(StrEnum):
    NONE = "NONE"
    USER = "USER"
    CLINICIAN = "CLINICIAN"


class CapabilityDefinition(BaseModel):
    name: str = Field(min_length=1)
    origin: CapabilityOrigin
    source_id: str = Field(min_length=1)
    trusted: bool
    impact: CapabilityImpact
    approval_role: ApprovalRole
    allowed_risk_levels: list[RiskLevel] = Field(min_length=1)
    max_per_turn: int = Field(default=1, ge=1, le=3)
    timeout_ms: int = Field(default=5_000, ge=100, le=30_000)


class ToolProposal(BaseModel):
    tool_call_id: str
    name: str
    arguments: dict[str, object]
    risk_level: RiskLevel
    turn_id: str
    capability_origin: CapabilityOrigin
    capability_impact: CapabilityImpact
    approval_role: ApprovalRole
    approved_by_role: ApprovalRole | None = None
    argument_digest: str
    idempotency_key: str
    state: ToolState


class ToolResult(BaseModel):
    tool_call_id: str
    state: ToolState
    output: dict[str, object] | None = None
    error: str | None = None

