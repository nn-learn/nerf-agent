from enum import StrEnum

from pydantic import BaseModel

from app.safety.models import RiskLevel


class ToolState(StrEnum):
    PROPOSED = "PROPOSED"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    APPROVED = "APPROVED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ToolProposal(BaseModel):
    tool_call_id: str
    name: str
    arguments: dict[str, object]
    risk_level: RiskLevel
    idempotency_key: str
    state: ToolState


class ToolResult(BaseModel):
    tool_call_id: str
    state: ToolState
    output: dict[str, object] | None = None
    error: str | None = None

