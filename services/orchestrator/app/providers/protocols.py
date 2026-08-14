from typing import Protocol

from pydantic import BaseModel, Field

from app.safety.models import AgentResponse, RiskAssessment


class AgentPlan(BaseModel):
    response: AgentResponse
    provider_metrics: dict[str, int | float | str] = Field(
        default_factory=dict
    )


class AgentProvider(Protocol):
    async def load_context(
        self,
        transcript: str,
        visual_summary: str,
        risk: RiskAssessment,
        *,
        reviewed_evidence_required: bool,
    ) -> dict[str, object]:
        raise NotImplementedError

    async def plan_reply(
        self,
        transcript: str,
        risk: RiskAssessment,
        context: dict[str, object],
        *,
        turn_id: str,
        cancel_token: str,
    ) -> AgentPlan:
        raise NotImplementedError

    async def plan_crisis(
        self,
        transcript: str,
        risk: RiskAssessment,
    ) -> AgentPlan:
        raise NotImplementedError

