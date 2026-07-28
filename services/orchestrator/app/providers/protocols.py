from typing import Protocol

from app.safety.models import AgentResponse, RiskAssessment


class AgentProvider(Protocol):
    def load_context(
        self, transcript: str, visual_summary: str
    ) -> dict[str, object]: ...

    def plan_reply(
        self,
        transcript: str,
        risk: RiskAssessment,
        context: dict[str, object],
    ) -> AgentResponse: ...

    def plan_crisis(
        self,
        transcript: str,
        risk: RiskAssessment,
    ) -> AgentResponse: ...

