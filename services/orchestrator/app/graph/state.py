import operator
from typing import Annotated, TypedDict

from app.safety.models import AgentResponse, RiskAssessment


class AgentState(TypedDict, total=False):
    session_id: str
    transcript: str
    visual_summary: str
    turn_id: str
    cancel_token: str
    visited: Annotated[list[str], operator.add]
    risk: RiskAssessment
    long_term_memory: list[dict[str, object]]
    context: dict[str, object]
    candidate_response: AgentResponse
    provider_metrics: dict[str, int | float | str]
    response: AgentResponse

