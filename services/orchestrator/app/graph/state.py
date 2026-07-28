import operator
from typing import Annotated, TypedDict

from app.safety.models import AgentResponse, RiskAssessment


class AgentState(TypedDict, total=False):
    transcript: str
    visual_summary: str
    visited: Annotated[list[str], operator.add]
    risk: RiskAssessment
    context: dict[str, object]
    candidate_response: AgentResponse
    response: AgentResponse

