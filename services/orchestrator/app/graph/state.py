import operator
from typing import Annotated, TypedDict

from app.agent.actions import CapabilityProposalAudit
from app.agent.avatar import AvatarResponsePlan
from app.agent.care import CareLoopState, CareLoopTrace
from app.agent.evidence import EvidenceContextAudit, EvidenceResponseAudit
from app.agent.models import AgentDecisionTrace, AgentDirective, IntentAssessment
from app.safety.models import AgentResponse, RiskAssessment


class AgentState(TypedDict, total=False):
    session_id: str
    transcript: str
    visual_summary: str
    turn_id: str
    cancel_token: str
    visited: Annotated[list[str], operator.add]
    risk: RiskAssessment
    intent: IntentAssessment
    agent_directive: AgentDirective
    agent_trace: AgentDecisionTrace
    evidence_context_audit: EvidenceContextAudit
    evidence_response_audit: EvidenceResponseAudit
    capability_audit: CapabilityProposalAudit
    avatar_plan: AvatarResponsePlan
    care_loop_state: CareLoopState
    care_loop_trace: CareLoopTrace
    long_term_memory: list[dict[str, object]]
    context: dict[str, object]
    candidate_response: AgentResponse
    provider_metrics: dict[str, int | float | str]
    response: AgentResponse

