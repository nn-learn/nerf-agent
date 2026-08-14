from app.agent.actions import CapabilityProposalGate
from app.agent.control import AgentControlPlane
from app.safety.models import AgentResponse, RiskAssessment, RiskLevel


def _response(actions: list[dict[str, object]]) -> AgentResponse:
    return AgentResponse(
        spoken_text="我可以先生成一个操作建议，由你确认。",
        display_text="我可以先生成一个操作建议，由你确认。",
        support_mode="listen",
        risk_level=RiskLevel.GREEN,
        evidence_ids=[],
        visual_observation_ids=[],
        action_proposals=actions,
        memory_candidates=[],
        avatar_style="warm",
    )


def test_model_cannot_expand_capabilities_beyond_control_directive() -> None:
    directive = AgentControlPlane().draft(
        "最近压力有点大",
        RiskAssessment(level=RiskLevel.GREEN, confidence=1),
    )

    response, audit = CapabilityProposalGate().apply(
        _response(
            [
                {
                    "tool": "delete_user_memory",
                    "arguments": {"memory_id": "memory_1"},
                }
            ]
        ),
        directive,
    )

    assert response.action_proposals == []
    assert audit.blocked_reason_codes == ["CAPABILITY_NOT_IN_DIRECTIVE"]


def test_high_impact_allowed_proposal_is_projected_as_awaiting_user_approval() -> None:
    directive = AgentControlPlane().draft(
        "请删除记忆里关于城市的那一条",
        RiskAssessment(level=RiskLevel.GREEN, confidence=1),
    )

    response, audit = CapabilityProposalGate().apply(
        _response(
            [
                {
                    "tool": "delete_user_memory",
                    "arguments": {"memory_id": "memory_1"},
                    "status": "SUCCEEDED",
                }
            ]
        ),
        directive,
    )

    assert audit.accepted_capabilities == ["delete_user_memory"]
    assert response.action_proposals[0]["status"] == "AWAITING_APPROVAL"
    assert response.action_proposals[0]["requires_approval"] is True
    assert response.action_proposals[0]["approval_role"] == "USER"
