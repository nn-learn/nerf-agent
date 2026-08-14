import pytest

from app.agent.control import AgentControlPlane, AgentPolicyViolation
from app.agent.models import (
    AgentIntent,
    EvidenceRequirement,
    MemoryAccessMode,
    ResponseStrategy,
)
from app.safety.models import AgentResponse, RiskAssessment, RiskLevel


def _risk(level: RiskLevel) -> RiskAssessment:
    return RiskAssessment(level=level, confidence=1.0)


def test_risk_override_forbids_context_and_normal_model_path() -> None:
    directive = AgentControlPlane().draft(
        "你还记得我以前说过什么吗？",
        _risk(RiskLevel.RED),
    )

    assert directive.response_strategy is ResponseStrategy.DETERMINISTIC_CRISIS
    assert directive.memory_access is MemoryAccessMode.FORBIDDEN
    assert directive.evidence_requirement is EvidenceRequirement.NONE
    assert "RISK_OVERRIDE" in directive.reason_codes


def test_memory_control_is_not_confused_with_memory_recall() -> None:
    directive = AgentControlPlane().draft(
        "请删除记忆里关于城市的那条内容",
        _risk(RiskLevel.GREEN),
    )

    assert directive.intent is AgentIntent.MEMORY_CONTROL
    assert directive.memory_access is MemoryAccessMode.FORBIDDEN
    assert directive.allowed_capabilities == [
        "export_user_memories",
        "delete_user_memory",
    ]


def test_missing_required_memory_fails_closed_without_raw_text_in_trace() -> None:
    control = AgentControlPlane()
    directive = control.draft(
        "你记得我之前说过什么吗？",
        _risk(RiskLevel.GREEN),
    )

    final, trace = control.finalize(
        directive,
        {
            "long_term_memory": [],
            "memory_retrieval_degraded": False,
        },
    )

    assert final.response_strategy is ResponseStrategy.DETERMINISTIC_ABSTAIN
    assert "REQUIRED_CONTEXT_UNAVAILABLE" in trace.reason_codes
    assert "你记得" not in trace.model_dump_json()


def test_response_cannot_change_risk_or_exceed_action_budget() -> None:
    control = AgentControlPlane()
    directive = control.draft("最近压力很大", _risk(RiskLevel.AMBER))
    wrong_risk = AgentResponse(
        spoken_text="我们慢慢聊。",
        display_text="我们慢慢聊。",
        support_mode="listen",
        risk_level=RiskLevel.GREEN,
        evidence_ids=[],
        visual_observation_ids=[],
        action_proposals=[],
        memory_candidates=[],
        avatar_style="warm",
    )

    with pytest.raises(AgentPolicyViolation, match="risk level"):
        control.validate_response(wrong_risk, directive)
