import pytest

from app.safety.models import AgentResponse, RiskLevel
from app.safety.output_guard import OutputGuard, UnsafeResponseError
from app.safety.rules import assess_risk


def test_visual_appearance_cannot_raise_risk() -> None:
    """Catches risk logic that treats posture or expression as clinical evidence."""
    result = assess_risk(
        transcript="我只是有点累",
        visual_summary="用户低头且没有微笑",
    )

    assert result.level is RiskLevel.GREEN
    assert "visual_only" not in result.reasons


def test_immediate_self_harm_language_routes_to_emergency() -> None:
    """Catches deterministic emergency language being left to a probabilistic model."""
    result = assess_risk(
        transcript="我现在正在伤害自己",
        visual_summary="",
    )

    assert result.level is RiskLevel.EMERGENCY
    assert result.confidence == 1.0


def test_explicit_ideation_routes_to_red_without_immediate_action() -> None:
    """Catches explicit ideation being collapsed into ordinary low mood."""
    result = assess_risk(
        transcript="我最近有自杀的想法，但现在没有在做",
        visual_summary="",
    )

    assert result.level is RiskLevel.RED


def test_output_guard_rejects_diagnosis_from_visual_appearance() -> None:
    """Catches the reply layer turning a visual description into diagnosis."""
    response = AgentResponse(
        spoken_text="从你的表情看，你患有抑郁症。",
        display_text="从你的表情看，你患有抑郁症。",
        support_mode="listen",
        risk_level=RiskLevel.GREEN,
        evidence_ids=[],
        visual_observation_ids=["vo_1"],
        action_proposals=[],
        memory_candidates=[],
        avatar_style="concerned_calm",
    )

    with pytest.raises(UnsafeResponseError, match="diagnosis"):
        OutputGuard().validate(response)

