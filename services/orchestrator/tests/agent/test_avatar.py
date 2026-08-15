from app.agent.avatar import (
    AvatarPolicy,
    FacialAffect,
    GazeMode,
    GestureIntensity,
)
from app.agent.control import AgentControlPlane
from app.safety.models import AgentResponse, RiskAssessment, RiskLevel


def _response(
    *,
    support_mode: str = "listen",
    avatar_style: str = "warm",
) -> AgentResponse:
    return AgentResponse.model_validate(
        {
            "spoken_text": "我们可以慢慢来。",
            "display_text": "我们可以慢慢来。",
            "support_mode": support_mode,
            "risk_level": "GREEN",
            "evidence_ids": [],
            "memory_ids": [],
            "visual_observation_ids": [],
            "action_proposals": [],
            "memory_candidates": [],
            "avatar_style": avatar_style,
        }
    )


def test_crisis_policy_overrides_model_style_with_low_stimulation_plan() -> None:
    directive = AgentControlPlane().draft(
        "我正在伤害自己",
        RiskAssessment(level=RiskLevel.EMERGENCY, confidence=1),
    )
    response = _response().model_copy(
        update={"risk_level": RiskLevel.EMERGENCY}
    )

    plan = AvatarPolicy().plan(response, directive)

    assert plan.style == "handoff_calm"
    assert plan.speech_rate < 0.9
    assert plan.gesture_intensity is GestureIntensity.NONE
    assert plan.gaze_mode is GazeMode.RESPECTFUL_NEUTRAL
    assert plan.facial_affect is FacialAffect.CONCERNED
    assert plan.interruptible is True
    assert plan.max_segment_seconds <= 8


def test_amber_policy_is_calm_and_always_interruptible() -> None:
    directive = AgentControlPlane().draft(
        "最近压力很大",
        RiskAssessment(level=RiskLevel.AMBER, confidence=1),
    )
    response = _response().model_copy(update={"risk_level": RiskLevel.AMBER})

    plan = AvatarPolicy().plan(response, directive)

    assert plan.style == "concerned_calm"
    assert plan.gesture_intensity is GestureIntensity.LOW
    assert plan.interruptible is True


def test_guided_exercise_uses_slow_pacing_without_high_intensity_gestures() -> None:
    directive = AgentControlPlane().draft(
        "带我做呼吸练习",
        RiskAssessment(level=RiskLevel.GREEN, confidence=1),
    )

    plan = AvatarPolicy().plan(_response(support_mode="exercise"), directive)

    assert plan.style == "gentle_guidance"
    assert plan.speech_rate == 0.86
    assert plan.sentence_pause_ms >= 600
    assert plan.gesture_intensity is GestureIntensity.LOW
    assert plan.gaze_mode is GazeMode.GUIDED
