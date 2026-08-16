import json

from app.agent.care import (
    CareGoalCategory,
    CareLoopPolicy,
    CareLoopState,
    CarePhase,
    GoalOwnership,
)
from app.agent.models import AgentIntent
from app.safety.models import RiskAssessment, RiskLevel


def _risk(level: RiskLevel) -> RiskAssessment:
    return RiskAssessment(level=level, confidence=1.0)


def test_emotional_disclosure_does_not_invent_a_user_goal() -> None:
    state, trace = CareLoopPolicy().advance(
        None,
        transcript="最近工作压力很大",
        risk=_risk(RiskLevel.GREEN),
        intent=AgentIntent.EMOTIONAL_DISCLOSURE,
    )

    assert state.phase is CarePhase.LISTEN
    assert state.goal_category is CareGoalCategory.UNSET
    assert state.goal_ownership is GoalOwnership.UNSET
    assert trace.reason_codes == ["LISTEN_BEFORE_GOAL"]


def test_explicit_calm_goal_is_confirmed_without_storing_raw_text() -> None:
    transcript = "我想冷静下来，请带我呼吸"
    state, trace = CareLoopPolicy().advance(
        None,
        transcript=transcript,
        risk=_risk(RiskLevel.GREEN),
        intent=AgentIntent.COPING_EXERCISE,
    )

    assert state.phase is CarePhase.CHOOSE_STEP
    assert state.goal_category is CareGoalCategory.CALM_BODY
    assert state.goal_ownership is GoalOwnership.USER_CONFIRMED
    assert state.confirmed_goal_revision == 1
    assert transcript not in json.dumps(trace.model_dump(mode="json"), ensure_ascii=False)


def test_memory_cannot_silently_become_a_care_goal() -> None:
    previous = CareLoopState(turn_count=1, phase_turn_count=1)
    state, _ = CareLoopPolicy().advance(
        previous,
        transcript="你记得我之前说过我想改善睡眠吗？",
        risk=_risk(RiskLevel.GREEN),
        intent=AgentIntent.MEMORY_RECALL,
    )

    assert state.goal_category is CareGoalCategory.UNSET
    assert state.goal_ownership is GoalOwnership.UNSET


def test_crisis_uses_distinct_safety_override_not_user_confirmation() -> None:
    state, trace = CareLoopPolicy().advance(
        None,
        transcript="我现在正在伤害自己",
        risk=_risk(RiskLevel.EMERGENCY),
        intent=AgentIntent.GENERAL_SUPPORT,
    )

    assert state.phase is CarePhase.HANDOFF
    assert state.goal_category is CareGoalCategory.SAFETY_CONNECTION
    assert state.goal_ownership is GoalOwnership.SAFETY_OVERRIDE
    assert trace.reason_codes == ["RISK_OVERRIDE"]


def test_user_confirmed_goal_continues_across_turns() -> None:
    policy = CareLoopPolicy()
    first, _ = policy.advance(
        None,
        transcript="我想聊聊，先听我说",
        risk=_risk(RiskLevel.GREEN),
        intent=AgentIntent.GENERAL_SUPPORT,
    )
    second, trace = policy.advance(
        first,
        transcript="今天开会的时候我很紧张",
        risk=_risk(RiskLevel.GREEN),
        intent=AgentIntent.EMOTIONAL_DISCLOSURE,
    )

    assert second.goal_category is CareGoalCategory.BE_HEARD
    assert second.goal_ownership is GoalOwnership.USER_CONFIRMED
    assert second.confirmed_goal_revision == 1
    assert second.turn_count == 2
    assert trace.reason_codes == ["USER_GOAL_CONTINUED"]
