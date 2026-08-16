import json

from app.agent.care import (
    CareGoalCategory,
    CareLoopState,
    CarePhase,
    GoalOwnership,
)
from app.agent.control import AgentControlPlane
from app.agent.interventions import (
    InterventionDecisionOutcome,
    InterventionKind,
    InterventionPolicy,
    InterventionRuntimeState,
)
from app.safety.models import RiskAssessment, RiskLevel


def _directive(text: str, level: RiskLevel = RiskLevel.GREEN):
    return AgentControlPlane().draft(
        text,
        RiskAssessment(level=level, confidence=1.0),
    )


def _care(
    *,
    phase: CarePhase,
    goal: CareGoalCategory,
    ownership: GoalOwnership,
    turn: int = 1,
) -> CareLoopState:
    return CareLoopState(
        phase=phase,
        goal_category=goal,
        goal_ownership=ownership,
        turn_count=turn,
        phase_turn_count=1,
    )


def test_unconfirmed_disclosure_only_selects_reflective_listening() -> None:
    _, proposal, audit = InterventionPolicy().select(
        None,
        care=_care(
            phase=CarePhase.LISTEN,
            goal=CareGoalCategory.UNSET,
            ownership=GoalOwnership.UNSET,
        ),
        directive=_directive("最近压力很大"),
    )

    assert proposal is not None
    assert proposal.kind is InterventionKind.REFLECTIVE_LISTENING
    assert proposal.capability is None
    assert proposal.requires_explicit_consent is False
    assert audit.outcome is InterventionDecisionOutcome.PROPOSED


def test_breathing_is_only_a_proposal_and_requires_explicit_consent() -> None:
    state, proposal, _ = InterventionPolicy().select(
        None,
        care=_care(
            phase=CarePhase.CHOOSE_STEP,
            goal=CareGoalCategory.CALM_BODY,
            ownership=GoalOwnership.USER_CONFIRMED,
        ),
        directive=_directive("请带我做呼吸练习"),
    )

    assert proposal is not None
    assert proposal.kind is InterventionKind.PACED_BREATHING
    assert proposal.capability == "start_breathing_exercise"
    assert proposal.requires_explicit_consent is True
    assert state.proposal_counts[InterventionKind.PACED_BREATHING] == 1


def test_memory_or_model_cannot_supply_required_goal_confirmation() -> None:
    _, proposal, audit = InterventionPolicy().select(
        None,
        care=_care(
            phase=CarePhase.CHOOSE_STEP,
            goal=CareGoalCategory.CALM_BODY,
            ownership=GoalOwnership.UNSET,
        ),
        directive=_directive("请带我做呼吸练习"),
    )

    assert proposal is None
    assert audit.reason_codes == ["CONFIRMED_GOAL_REQUIRED"]


def test_crisis_blocks_normal_intervention_catalog() -> None:
    _, proposal, audit = InterventionPolicy().select(
        None,
        care=_care(
            phase=CarePhase.HANDOFF,
            goal=CareGoalCategory.SAFETY_CONNECTION,
            ownership=GoalOwnership.SAFETY_OVERRIDE,
        ),
        directive=_directive("我正在伤害自己", RiskLevel.EMERGENCY),
    )

    assert proposal is None
    assert audit.outcome is InterventionDecisionOutcome.BLOCKED
    assert audit.reason_codes == ["CRISIS_RESPONSE_OWNS_TURN"]


def test_cooldown_and_session_limit_prevent_repetitive_prompting() -> None:
    policy = InterventionPolicy()
    first_state, first, _ = policy.select(
        None,
        care=_care(
            phase=CarePhase.CHOOSE_STEP,
            goal=CareGoalCategory.CALM_BODY,
            ownership=GoalOwnership.USER_CONFIRMED,
            turn=1,
        ),
        directive=_directive("请带我做呼吸练习"),
    )
    assert first is not None

    _, second, cooldown_audit = policy.select(
        first_state,
        care=_care(
            phase=CarePhase.CHOOSE_STEP,
            goal=CareGoalCategory.CALM_BODY,
            ownership=GoalOwnership.USER_CONFIRMED,
            turn=2,
        ),
        directive=_directive("请带我做呼吸练习"),
    )
    assert second is None
    assert cooldown_audit.reason_codes == ["INTERVENTION_COOLDOWN_ACTIVE"]

    capped = InterventionRuntimeState(
        proposal_counts={InterventionKind.PACED_BREATHING: 2},
        last_proposed_turn={InterventionKind.PACED_BREATHING: 1},
    )
    _, third, limit_audit = policy.select(
        capped,
        care=_care(
            phase=CarePhase.CHOOSE_STEP,
            goal=CareGoalCategory.CALM_BODY,
            ownership=GoalOwnership.USER_CONFIRMED,
            turn=20,
        ),
        directive=_directive("请带我做呼吸练习"),
    )
    assert third is None
    assert limit_audit.reason_codes == ["SESSION_PROPOSAL_LIMIT_REACHED"]


def test_intervention_audit_contains_no_transcript() -> None:
    transcript = "请带我做呼吸练习，这是不应进入审计的原文"
    _, _, audit = InterventionPolicy().select(
        None,
        care=_care(
            phase=CarePhase.CHOOSE_STEP,
            goal=CareGoalCategory.CALM_BODY,
            ownership=GoalOwnership.USER_CONFIRMED,
        ),
        directive=_directive(transcript),
    )

    payload = json.dumps(audit.model_dump(mode="json"), ensure_ascii=False)
    assert transcript not in payload
