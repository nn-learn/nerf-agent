import json

from app.agent.care import (
    CareGoalCategory,
    CareLoopState,
    CarePhase,
    GoalOwnership,
)
from app.agent.control import AgentControlPlane
from app.agent.intervention_consent import (
    ConsentSignal,
    InterventionActionConsentGate,
    InterventionConsentPolicy,
    InterventionConsentState,
    InterventionConsentStatus,
)
from app.agent.interventions import InterventionKind, InterventionProposal
from app.agent.models import EvidenceRequirement
from app.safety.models import AgentResponse, RiskAssessment, RiskLevel


def _care(turn: int = 1) -> CareLoopState:
    return CareLoopState(
        phase=CarePhase.CHOOSE_STEP,
        goal_category=CareGoalCategory.CALM_BODY,
        goal_ownership=GoalOwnership.USER_CONFIRMED,
        turn_count=turn,
        phase_turn_count=turn,
    )


def _directive(text: str, level: RiskLevel = RiskLevel.GREEN):
    return AgentControlPlane().draft(
        text,
        RiskAssessment(level=level, confidence=1.0),
    )


def _proposal() -> InterventionProposal:
    return InterventionProposal(
        kind=InterventionKind.PACED_BREATHING,
        capability="start_breathing_exercise",
        requires_explicit_consent=True,
        evidence_requirement=EvidenceRequirement.NONE,
        expires_after_turn=3,
    )


def _response(actions: list[dict[str, object]]) -> AgentResponse:
    return AgentResponse(
        spoken_text="我们可以开始。",
        display_text="我们可以开始。",
        support_mode="exercise",
        risk_level=RiskLevel.GREEN,
        evidence_ids=[],
        visual_observation_ids=[],
        action_proposals=actions,
        memory_candidates=[],
        avatar_style="gentle_guidance",
    )


def test_first_request_only_creates_offer_and_does_not_authorize_tool() -> None:
    state, directive, audit = InterventionConsentPolicy().transition(
        None,
        proposal=_proposal(),
        transcript="请带我做呼吸练习",
        care=_care(),
        directive=_directive("请带我做呼吸练习"),
    )

    assert state.status is InterventionConsentStatus.OFFERED
    assert audit.signal is ConsentSignal.NONE
    assert "CONSENT_SCOPED_INTERVENTION_CAPABILITY" not in directive.reason_codes

    controlled, _, action_audit = InterventionActionConsentGate().apply(
        _response([{"tool": "start_breathing_exercise", "arguments": {}}]),
        state,
        current_turn=1,
    )
    assert controlled.action_proposals == []
    assert action_audit.blocked_count == 1


def test_short_acceptance_only_works_inside_pending_offer_scope() -> None:
    policy = InterventionConsentPolicy()
    idle, _, idle_audit = policy.transition(
        None,
        proposal=None,
        transcript="好的",
        care=_care(),
        directive=_directive("好的"),
    )
    assert idle.status is InterventionConsentStatus.IDLE
    assert idle_audit.signal is ConsentSignal.NONE

    offered, _, _ = policy.transition(
        None,
        proposal=_proposal(),
        transcript="请带我做呼吸练习",
        care=_care(),
        directive=_directive("请带我做呼吸练习"),
    )
    accepted, directive, audit = policy.transition(
        offered,
        proposal=None,
        transcript="可以，开始吧",
        care=_care(2),
        directive=_directive("可以，开始吧"),
    )

    assert accepted.status is InterventionConsentStatus.ACCEPTED
    assert audit.signal is ConsentSignal.ACCEPT
    assert "start_breathing_exercise" in directive.allowed_capabilities
    assert directive.max_tool_proposals == 1


def test_exact_accepted_capability_becomes_active_and_approved() -> None:
    state = InterventionConsentState(
        status=InterventionConsentStatus.ACCEPTED,
        intervention=InterventionKind.PACED_BREATHING,
        capability="start_breathing_exercise",
        offered_turn=1,
        expires_after_turn=3,
        accepted_turn=2,
    )
    response, updated, audit = InterventionActionConsentGate().apply(
        _response([{"tool": "start_breathing_exercise", "arguments": {}}]),
        state,
        current_turn=2,
    )

    assert updated.status is InterventionConsentStatus.ACTIVE
    assert response.action_proposals[0]["status"] == "APPROVED"
    assert response.action_proposals[0]["consent_scope"] == (
        "EXACT_PENDING_INTERVENTION"
    )
    assert audit.authorized_count == 1


def test_decline_stop_expiry_and_risk_override_are_fail_closed() -> None:
    policy = InterventionConsentPolicy()
    offered = InterventionConsentState(
        status=InterventionConsentStatus.OFFERED,
        intervention=InterventionKind.PACED_BREATHING,
        capability="start_breathing_exercise",
        offered_turn=1,
        expires_after_turn=3,
    )
    declined, _, _ = policy.transition(
        offered,
        proposal=None,
        transcript="不用了，算了",
        care=_care(2),
        directive=_directive("不用了，算了"),
    )
    assert declined.status is InterventionConsentStatus.DECLINED

    expired, _, _ = policy.transition(
        offered,
        proposal=None,
        transcript="我们先聊点别的",
        care=_care(4),
        directive=_directive("我们先聊点别的"),
    )
    assert expired.status is InterventionConsentStatus.EXPIRED

    active = offered.model_copy(
        update={"status": InterventionConsentStatus.ACTIVE, "active_turn": 2}
    )
    stopped, _, _ = policy.transition(
        active,
        proposal=None,
        transcript="先停一下",
        care=_care(3),
        directive=_directive("先停一下"),
    )
    assert stopped.status is InterventionConsentStatus.CANCELLED

    crisis, _, crisis_audit = policy.transition(
        active,
        proposal=None,
        transcript="我正在伤害自己",
        care=_care(3),
        directive=_directive("我正在伤害自己", RiskLevel.EMERGENCY),
    )
    assert crisis.status is InterventionConsentStatus.CANCELLED
    assert crisis_audit.reason_codes == [
        "RISK_OVERRIDE_CANCELLED_INTERVENTION"
    ]


def test_barge_in_revokes_pending_or_active_scope_without_raw_reason() -> None:
    state = InterventionConsentState(
        status=InterventionConsentStatus.ACTIVE,
        intervention=InterventionKind.PACED_BREATHING,
        capability="start_breathing_exercise",
        offered_turn=1,
        expires_after_turn=3,
        accepted_turn=2,
        active_turn=2,
    )
    cancelled, audit = InterventionConsentPolicy().cancel_for_interrupt(
        state,
        reason="user_barge_in",
    )

    assert cancelled.status is InterventionConsentStatus.CANCELLED
    assert audit is not None
    payload = json.dumps(audit.model_dump(mode="json"), ensure_ascii=False)
    assert "USER_BARGE_IN_CANCELLED_INTERVENTION" in payload
    assert "user_barge_in" not in payload
