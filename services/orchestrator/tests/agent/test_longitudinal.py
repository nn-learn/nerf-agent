import json

from app.agent.actions import CapabilityProposalAudit
from app.agent.care import (
    CareGoalCategory,
    CareLoopState,
    CareLoopTrace,
    CarePhase,
    GoalOwnership,
)
from app.agent.control import AgentControlPlane
from app.agent.evidence import EvidenceGateOutcome, EvidenceResponseAudit
from app.agent.intervention_consent import (
    ConsentSignal,
    InterventionActionConsentAudit,
    InterventionConsentAudit,
    InterventionConsentState,
    InterventionConsentStatus,
)
from app.agent.interventions import (
    InterventionDecisionOutcome,
    InterventionKind,
    InterventionPolicyAudit,
)
from app.agent.longitudinal import (
    DriftSeverity,
    LongitudinalCareState,
    LongitudinalObserver,
    UserReportedHelpfulness,
)
from app.agent.models import EvidenceRequirement
from app.safety.models import RiskAssessment, RiskLevel


def _care(turn: int = 1, phase_turns: int = 1) -> CareLoopState:
    return CareLoopState(
        phase=CarePhase.PRACTICE,
        goal_category=CareGoalCategory.CALM_BODY,
        goal_ownership=GoalOwnership.USER_CONFIRMED,
        turn_count=turn,
        phase_turn_count=phase_turns,
    )


def _trace(care: CareLoopState) -> CareLoopTrace:
    return CareLoopTrace(
        policy_version=care.policy_version,
        previous_phase=care.phase,
        next_phase=care.phase,
        goal_category=care.goal_category,
        goal_ownership=care.goal_ownership,
        turn_count=care.turn_count,
        phase_turn_count=care.phase_turn_count,
        confirmed_goal_revision=care.confirmed_goal_revision,
        safety_override_count=care.safety_override_count,
    )


def _consent(status: InterventionConsentStatus) -> InterventionConsentState:
    return InterventionConsentState(
        status=status,
        intervention=InterventionKind.PACED_BREATHING,
        capability="start_breathing_exercise",
        offered_turn=1,
        expires_after_turn=3,
        accepted_turn=2,
        active_turn=2 if status is InterventionConsentStatus.ACTIVE else None,
    )


def _observe(
    previous: LongitudinalCareState | None,
    *,
    transcript: str,
    consent: InterventionConsentState,
    care: CareLoopState | None = None,
):
    current_care = care or _care()
    return LongitudinalObserver().observe(
        previous,
        transcript=transcript,
        care=current_care,
        care_trace=_trace(current_care),
        directive=AgentControlPlane().draft(
            "最近压力很大",
            RiskAssessment(level=RiskLevel.GREEN, confidence=1.0),
        ),
        consent=consent,
        consent_audit=InterventionConsentAudit(
            policy_version=consent.policy_version,
            previous_status=consent.status,
            next_status=consent.status,
            intervention=consent.intervention,
            signal=ConsentSignal.NONE,
            capability_scoped=False,
        ),
        intervention_audit=InterventionPolicyAudit(
            policy_version="intervention-policy-v4.1",
            outcome=InterventionDecisionOutcome.NO_MATCH,
            care_phase=current_care.phase,
            goal_category=current_care.goal_category,
            risk_level=RiskLevel.GREEN,
        ),
        action_consent_audit=InterventionActionConsentAudit(
            policy_version=consent.policy_version,
            consent_status=consent.status,
            considered_count=0,
            authorized_count=0,
            blocked_count=0,
        ),
        evidence_audit=EvidenceResponseAudit(
            outcome=EvidenceGateOutcome.PASS,
            requirement=EvidenceRequirement.NONE,
            reviewed_citation_count=0,
            memory_citation_count=0,
            blocked_citation_count=0,
        ),
        capability_audit=CapabilityProposalAudit(
            proposed_count=0,
            accepted_count=0,
            blocked_count=0,
        ),
    )


def test_helpfulness_is_counted_only_for_an_active_scoped_intervention() -> None:
    idle = InterventionConsentState()
    idle_state, idle_snapshot = _observe(
        None,
        transcript="这很有帮助",
        consent=idle,
    )
    assert idle_state.feedback_helped_count == 0
    assert idle_snapshot.user_reported_helpfulness is UserReportedHelpfulness.NONE

    active_state, snapshot = _observe(
        None,
        transcript="这个练习有帮助，我感觉好点了",
        consent=_consent(InterventionConsentStatus.ACTIVE),
    )
    assert active_state.feedback_helped_count == 1
    assert snapshot.user_reported_helpfulness is UserReportedHelpfulness.HELPED


def test_repeated_explicit_not_helpful_feedback_raises_watch_not_diagnosis() -> None:
    first, _ = _observe(
        None,
        transcript="这个练习没帮助",
        consent=_consent(InterventionConsentStatus.ACTIVE),
    )
    second, snapshot = _observe(
        first,
        transcript="还是没有用，没效果",
        consent=_consent(InterventionConsentStatus.ACTIVE),
        care=_care(turn=2, phase_turns=2),
    )

    assert "REPEATED_USER_REPORTED_NOT_HELPFUL" in second.drift_flags
    assert snapshot.drift_severity is DriftSeverity.WATCH


def test_telemetry_is_allowlisted_and_never_claims_media_outcome_inference() -> None:
    transcript = "这个练习有帮助，但这段原文不能进入遥测"
    _, snapshot = _observe(
        None,
        transcript=transcript,
        consent=_consent(InterventionConsentStatus.ACTIVE),
    )
    payload = json.dumps(snapshot.model_dump(mode="json"), ensure_ascii=False)

    assert transcript not in payload
    assert snapshot.raw_content_recorded is False
    assert snapshot.visual_outcome_inference_used is False
    assert snapshot.audio_outcome_inference_used is False
    assert snapshot.operation_name == "invoke_agent"
