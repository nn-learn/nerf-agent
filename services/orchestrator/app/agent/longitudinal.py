from collections.abc import Iterable
from enum import StrEnum

from pydantic import BaseModel, Field

from app.agent.actions import CapabilityProposalAudit
from app.agent.care import CareLoopState, CareLoopTrace, CarePhase
from app.agent.evidence import EvidenceGateOutcome, EvidenceResponseAudit
from app.agent.intervention_consent import (
    InterventionActionConsentAudit,
    InterventionConsentAudit,
    InterventionConsentState,
    InterventionConsentStatus,
)
from app.agent.interventions import (
    InterventionDecisionOutcome,
    InterventionPolicyAudit,
)
from app.agent.models import AgentDirective, ResponseStrategy
from app.safety.models import RiskLevel


class UserReportedHelpfulness(StrEnum):
    NONE = "NONE"
    HELPED = "HELPED"
    NOT_HELPED = "NOT_HELPED"
    SKIPPED = "SKIPPED"


class DriftSeverity(StrEnum):
    NONE = "NONE"
    WATCH = "WATCH"
    ALERT = "ALERT"


class LongitudinalCareState(BaseModel):
    """Content-free aggregate state; not a clinical outcome record."""

    policy_version: str = "care-observability-v4.3"
    observed_turns: int = Field(default=0, ge=0)
    phase_transition_count: int = Field(default=0, ge=0)
    risk_override_count: int = Field(default=0, ge=0)
    abstention_count: int = Field(default=0, ge=0)
    evidence_block_count: int = Field(default=0, ge=0)
    capability_block_count: int = Field(default=0, ge=0)
    intervention_block_count: int = Field(default=0, ge=0)
    intervention_authorized_count: int = Field(default=0, ge=0)
    consent_decline_count: int = Field(default=0, ge=0)
    consent_cancel_count: int = Field(default=0, ge=0)
    consent_expiry_count: int = Field(default=0, ge=0)
    feedback_helped_count: int = Field(default=0, ge=0)
    feedback_not_helped_count: int = Field(default=0, ge=0)
    feedback_skipped_count: int = Field(default=0, ge=0)
    last_feedback: UserReportedHelpfulness = UserReportedHelpfulness.NONE
    drift_flags: list[str] = Field(default_factory=list)
    drift_severity: DriftSeverity = DriftSeverity.NONE


class CareTelemetrySnapshot(BaseModel):
    """Allowlisted operational fields suitable for structured event export."""

    schema_version: str = "psyavatar-care-telemetry-v4.3"
    operation_name: str = "invoke_agent"
    workflow_name: str = "psyavatar_care_loop"
    care_phase: CarePhase
    risk_level: RiskLevel
    consent_status: InterventionConsentStatus
    user_reported_helpfulness: UserReportedHelpfulness
    observed_turns: int = Field(ge=1)
    phase_transition_count: int = Field(ge=0)
    intervention_authorized_count: int = Field(ge=0)
    abstention_ratio: float = Field(ge=0, le=1)
    policy_blocks_per_turn: float = Field(ge=0)
    drift_flags: list[str] = Field(default_factory=list)
    drift_severity: DriftSeverity
    raw_content_recorded: bool = False
    visual_outcome_inference_used: bool = False
    audio_outcome_inference_used: bool = False


class LongitudinalObserver:
    """Observe policy health and explicit feedback without media inference."""

    _helped_terms = (
        "有帮助",
        "有点帮助",
        "好一些",
        "缓解了",
        "有效",
        "感觉好点",
        "that helped",
        "i feel a bit better",
    )
    _not_helped_terms = (
        "没帮助",
        "没有帮助",
        "没有用",
        "没效果",
        "更糟",
        "didn't help",
        "not helpful",
    )
    _skip_terms = (
        "不想评价",
        "跳过反馈",
        "不评价",
        "skip feedback",
    )

    def observe(
        self,
        previous: LongitudinalCareState | dict[str, object] | None,
        *,
        transcript: str,
        care: CareLoopState,
        care_trace: CareLoopTrace,
        directive: AgentDirective,
        consent: InterventionConsentState,
        consent_audit: InterventionConsentAudit,
        intervention_audit: InterventionPolicyAudit,
        action_consent_audit: InterventionActionConsentAudit,
        evidence_audit: EvidenceResponseAudit | None,
        capability_audit: CapabilityProposalAudit | None,
    ) -> tuple[LongitudinalCareState, CareTelemetrySnapshot]:
        state = self._coerce(previous)
        updated = state.model_copy(deep=True)
        updated.observed_turns += 1
        if care_trace.previous_phase is not care_trace.next_phase:
            updated.phase_transition_count += 1
        if directive.risk_level in {RiskLevel.RED, RiskLevel.EMERGENCY}:
            updated.risk_override_count += 1
        if directive.response_strategy is ResponseStrategy.DETERMINISTIC_ABSTAIN:
            updated.abstention_count += 1
        if evidence_audit is not None and (
            evidence_audit.outcome is EvidenceGateOutcome.BLOCKED
        ):
            updated.evidence_block_count += 1
        if capability_audit is not None and capability_audit.blocked_count:
            updated.capability_block_count += capability_audit.blocked_count
        if intervention_audit.outcome is InterventionDecisionOutcome.BLOCKED:
            updated.intervention_block_count += 1
        updated.intervention_authorized_count += (
            action_consent_audit.authorized_count
        )
        if (
            consent_audit.previous_status is not consent_audit.next_status
            and consent_audit.next_status is InterventionConsentStatus.DECLINED
        ):
            updated.consent_decline_count += 1
        if (
            consent_audit.previous_status is not consent_audit.next_status
            and consent_audit.next_status is InterventionConsentStatus.CANCELLED
        ):
            updated.consent_cancel_count += 1
        if (
            consent_audit.previous_status is not consent_audit.next_status
            and consent_audit.next_status is InterventionConsentStatus.EXPIRED
        ):
            updated.consent_expiry_count += 1

        feedback = self._feedback(transcript, consent)
        updated.last_feedback = feedback
        if feedback is UserReportedHelpfulness.HELPED:
            updated.feedback_helped_count += 1
        elif feedback is UserReportedHelpfulness.NOT_HELPED:
            updated.feedback_not_helped_count += 1
        elif feedback is UserReportedHelpfulness.SKIPPED:
            updated.feedback_skipped_count += 1

        flags = self._drift_flags(updated, care)
        updated.drift_flags = flags
        updated.drift_severity = self._severity(flags)
        return updated, self._snapshot(
            updated,
            care=care,
            directive=directive,
            consent=consent,
            feedback=feedback,
        )

    def _feedback(
        self,
        transcript: str,
        consent: InterventionConsentState,
    ) -> UserReportedHelpfulness:
        if consent.intervention is None or consent.status not in {
            InterventionConsentStatus.ACTIVE,
            InterventionConsentStatus.COMPLETED,
            InterventionConsentStatus.CANCELLED,
        }:
            return UserReportedHelpfulness.NONE
        normalized = transcript.strip().casefold()
        if self._contains_any(normalized, self._not_helped_terms):
            return UserReportedHelpfulness.NOT_HELPED
        if self._contains_any(normalized, self._helped_terms):
            return UserReportedHelpfulness.HELPED
        if self._contains_any(normalized, self._skip_terms):
            return UserReportedHelpfulness.SKIPPED
        return UserReportedHelpfulness.NONE

    @staticmethod
    def _drift_flags(
        state: LongitudinalCareState,
        care: CareLoopState,
    ) -> list[str]:
        flags: list[str] = []
        if state.intervention_block_count >= 2:
            flags.append("REPEATED_INTERVENTION_POLICY_BLOCKS")
        if (
            care.phase in {CarePhase.CHOOSE_STEP, CarePhase.PRACTICE}
            and care.phase_turn_count >= 5
        ):
            flags.append("STALLED_CONFIRMED_STEP")
        if (
            state.feedback_not_helped_count >= 2
            and state.feedback_helped_count == 0
        ):
            flags.append("REPEATED_USER_REPORTED_NOT_HELPFUL")
        if (
            state.observed_turns >= 5
            and state.abstention_count / state.observed_turns >= 0.4
        ):
            flags.append("HIGH_ABSTENTION_RATIO")
        if state.risk_override_count >= 2:
            flags.append("RECURRENT_RISK_OVERRIDE")
        return flags

    @staticmethod
    def _severity(flags: list[str]) -> DriftSeverity:
        if any(
            item in {"RECURRENT_RISK_OVERRIDE", "STALLED_CONFIRMED_STEP"}
            for item in flags
        ):
            return DriftSeverity.ALERT
        if flags:
            return DriftSeverity.WATCH
        return DriftSeverity.NONE

    @staticmethod
    def _snapshot(
        state: LongitudinalCareState,
        *,
        care: CareLoopState,
        directive: AgentDirective,
        consent: InterventionConsentState,
        feedback: UserReportedHelpfulness,
    ) -> CareTelemetrySnapshot:
        turns = state.observed_turns
        policy_blocks = (
            state.evidence_block_count
            + state.capability_block_count
            + state.intervention_block_count
        )
        return CareTelemetrySnapshot(
            care_phase=care.phase,
            risk_level=directive.risk_level,
            consent_status=consent.status,
            user_reported_helpfulness=feedback,
            observed_turns=turns,
            phase_transition_count=state.phase_transition_count,
            intervention_authorized_count=(
                state.intervention_authorized_count
            ),
            abstention_ratio=state.abstention_count / turns,
            policy_blocks_per_turn=policy_blocks / turns,
            drift_flags=state.drift_flags,
            drift_severity=state.drift_severity,
        )

    @staticmethod
    def _coerce(
        previous: LongitudinalCareState | dict[str, object] | None,
    ) -> LongitudinalCareState:
        if previous is None:
            return LongitudinalCareState()
        if isinstance(previous, LongitudinalCareState):
            return previous.model_copy(deep=True)
        return LongitudinalCareState.model_validate(previous)

    @staticmethod
    def _contains_any(text: str, terms: Iterable[str]) -> bool:
        return any(term.casefold() in text for term in terms)
