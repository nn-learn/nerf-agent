from collections.abc import Iterable
from enum import StrEnum

from pydantic import BaseModel, Field

from app.agent.care import CareLoopState, CarePhase
from app.agent.interventions import InterventionKind, InterventionProposal
from app.agent.models import AgentDirective
from app.safety.models import AgentResponse, RiskLevel


class InterventionConsentStatus(StrEnum):
    IDLE = "IDLE"
    OFFERED = "OFFERED"
    ACCEPTED = "ACCEPTED"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    DECLINED = "DECLINED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class ConsentSignal(StrEnum):
    NONE = "NONE"
    ACCEPT = "ACCEPT"
    DECLINE = "DECLINE"
    STOP = "STOP"
    COMPLETE = "COMPLETE"


class InterventionConsentState(BaseModel):
    policy_version: str = "intervention-consent-v4.2"
    status: InterventionConsentStatus = InterventionConsentStatus.IDLE
    intervention: InterventionKind | None = None
    capability: str | None = None
    offered_turn: int | None = Field(default=None, ge=1)
    expires_after_turn: int | None = Field(default=None, ge=1)
    accepted_turn: int | None = Field(default=None, ge=1)
    active_turn: int | None = Field(default=None, ge=1)


class InterventionConsentAudit(BaseModel):
    policy_version: str
    previous_status: InterventionConsentStatus
    next_status: InterventionConsentStatus
    intervention: InterventionKind | None
    signal: ConsentSignal
    capability_scoped: bool
    reason_codes: list[str] = Field(default_factory=list)


class InterventionActionConsentAudit(BaseModel):
    policy_version: str
    consent_status: InterventionConsentStatus
    considered_count: int = Field(ge=0)
    authorized_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    reason_codes: list[str] = Field(default_factory=list)


class InterventionConsentPolicy:
    """Two-phase, revocable consent for intervention capabilities."""

    _accept_terms = (
        "可以",
        "好的",
        "好啊",
        "开始吧",
        "我愿意",
        "同意",
        "yes",
        "let's start",
        "i agree",
    )
    _decline_terms = (
        "不想",
        "不要",
        "不用",
        "算了",
        "拒绝",
        "换一个",
        "no thanks",
        "i decline",
    )
    _stop_terms = (
        "停止",
        "停一下",
        "先停",
        "别继续",
        "stop",
        "pause it",
    )
    _complete_terms = (
        "完成了",
        "做完了",
        "结束练习",
        "i finished",
        "exercise complete",
    )

    def transition(
        self,
        previous: InterventionConsentState | dict[str, object] | None,
        *,
        proposal: InterventionProposal | None,
        transcript: str,
        care: CareLoopState,
        directive: AgentDirective,
    ) -> tuple[InterventionConsentState, AgentDirective, InterventionConsentAudit]:
        prior = self._coerce(previous)
        signal = self._signal(transcript, scoped=prior.status in self._pending())
        state = prior.model_copy(deep=True)
        reasons: list[str] = []

        if directive.risk_level in {RiskLevel.RED, RiskLevel.EMERGENCY}:
            state = self._terminal(prior, InterventionConsentStatus.CANCELLED)
            reasons.append("RISK_OVERRIDE_CANCELLED_INTERVENTION")
        elif prior.status is InterventionConsentStatus.OFFERED:
            if signal in {ConsentSignal.DECLINE, ConsentSignal.STOP}:
                state.status = InterventionConsentStatus.DECLINED
                reasons.append("USER_DECLINED_OFFER")
            elif signal is ConsentSignal.ACCEPT:
                state.status = InterventionConsentStatus.ACCEPTED
                state.accepted_turn = care.turn_count
                reasons.append("USER_EXPLICITLY_ACCEPTED")
            elif (
                prior.expires_after_turn is not None
                and care.turn_count > prior.expires_after_turn
            ):
                state.status = InterventionConsentStatus.EXPIRED
                reasons.append("OFFER_EXPIRED")
            else:
                reasons.append("AWAITING_EXPLICIT_CONSENT")
        elif prior.status is InterventionConsentStatus.ACCEPTED:
            if signal in {ConsentSignal.DECLINE, ConsentSignal.STOP}:
                state.status = InterventionConsentStatus.CANCELLED
                reasons.append("USER_REVOKED_BEFORE_START")
            else:
                reasons.append("ACCEPTED_SCOPE_CONTINUED")
        elif prior.status is InterventionConsentStatus.ACTIVE:
            if signal in {ConsentSignal.DECLINE, ConsentSignal.STOP}:
                state.status = InterventionConsentStatus.CANCELLED
                reasons.append("USER_STOPPED_ACTIVE_INTERVENTION")
            elif signal is ConsentSignal.COMPLETE:
                state.status = InterventionConsentStatus.COMPLETED
                reasons.append("USER_REPORTED_COMPLETION")
            else:
                reasons.append("ACTIVE_INTERVENTION_CONTINUED")
        elif proposal is not None and proposal.requires_explicit_consent:
            state = InterventionConsentState(
                status=InterventionConsentStatus.OFFERED,
                intervention=proposal.kind,
                capability=proposal.capability,
                offered_turn=care.turn_count,
                expires_after_turn=proposal.expires_after_turn,
            )
            signal = ConsentSignal.NONE
            reasons.append("EXPLICIT_CONSENT_REQUIRED")
        else:
            signal = ConsentSignal.NONE
            reasons.append("NO_SCOPED_CONSENT_REQUEST")

        controlled = directive
        if (
            state.status is InterventionConsentStatus.ACCEPTED
            and state.capability is not None
        ):
            capabilities = list(dict.fromkeys([
                *directive.allowed_capabilities,
                state.capability,
            ]))
            controlled = directive.model_copy(
                update={
                    "allowed_capabilities": capabilities,
                    "max_tool_proposals": max(1, directive.max_tool_proposals),
                    "reason_codes": [
                        *directive.reason_codes,
                        "CONSENT_SCOPED_INTERVENTION_CAPABILITY",
                    ],
                }
            )

        return state, controlled, self._audit(prior, state, signal, reasons)

    def cancel_for_interrupt(
        self,
        previous: InterventionConsentState,
        *,
        reason: str,
    ) -> tuple[InterventionConsentState, InterventionConsentAudit | None]:
        if previous.status not in {
            InterventionConsentStatus.OFFERED,
            InterventionConsentStatus.ACCEPTED,
            InterventionConsentStatus.ACTIVE,
        }:
            return previous.model_copy(deep=True), None
        state = previous.model_copy(
            deep=True,
            update={"status": InterventionConsentStatus.CANCELLED},
        )
        reason_code = (
            "USER_BARGE_IN_CANCELLED_INTERVENTION"
            if reason == "user_barge_in"
            else "TURN_INTERRUPT_CANCELLED_INTERVENTION"
        )
        return state, self._audit(
            previous,
            state,
            ConsentSignal.STOP,
            [reason_code],
        )

    @staticmethod
    def reconcile_care(
        care: CareLoopState,
        consent: InterventionConsentState,
        *,
        previous_status: InterventionConsentStatus,
    ) -> CareLoopState:
        next_phase = care.phase
        if consent.status is InterventionConsentStatus.ACTIVE:
            next_phase = CarePhase.PRACTICE
        elif consent.status is InterventionConsentStatus.COMPLETED:
            next_phase = CarePhase.REFLECT
        elif consent.status in {
            InterventionConsentStatus.DECLINED,
            InterventionConsentStatus.EXPIRED,
        }:
            next_phase = CarePhase.LISTEN
        elif consent.status is InterventionConsentStatus.CANCELLED:
            next_phase = (
                CarePhase.REFLECT
                if previous_status in {
                    InterventionConsentStatus.ACCEPTED,
                    InterventionConsentStatus.ACTIVE,
                }
                else CarePhase.LISTEN
            )
        if next_phase is care.phase:
            return care.model_copy(deep=True)
        return care.model_copy(
            deep=True,
            update={"phase": next_phase, "phase_turn_count": 1},
        )

    @staticmethod
    def _coerce(
        previous: InterventionConsentState | dict[str, object] | None,
    ) -> InterventionConsentState:
        if previous is None:
            return InterventionConsentState()
        if isinstance(previous, InterventionConsentState):
            return previous.model_copy(deep=True)
        return InterventionConsentState.model_validate(previous)

    def _signal(self, transcript: str, *, scoped: bool) -> ConsentSignal:
        if not scoped:
            return ConsentSignal.NONE
        normalized = transcript.strip().casefold()
        ordered = (
            (ConsentSignal.STOP, self._stop_terms),
            (ConsentSignal.DECLINE, self._decline_terms),
            (ConsentSignal.COMPLETE, self._complete_terms),
            (ConsentSignal.ACCEPT, self._accept_terms),
        )
        for signal, terms in ordered:
            if self._contains_any(normalized, terms):
                return signal
        return ConsentSignal.NONE

    @staticmethod
    def _pending() -> set[InterventionConsentStatus]:
        return {
            InterventionConsentStatus.OFFERED,
            InterventionConsentStatus.ACCEPTED,
            InterventionConsentStatus.ACTIVE,
        }

    @staticmethod
    def _terminal(
        previous: InterventionConsentState,
        status: InterventionConsentStatus,
    ) -> InterventionConsentState:
        if previous.status is InterventionConsentStatus.IDLE:
            return previous.model_copy(deep=True)
        return previous.model_copy(deep=True, update={"status": status})

    @staticmethod
    def _audit(
        previous: InterventionConsentState,
        state: InterventionConsentState,
        signal: ConsentSignal,
        reasons: list[str],
    ) -> InterventionConsentAudit:
        return InterventionConsentAudit(
            policy_version=state.policy_version,
            previous_status=previous.status,
            next_status=state.status,
            intervention=state.intervention,
            signal=signal,
            capability_scoped=(
                state.status is InterventionConsentStatus.ACCEPTED
                and state.capability is not None
            ),
            reason_codes=reasons,
        )

    @staticmethod
    def _contains_any(text: str, terms: Iterable[str]) -> bool:
        return any(term.casefold() in text for term in terms)


class InterventionActionConsentGate:
    """Remove intervention tool proposals until exact scoped consent exists."""

    def apply(
        self,
        response: AgentResponse,
        state: InterventionConsentState,
        *,
        current_turn: int,
    ) -> tuple[
        AgentResponse,
        InterventionConsentState,
        InterventionActionConsentAudit,
    ]:
        accepted: list[dict[str, object]] = []
        blocked = 0
        authorized = 0
        reasons: list[str] = []
        updated = state.model_copy(deep=True)

        for proposal in response.action_proposals:
            tool = proposal.get("tool") if isinstance(proposal, dict) else None
            if state.capability is None or tool != state.capability:
                accepted.append(proposal)
                continue
            if state.status is not InterventionConsentStatus.ACCEPTED:
                blocked += 1
                reasons.append("INTERVENTION_EXPLICIT_CONSENT_MISSING")
                continue
            controlled = dict(proposal)
            controlled["status"] = "APPROVED"
            controlled["intervention"] = (
                state.intervention.value if state.intervention is not None else None
            )
            controlled["consent_scope"] = "EXACT_PENDING_INTERVENTION"
            accepted.append(controlled)
            authorized += 1
            updated.status = InterventionConsentStatus.ACTIVE
            updated.active_turn = current_turn
            reasons.append("INTERVENTION_ACTION_AUTHORIZED")

        if not reasons:
            reasons.append("NO_INTERVENTION_ACTION_PROPOSAL")
        return (
            response.model_copy(update={"action_proposals": accepted}),
            updated,
            InterventionActionConsentAudit(
                policy_version=state.policy_version,
                consent_status=updated.status,
                considered_count=len(response.action_proposals),
                authorized_count=authorized,
                blocked_count=blocked,
                reason_codes=list(dict.fromkeys(reasons)),
            ),
        )
