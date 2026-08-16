from enum import StrEnum

from pydantic import BaseModel, Field

from app.agent.care import CareGoalCategory, CareLoopState, CarePhase, GoalOwnership
from app.agent.models import AgentDirective, EvidenceRequirement
from app.safety.models import RiskLevel


class InterventionKind(StrEnum):
    REFLECTIVE_LISTENING = "REFLECTIVE_LISTENING"
    PACED_BREATHING = "PACED_BREATHING"
    REVIEWED_PSYCHOEDUCATION = "REVIEWED_PSYCHOEDUCATION"
    HUMAN_HANDOFF = "HUMAN_HANDOFF"
    SESSION_SUMMARY = "SESSION_SUMMARY"


class InterventionDecisionOutcome(StrEnum):
    PROPOSED = "PROPOSED"
    NO_MATCH = "NO_MATCH"
    BLOCKED = "BLOCKED"


class InterventionDefinition(BaseModel):
    kind: InterventionKind
    allowed_phases: list[CarePhase] = Field(min_length=1)
    allowed_risk_levels: list[RiskLevel] = Field(min_length=1)
    goal_categories: list[CareGoalCategory] = Field(min_length=1)
    requires_confirmed_goal: bool
    requires_explicit_consent: bool
    evidence_requirement: EvidenceRequirement
    capability: str | None = None
    cooldown_turns: int = Field(ge=0, le=20)
    max_proposals_per_session: int = Field(ge=1, le=20)


class InterventionProposal(BaseModel):
    kind: InterventionKind
    capability: str | None
    requires_explicit_consent: bool
    evidence_requirement: EvidenceRequirement
    expires_after_turn: int = Field(ge=1)


class InterventionRuntimeState(BaseModel):
    """Session-local counters only; contains no transcript or clinical content."""

    policy_version: str = "intervention-policy-v4.1"
    proposal_counts: dict[InterventionKind, int] = Field(default_factory=dict)
    last_proposed_turn: dict[InterventionKind, int] = Field(default_factory=dict)


class InterventionPolicyAudit(BaseModel):
    policy_version: str
    outcome: InterventionDecisionOutcome
    care_phase: CarePhase
    goal_category: CareGoalCategory
    risk_level: RiskLevel
    selected_intervention: InterventionKind | None = None
    requires_explicit_consent: bool = False
    proposal_count_for_selected: int = Field(default=0, ge=0)
    reason_codes: list[str] = Field(default_factory=list)


class InterventionCatalog:
    """Host-owned catalogue; model output cannot register interventions."""

    def __init__(self) -> None:
        normal = [RiskLevel.GREEN, RiskLevel.AMBER]
        self._definitions = [
            InterventionDefinition(
                kind=InterventionKind.REFLECTIVE_LISTENING,
                allowed_phases=[CarePhase.LISTEN, CarePhase.CLARIFY],
                allowed_risk_levels=normal,
                goal_categories=[CareGoalCategory.UNSET, CareGoalCategory.BE_HEARD],
                requires_confirmed_goal=False,
                requires_explicit_consent=False,
                evidence_requirement=EvidenceRequirement.NONE,
                cooldown_turns=0,
                max_proposals_per_session=20,
            ),
            InterventionDefinition(
                kind=InterventionKind.PACED_BREATHING,
                allowed_phases=[CarePhase.CHOOSE_STEP],
                allowed_risk_levels=normal,
                goal_categories=[CareGoalCategory.CALM_BODY],
                requires_confirmed_goal=True,
                requires_explicit_consent=True,
                evidence_requirement=EvidenceRequirement.NONE,
                capability="start_breathing_exercise",
                cooldown_turns=3,
                max_proposals_per_session=2,
            ),
            InterventionDefinition(
                kind=InterventionKind.REVIEWED_PSYCHOEDUCATION,
                allowed_phases=[CarePhase.UNDERSTAND],
                allowed_risk_levels=normal,
                goal_categories=[CareGoalCategory.UNDERSTAND_EXPERIENCE],
                requires_confirmed_goal=True,
                requires_explicit_consent=False,
                evidence_requirement=EvidenceRequirement.REVIEWED_KNOWLEDGE,
                capability="show_reviewed_resource",
                cooldown_turns=1,
                max_proposals_per_session=4,
            ),
            InterventionDefinition(
                kind=InterventionKind.HUMAN_HANDOFF,
                allowed_phases=[CarePhase.HANDOFF],
                allowed_risk_levels=normal,
                goal_categories=[CareGoalCategory.CONNECT_HUMAN],
                requires_confirmed_goal=True,
                requires_explicit_consent=True,
                evidence_requirement=EvidenceRequirement.NONE,
                capability="request_clinician_handoff",
                cooldown_turns=2,
                max_proposals_per_session=2,
            ),
            InterventionDefinition(
                kind=InterventionKind.SESSION_SUMMARY,
                allowed_phases=[CarePhase.CLOSE],
                allowed_risk_levels=normal,
                goal_categories=[CareGoalCategory.SESSION_CLOSURE],
                requires_confirmed_goal=True,
                requires_explicit_consent=False,
                evidence_requirement=EvidenceRequirement.NONE,
                cooldown_turns=0,
                max_proposals_per_session=1,
            ),
        ]

    def candidates(self, care: CareLoopState) -> list[InterventionDefinition]:
        return [
            item.model_copy(deep=True)
            for item in self._definitions
            if care.phase in item.allowed_phases
            and care.goal_category in item.goal_categories
        ]


class InterventionPolicy:
    """Select one bounded proposal; never executes an intervention."""

    def __init__(self, catalog: InterventionCatalog | None = None) -> None:
        self._catalog = catalog or InterventionCatalog()

    def select(
        self,
        previous: InterventionRuntimeState | dict[str, object] | None,
        *,
        care: CareLoopState,
        directive: AgentDirective,
    ) -> tuple[
        InterventionRuntimeState,
        InterventionProposal | None,
        InterventionPolicyAudit,
    ]:
        runtime = self._coerce(previous)
        if directive.risk_level in {RiskLevel.RED, RiskLevel.EMERGENCY}:
            return runtime, None, self._audit(
                runtime,
                care,
                directive,
                InterventionDecisionOutcome.BLOCKED,
                reason_codes=["CRISIS_RESPONSE_OWNS_TURN"],
            )

        candidates = self._catalog.candidates(care)
        if not candidates:
            return runtime, None, self._audit(
                runtime,
                care,
                directive,
                InterventionDecisionOutcome.NO_MATCH,
                reason_codes=["NO_CATALOG_MATCH"],
            )

        blocked: list[str] = []
        for definition in candidates:
            if directive.risk_level not in definition.allowed_risk_levels:
                blocked.append("INTERVENTION_RISK_BLOCKED")
                continue
            if (
                definition.requires_confirmed_goal
                and care.goal_ownership is not GoalOwnership.USER_CONFIRMED
            ):
                blocked.append("CONFIRMED_GOAL_REQUIRED")
                continue
            if (
                definition.capability is not None
                and definition.capability not in directive.allowed_capabilities
            ):
                blocked.append("CAPABILITY_NOT_IN_DIRECTIVE")
                continue
            count = runtime.proposal_counts.get(definition.kind, 0)
            if count >= definition.max_proposals_per_session:
                blocked.append("SESSION_PROPOSAL_LIMIT_REACHED")
                continue
            last_turn = runtime.last_proposed_turn.get(definition.kind)
            if (
                last_turn is not None
                and care.turn_count - last_turn <= definition.cooldown_turns
            ):
                blocked.append("INTERVENTION_COOLDOWN_ACTIVE")
                continue

            updated = runtime.model_copy(deep=True)
            updated.proposal_counts[definition.kind] = count + 1
            updated.last_proposed_turn[definition.kind] = care.turn_count
            proposal = InterventionProposal(
                kind=definition.kind,
                capability=definition.capability,
                requires_explicit_consent=definition.requires_explicit_consent,
                evidence_requirement=definition.evidence_requirement,
                expires_after_turn=care.turn_count + 2,
            )
            return updated, proposal, self._audit(
                updated,
                care,
                directive,
                InterventionDecisionOutcome.PROPOSED,
                selected=definition,
                reason_codes=["HOST_CATALOG_PROPOSAL"],
            )

        return runtime, None, self._audit(
            runtime,
            care,
            directive,
            InterventionDecisionOutcome.BLOCKED,
            reason_codes=self._deduplicate(blocked),
        )

    @staticmethod
    def _coerce(
        previous: InterventionRuntimeState | dict[str, object] | None,
    ) -> InterventionRuntimeState:
        if previous is None:
            return InterventionRuntimeState()
        if isinstance(previous, InterventionRuntimeState):
            return previous.model_copy(deep=True)
        return InterventionRuntimeState.model_validate(previous)

    @staticmethod
    def _audit(
        runtime: InterventionRuntimeState,
        care: CareLoopState,
        directive: AgentDirective,
        outcome: InterventionDecisionOutcome,
        *,
        selected: InterventionDefinition | None = None,
        reason_codes: list[str],
    ) -> InterventionPolicyAudit:
        kind = selected.kind if selected is not None else None
        return InterventionPolicyAudit(
            policy_version=runtime.policy_version,
            outcome=outcome,
            care_phase=care.phase,
            goal_category=care.goal_category,
            risk_level=directive.risk_level,
            selected_intervention=kind,
            requires_explicit_consent=(
                selected.requires_explicit_consent
                if selected is not None
                else False
            ),
            proposal_count_for_selected=(
                runtime.proposal_counts.get(kind, 0) if kind is not None else 0
            ),
            reason_codes=reason_codes,
        )

    @staticmethod
    def _deduplicate(values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))
