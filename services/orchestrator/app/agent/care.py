from collections.abc import Iterable
from enum import StrEnum

from pydantic import BaseModel, Field

from app.agent.models import AgentIntent
from app.safety.models import RiskAssessment, RiskLevel


class CarePhase(StrEnum):
    LISTEN = "LISTEN"
    CLARIFY = "CLARIFY"
    UNDERSTAND = "UNDERSTAND"
    CHOOSE_STEP = "CHOOSE_STEP"
    PRACTICE = "PRACTICE"
    REFLECT = "REFLECT"
    CLOSE = "CLOSE"
    HANDOFF = "HANDOFF"


class CareGoalCategory(StrEnum):
    UNSET = "UNSET"
    BE_HEARD = "BE_HEARD"
    CALM_BODY = "CALM_BODY"
    UNDERSTAND_EXPERIENCE = "UNDERSTAND_EXPERIENCE"
    CONNECT_HUMAN = "CONNECT_HUMAN"
    SAFETY_CONNECTION = "SAFETY_CONNECTION"
    SESSION_CLOSURE = "SESSION_CLOSURE"


class GoalOwnership(StrEnum):
    UNSET = "UNSET"
    USER_CONFIRMED = "USER_CONFIRMED"
    SAFETY_OVERRIDE = "SAFETY_OVERRIDE"


class CareLoopState(BaseModel):
    """Content-free, session-scoped state for a user-owned support loop."""

    policy_version: str = "care-loop-v4.0"
    phase: CarePhase = CarePhase.LISTEN
    goal_category: CareGoalCategory = CareGoalCategory.UNSET
    goal_ownership: GoalOwnership = GoalOwnership.UNSET
    turn_count: int = Field(default=0, ge=0)
    phase_turn_count: int = Field(default=0, ge=0)
    confirmed_goal_revision: int = Field(default=0, ge=0)
    safety_override_count: int = Field(default=0, ge=0)


class CareLoopTrace(BaseModel):
    """Content-free transition trace; never contains transcripts or model text."""

    policy_version: str
    previous_phase: CarePhase
    next_phase: CarePhase
    goal_category: CareGoalCategory
    goal_ownership: GoalOwnership
    turn_count: int
    phase_turn_count: int
    confirmed_goal_revision: int
    safety_override_count: int
    reason_codes: list[str] = Field(default_factory=list)


class CareLoopPolicy:
    """Deterministic care-loop policy above memory, RAG, and the model worker."""

    _be_heard_terms = (
        "我想聊聊",
        "听我说",
        "陪我聊",
        "只想说说",
        "先听我说",
        "i want to talk",
        "just listen",
    )
    _calm_terms = (
        "我想冷静",
        "想平静下来",
        "帮我放松",
        "带我呼吸",
        "呼吸练习",
        "放松练习",
        "help me calm",
        "breathing exercise",
        "grounding exercise",
    )
    _understand_terms = (
        "我想弄明白",
        "帮我理解",
        "为什么我会",
        "什么是",
        "为什么会",
        "help me understand",
        "why do i",
        "what is",
    )
    _handoff_terms = (
        "转人工",
        "真人咨询师",
        "联系咨询师",
        "联系医生",
        "找医生",
        "人工接管",
        "human counselor",
        "talk to a clinician",
    )
    _close_terms = (
        "先到这里",
        "今天就到这",
        "结束聊天",
        "不聊了",
        "stop here",
        "end the session",
    )

    def advance(
        self,
        previous: CareLoopState | dict[str, object] | None,
        *,
        transcript: str,
        risk: RiskAssessment,
        intent: AgentIntent,
    ) -> tuple[CareLoopState, CareLoopTrace]:
        prior = self._coerce(previous)
        reasons: list[str] = []
        next_phase = prior.phase
        next_goal = prior.goal_category
        next_ownership = prior.goal_ownership
        revision = prior.confirmed_goal_revision
        safety_count = prior.safety_override_count

        if risk.level in {RiskLevel.RED, RiskLevel.EMERGENCY}:
            next_phase = CarePhase.HANDOFF
            next_goal = CareGoalCategory.SAFETY_CONNECTION
            next_ownership = GoalOwnership.SAFETY_OVERRIDE
            safety_count += 1
            reasons.append("RISK_OVERRIDE")
        else:
            if prior.goal_ownership is GoalOwnership.SAFETY_OVERRIDE:
                next_phase = CarePhase.LISTEN
                next_goal = CareGoalCategory.UNSET
                next_ownership = GoalOwnership.UNSET
                reasons.append("SAFETY_OVERRIDE_CLEARED")

            explicit = self._explicit_goal(transcript, intent)
            if explicit is not None:
                goal, phase, reason = explicit
                if (
                    next_ownership is not GoalOwnership.USER_CONFIRMED
                    or next_goal is not goal
                ):
                    revision += 1
                next_goal = goal
                next_phase = phase
                next_ownership = GoalOwnership.USER_CONFIRMED
                reasons.append(reason)
            elif next_ownership is GoalOwnership.USER_CONFIRMED:
                reasons.append("USER_GOAL_CONTINUED")
            elif intent is AgentIntent.EMOTIONAL_DISCLOSURE:
                next_phase = (
                    CarePhase.CLARIFY
                    if prior.turn_count > 0 and prior.phase is CarePhase.LISTEN
                    else CarePhase.LISTEN
                )
                reasons.append(
                    "GOAL_CLARIFICATION_DUE"
                    if next_phase is CarePhase.CLARIFY
                    else "LISTEN_BEFORE_GOAL"
                )
            else:
                next_phase = CarePhase.LISTEN
                reasons.append("NO_CONFIRMED_GOAL")

        phase_turn_count = (
            prior.phase_turn_count + 1 if next_phase is prior.phase else 1
        )
        state = CareLoopState(
            phase=next_phase,
            goal_category=next_goal,
            goal_ownership=next_ownership,
            turn_count=prior.turn_count + 1,
            phase_turn_count=phase_turn_count,
            confirmed_goal_revision=revision,
            safety_override_count=safety_count,
        )
        trace = CareLoopTrace(
            policy_version=state.policy_version,
            previous_phase=prior.phase,
            next_phase=state.phase,
            goal_category=state.goal_category,
            goal_ownership=state.goal_ownership,
            turn_count=state.turn_count,
            phase_turn_count=state.phase_turn_count,
            confirmed_goal_revision=state.confirmed_goal_revision,
            safety_override_count=state.safety_override_count,
            reason_codes=reasons,
        )
        return state, trace

    def _explicit_goal(
        self,
        transcript: str,
        intent: AgentIntent,
    ) -> tuple[CareGoalCategory, CarePhase, str] | None:
        normalized = transcript.strip().casefold()
        if self._contains_any(normalized, self._close_terms):
            return (
                CareGoalCategory.SESSION_CLOSURE,
                CarePhase.CLOSE,
                "USER_CONFIRMED_CLOSURE",
            )
        if intent is AgentIntent.CLINICIAN_HANDOFF or self._contains_any(
            normalized, self._handoff_terms
        ):
            return (
                CareGoalCategory.CONNECT_HUMAN,
                CarePhase.HANDOFF,
                "USER_CONFIRMED_HUMAN_SUPPORT",
            )
        if intent is AgentIntent.COPING_EXERCISE or self._contains_any(
            normalized, self._calm_terms
        ):
            return (
                CareGoalCategory.CALM_BODY,
                CarePhase.CHOOSE_STEP,
                "USER_CONFIRMED_CALM_GOAL",
            )
        if intent is AgentIntent.PSYCHOEDUCATION or self._contains_any(
            normalized, self._understand_terms
        ):
            return (
                CareGoalCategory.UNDERSTAND_EXPERIENCE,
                CarePhase.UNDERSTAND,
                "USER_CONFIRMED_UNDERSTANDING_GOAL",
            )
        if self._contains_any(normalized, self._be_heard_terms):
            return (
                CareGoalCategory.BE_HEARD,
                CarePhase.LISTEN,
                "USER_CONFIRMED_LISTENING_GOAL",
            )
        return None

    @staticmethod
    def _coerce(
        previous: CareLoopState | dict[str, object] | None,
    ) -> CareLoopState:
        if previous is None:
            return CareLoopState()
        if isinstance(previous, CareLoopState):
            return previous.model_copy(deep=True)
        return CareLoopState.model_validate(previous)

    @staticmethod
    def _contains_any(text: str, terms: Iterable[str]) -> bool:
        return any(term.casefold() in text for term in terms)
