from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from app.agent.models import (
    AgentDirective,
    AgentIntent,
    ResponseStrategy,
)
from app.safety.models import AgentResponse, RiskLevel

AvatarStyle = Literal[
    "neutral_listening",
    "warm",
    "concerned_calm",
    "gentle_guidance",
    "handoff_calm",
]


class GestureIntensity(StrEnum):
    NONE = "NONE"
    LOW = "LOW"
    MODERATE = "MODERATE"


class GazeMode(StrEnum):
    SOFT_DIRECT = "SOFT_DIRECT"
    RESPECTFUL_NEUTRAL = "RESPECTFUL_NEUTRAL"
    GUIDED = "GUIDED"


class FacialAffect(StrEnum):
    NEUTRAL = "NEUTRAL"
    WARM = "WARM"
    CONCERNED = "CONCERNED"


class AvatarResponsePlan(BaseModel):
    policy_version: str = "avatar-policy-v3.3"
    style: AvatarStyle
    speech_rate: float = Field(ge=0.80, le=1.10)
    initial_pause_ms: int = Field(ge=0, le=1_500)
    sentence_pause_ms: int = Field(ge=100, le=1_000)
    gesture_intensity: GestureIntensity
    gaze_mode: GazeMode
    facial_affect: FacialAffect
    interruptible: bool
    max_segment_seconds: float = Field(gt=0, le=15)
    reason_codes: list[str] = Field(default_factory=list)


class AvatarPolicy:
    """Host-owned non-verbal policy; model avatar_style is only a suggestion."""

    def plan(
        self,
        response: AgentResponse,
        directive: AgentDirective,
    ) -> AvatarResponsePlan:
        _ = response.avatar_style
        if directive.risk_level in {RiskLevel.RED, RiskLevel.EMERGENCY}:
            return AvatarResponsePlan(
                style="handoff_calm",
                speech_rate=0.88,
                initial_pause_ms=350,
                sentence_pause_ms=550,
                gesture_intensity=GestureIntensity.NONE,
                gaze_mode=GazeMode.RESPECTFUL_NEUTRAL,
                facial_affect=FacialAffect.CONCERNED,
                interruptible=True,
                max_segment_seconds=8,
                reason_codes=["RISK_AVATAR_OVERRIDE", "CRISIS_LOW_STIMULATION"],
            )
        if directive.response_strategy is ResponseStrategy.DETERMINISTIC_ABSTAIN:
            return AvatarResponsePlan(
                style="neutral_listening",
                speech_rate=0.94,
                initial_pause_ms=250,
                sentence_pause_ms=450,
                gesture_intensity=GestureIntensity.NONE,
                gaze_mode=GazeMode.RESPECTFUL_NEUTRAL,
                facial_affect=FacialAffect.NEUTRAL,
                interruptible=True,
                max_segment_seconds=9,
                reason_codes=["EVIDENCE_ABSTENTION_NEUTRAL"],
            )
        if directive.risk_level is RiskLevel.AMBER:
            return AvatarResponsePlan(
                style="concerned_calm",
                speech_rate=0.92,
                initial_pause_ms=300,
                sentence_pause_ms=500,
                gesture_intensity=GestureIntensity.LOW,
                gaze_mode=GazeMode.RESPECTFUL_NEUTRAL,
                facial_affect=FacialAffect.CONCERNED,
                interruptible=True,
                max_segment_seconds=9,
                reason_codes=["AMBER_CALMING_STYLE"],
            )
        if (
            directive.intent is AgentIntent.COPING_EXERCISE
            or response.support_mode == "exercise"
        ):
            return AvatarResponsePlan(
                style="gentle_guidance",
                speech_rate=0.86,
                initial_pause_ms=300,
                sentence_pause_ms=650,
                gesture_intensity=GestureIntensity.LOW,
                gaze_mode=GazeMode.GUIDED,
                facial_affect=FacialAffect.WARM,
                interruptible=True,
                max_segment_seconds=8,
                reason_codes=["GUIDED_EXERCISE_PACING"],
            )
        if response.support_mode == "educate":
            return AvatarResponsePlan(
                style="neutral_listening",
                speech_rate=1.0,
                initial_pause_ms=150,
                sentence_pause_ms=350,
                gesture_intensity=GestureIntensity.LOW,
                gaze_mode=GazeMode.SOFT_DIRECT,
                facial_affect=FacialAffect.NEUTRAL,
                interruptible=True,
                max_segment_seconds=10,
                reason_codes=["EDUCATION_NEUTRAL_STYLE"],
            )
        return AvatarResponsePlan(
            style="warm",
            speech_rate=0.96,
            initial_pause_ms=200,
            sentence_pause_ms=400,
            gesture_intensity=GestureIntensity.LOW,
            gaze_mode=GazeMode.SOFT_DIRECT,
            facial_affect=FacialAffect.WARM,
            interruptible=True,
            max_segment_seconds=9,
            reason_codes=["SUPPORTIVE_LISTENING_STYLE"],
        )
