from enum import StrEnum

from pydantic import BaseModel

from app.memory.models import (
    MemoryAllowedUse,
    MemoryItem,
    MemorySensitivity,
    MemorySourceType,
)
from app.safety.models import RiskLevel


class MemoryUseDecision(StrEnum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"


class MemoryUseContext(BaseModel):
    requested_use: MemoryAllowedUse = MemoryAllowedUse.RESPONSE_CONTEXT
    user_initiated_recall: bool = True
    explicit_sensitive_revisit: bool = False
    current_risk: RiskLevel = RiskLevel.GREEN
    current_turn_corroborates_crisis: bool = False


class MemoryUseResult(BaseModel):
    decision: MemoryUseDecision
    reason_codes: list[str]


class MemoryUsePolicy:
    """Host-owned authorization after candidate retrieval and before prompting."""

    _UNTRUSTED_PERSONALIZATION_SOURCES = {
        MemorySourceType.EXTERNAL_TOOL,
        MemorySourceType.VISUAL_OBSERVATION,
    }

    def evaluate(
        self,
        item: MemoryItem,
        *,
        context: MemoryUseContext,
    ) -> MemoryUseResult:
        candidate = item.candidate
        if context.requested_use not in candidate.allowed_uses:
            return self._block("USE_NOT_ALLOWED")
        if (
            context.requested_use is MemoryAllowedUse.PERSONALIZATION
            and candidate.source_type in self._UNTRUSTED_PERSONALIZATION_SOURCES
        ):
            return self._block("UNTRUSTED_PERSONALIZATION_SOURCE")
        if candidate.sensitivity is MemorySensitivity.CRISIS_SENSITIVE:
            if context.requested_use is not MemoryAllowedUse.SAFETY_SUPPORT:
                return self._block("CRISIS_MEMORY_PURPOSE_MISMATCH")
            if context.current_risk not in {RiskLevel.RED, RiskLevel.EMERGENCY}:
                return self._block("HISTORICAL_CRISIS_IS_NOT_CURRENT_RISK")
            if not context.current_turn_corroborates_crisis:
                return self._block("CURRENT_TURN_CRISIS_CORROBORATION_REQUIRED")
        if candidate.sensitivity is MemorySensitivity.HEALTH_SENSITIVE:
            if not context.user_initiated_recall:
                return self._block("UNSOLICITED_HEALTH_MEMORY")
            if (
                context.requested_use
                in {
                    MemoryAllowedUse.PERSONALIZATION,
                    MemoryAllowedUse.RESPONSE_CONTEXT,
                }
                and not context.explicit_sensitive_revisit
            ):
                return self._block("EXPLICIT_SENSITIVE_REVISIT_REQUIRED")
        return MemoryUseResult(
            decision=MemoryUseDecision.ALLOW,
            reason_codes=["MEMORY_USE_POLICY_ALLOWED"],
        )

    @staticmethod
    def _block(reason: str) -> MemoryUseResult:
        return MemoryUseResult(
            decision=MemoryUseDecision.BLOCK,
            reason_codes=[reason],
        )
