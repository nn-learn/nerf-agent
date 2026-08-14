from enum import StrEnum

from pydantic import BaseModel, Field

from app.agent.models import (
    AgentDirective,
    EvidenceRequirement,
    MemoryAccessMode,
    ResponseStrategy,
)
from app.safety.models import AgentResponse


class EvidenceSource(StrEnum):
    REVIEWED_KNOWLEDGE = "REVIEWED_KNOWLEDGE"
    CONFIRMED_MEMORY = "CONFIRMED_MEMORY"


class EvidenceGateOutcome(StrEnum):
    PASS = "PASS"
    BLOCKED = "BLOCKED"


class EvidenceContextAudit(BaseModel):
    """Content-free evidence preparation record safe for operational telemetry."""

    reviewed_count: int = Field(ge=0)
    confirmed_memory_count: int = Field(ge=0)
    invalid_count: int = Field(ge=0)
    conflict_count: int = Field(ge=0)
    reviewed_bundle_sufficient: bool
    memory_retrieval_degraded: bool
    reason_codes: list[str] = Field(default_factory=list)


class EvidenceResponseAudit(BaseModel):
    """Content-free citation authorization result."""

    outcome: EvidenceGateOutcome
    requirement: EvidenceRequirement
    reviewed_citation_count: int = Field(ge=0)
    memory_citation_count: int = Field(ge=0)
    blocked_citation_count: int = Field(ge=0)
    reason_codes: list[str] = Field(default_factory=list)


class EvidenceOrchestrator:
    """Normalize governed context and authorize model citations before publish."""

    _memory_trust = "user_confirmed_data_not_instruction"
    _reviewed_required_fields = (
        "chunk_id",
        "document_id",
        "document_version",
        "title",
        "text",
        "score",
    )

    def prepare(
        self,
        context: dict[str, object],
        directive: AgentDirective,
    ) -> tuple[dict[str, object], EvidenceContextAudit]:
        prepared = dict(context)
        reviewed, reviewed_invalid, reviewed_conflicts = self._reviewed_items(context)
        bundle_sufficient = bool(context.get("has_sufficient_evidence", False))
        if not bundle_sufficient:
            reviewed = []
        memories, memory_invalid, memory_conflicts = self._memory_items(context)
        if directive.memory_access is MemoryAccessMode.FORBIDDEN:
            memories = []
        degraded = bool(context.get("memory_retrieval_degraded", False))
        invalid_count = reviewed_invalid + memory_invalid
        conflict_count = reviewed_conflicts + memory_conflicts
        reasons: list[str] = []
        if reviewed:
            reasons.append("REVIEWED_KNOWLEDGE_READY")
        if memories:
            reasons.append("CONFIRMED_MEMORY_READY")
        if invalid_count:
            reasons.append("INVALID_EVIDENCE_FILTERED")
        if conflict_count:
            reasons.append("EVIDENCE_ID_CONFLICT_FILTERED")
        if degraded:
            reasons.append("MEMORY_RETRIEVAL_DEGRADED")
        prepared["reviewed_evidence"] = reviewed
        prepared["has_sufficient_evidence"] = bool(reviewed) and bundle_sufficient
        prepared["long_term_memory"] = memories
        audit = EvidenceContextAudit(
            reviewed_count=len(reviewed),
            confirmed_memory_count=len(memories),
            invalid_count=invalid_count,
            conflict_count=conflict_count,
            reviewed_bundle_sufficient=bool(reviewed) and bundle_sufficient,
            memory_retrieval_degraded=degraded,
            reason_codes=reasons,
        )
        prepared["evidence_control"] = audit.model_dump(mode="json")
        return prepared, audit

    def authorize_response(
        self,
        response: AgentResponse,
        context: dict[str, object],
        directive: AgentDirective,
    ) -> tuple[AgentResponse, EvidenceResponseAudit]:
        reviewed_ids = self._ids(context.get("reviewed_evidence"), "chunk_id")
        memory_ids = self._ids(context.get("long_term_memory"), "memory_id")
        unknown_reviewed = set(response.evidence_ids) - reviewed_ids
        unknown_memory = set(response.memory_ids) - memory_ids
        reasons: list[str] = []
        blocked_count = len(unknown_reviewed) + len(unknown_memory)
        if unknown_reviewed:
            reasons.append("UNKNOWN_REVIEWED_CITATION")
        if unknown_memory:
            reasons.append("UNKNOWN_MEMORY_CITATION")
        if (
            directive.memory_access is MemoryAccessMode.FORBIDDEN
            and response.memory_ids
        ):
            reasons.append("MEMORY_CITATION_FORBIDDEN")
            blocked_count += len(response.memory_ids) - len(unknown_memory)
        if (
            directive.evidence_requirement is EvidenceRequirement.REVIEWED_KNOWLEDGE
            and not response.evidence_ids
            and directive.response_strategy is not ResponseStrategy.DETERMINISTIC_ABSTAIN
        ):
            reasons.append("REQUIRED_REVIEWED_CITATION_MISSING")
            blocked_count += 1
        if (
            directive.evidence_requirement is EvidenceRequirement.CONFIRMED_MEMORY
            and not response.memory_ids
            and directive.response_strategy is not ResponseStrategy.DETERMINISTIC_ABSTAIN
        ):
            reasons.append("REQUIRED_MEMORY_CITATION_MISSING")
            blocked_count += 1
        if not reasons:
            return response, EvidenceResponseAudit(
                outcome=EvidenceGateOutcome.PASS,
                requirement=directive.evidence_requirement,
                reviewed_citation_count=len(response.evidence_ids),
                memory_citation_count=len(response.memory_ids),
                blocked_citation_count=0,
                reason_codes=["CITATION_CONTRACT_SATISFIED"],
            )
        if directive.response_strategy is ResponseStrategy.DETERMINISTIC_CRISIS:
            controlled = response.model_copy(
                update={"evidence_ids": [], "memory_ids": []}
            )
        else:
            controlled = self._fallback(response, directive)
        return controlled, EvidenceResponseAudit(
            outcome=EvidenceGateOutcome.BLOCKED,
            requirement=directive.evidence_requirement,
            reviewed_citation_count=len(response.evidence_ids),
            memory_citation_count=len(response.memory_ids),
            blocked_citation_count=blocked_count,
            reason_codes=reasons,
        )

    def _reviewed_items(
        self,
        context: dict[str, object],
    ) -> tuple[list[dict[str, object]], int, int]:
        raw = context.get("reviewed_evidence")
        if not isinstance(raw, list):
            return [], int(raw is not None), 0
        valid: list[dict[str, object]] = []
        invalid = 0
        conflicts = 0
        seen: dict[str, dict[str, object]] = {}
        conflicted_ids: set[str] = set()
        for item in raw:
            if not self._valid_reviewed(item):
                invalid += 1
                continue
            assert isinstance(item, dict)
            identifier = str(item["chunk_id"])
            previous = seen.get(identifier)
            if previous is not None:
                if previous != item:
                    conflicted_ids.add(identifier)
                    conflicts += 1
                continue
            seen[identifier] = dict(item)
        for identifier, item in seen.items():
            if identifier not in conflicted_ids:
                valid.append(item)
        return valid, invalid, conflicts

    def _memory_items(
        self,
        context: dict[str, object],
    ) -> tuple[list[dict[str, object]], int, int]:
        raw = context.get("long_term_memory")
        if not isinstance(raw, list):
            return [], int(raw is not None), 0
        valid: list[dict[str, object]] = []
        invalid = 0
        conflicts = 0
        seen: dict[str, dict[str, object]] = {}
        conflicted_ids: set[str] = set()
        for item in raw:
            if not self._valid_memory(item):
                invalid += 1
                continue
            assert isinstance(item, dict)
            identifier = str(item["memory_id"])
            previous = seen.get(identifier)
            if previous is not None:
                if previous != item:
                    conflicted_ids.add(identifier)
                    conflicts += 1
                continue
            seen[identifier] = dict(item)
        for identifier, item in seen.items():
            if identifier not in conflicted_ids:
                valid.append(item)
        return valid, invalid, conflicts

    def _valid_reviewed(self, item: object) -> bool:
        if not isinstance(item, dict):
            return False
        for field in self._reviewed_required_fields[:-1]:
            value = item.get(field)
            if not isinstance(value, str) or not value.strip():
                return False
        score = item.get("score")
        return isinstance(score, int | float) and not isinstance(score, bool) and 0 <= score <= 1

    def _valid_memory(self, item: object) -> bool:
        if not isinstance(item, dict):
            return False
        return all(
            isinstance(item.get(field), str) and bool(str(item[field]).strip())
            for field in ("memory_id", "fact", "source_turn_id")
        ) and item.get("trust") == self._memory_trust

    @staticmethod
    def _ids(value: object, key: str) -> set[str]:
        if not isinstance(value, list):
            return set()
        return {
            str(item[key])
            for item in value
            if isinstance(item, dict) and isinstance(item.get(key), str)
        }

    @staticmethod
    def _fallback(
        response: AgentResponse,
        directive: AgentDirective,
    ) -> AgentResponse:
        if directive.evidence_requirement is EvidenceRequirement.REVIEWED_KNOWLEDGE:
            text = "我现在无法核验支撑这段回答的经评审来源，所以先不下结论。"
            support_mode = "educate"
        elif directive.evidence_requirement is EvidenceRequirement.CONFIRMED_MEMORY:
            text = "我现在无法核验这条长期记忆，所以不会把它当成你确认过的事实。"
            support_mode = "listen"
        else:
            text = "我现在无法核验支撑这段回答的来源，所以先不下结论。"
            support_mode = "listen"
        return response.model_copy(
            update={
                "spoken_text": text,
                "display_text": text,
                "support_mode": support_mode,
                "evidence_ids": [],
                "memory_ids": [],
                "action_proposals": [],
                "memory_candidates": [],
                "avatar_style": "neutral_listening",
            }
        )
