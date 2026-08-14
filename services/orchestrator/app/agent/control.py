from collections.abc import Iterable

from app.agent.models import (
    AgentDecisionTrace,
    AgentDirective,
    AgentIntent,
    EvidenceRequirement,
    IntentAssessment,
    MemoryAccessMode,
    ResponseStrategy,
)
from app.safety.models import AgentResponse, RiskAssessment, RiskLevel


class AgentPolicyViolation(ValueError):
    pass


class AgentControlPlane:
    """Deterministic policy above model, memory, RAG, and tool execution."""

    _memory_control_terms = (
        "删除记忆",
        "忘掉这条",
        "忘记这条",
        "导出记忆",
        "查看记忆",
        "不要记住",
        "别再记住",
        "delete my memory",
        "export my memory",
        "forget this",
    )
    _handoff_terms = (
        "转人工",
        "真人咨询师",
        "联系咨询师",
        "联系医生",
        "找医生",
        "人工接管",
        "human counselor",
        "clinician",
    )
    _memory_recall_terms = (
        "你记得",
        "还记得我",
        "我之前说过",
        "之前对我有效",
        "我的偏好",
        "what do you remember",
        "do you remember",
    )
    _exercise_terms = (
        "带我呼吸",
        "呼吸练习",
        "放松练习",
        "现在怎么平静",
        "帮我冷静",
        "grounding exercise",
        "breathing exercise",
    )
    _education_terms = (
        "什么是",
        "为什么会",
        "科普",
        "原理",
        "有哪些表现",
        "如何改善睡眠",
        "what is",
        "why does",
        "psychoeducation",
    )
    _disclosure_terms = (
        "压力",
        "难过",
        "焦虑",
        "害怕",
        "孤独",
        "睡不着",
        "撑不住",
        "心情",
        "stress",
        "anxious",
        "sad",
        "lonely",
    )

    def classify(self, transcript: str) -> IntentAssessment:
        normalized = transcript.strip().casefold()
        ordered: tuple[tuple[AgentIntent, tuple[str, ...], str], ...] = (
            (
                AgentIntent.MEMORY_CONTROL,
                self._memory_control_terms,
                "MEMORY_CONTROL_LANGUAGE",
            ),
            (
                AgentIntent.CLINICIAN_HANDOFF,
                self._handoff_terms,
                "HUMAN_HANDOFF_LANGUAGE",
            ),
            (
                AgentIntent.MEMORY_RECALL,
                self._memory_recall_terms,
                "MEMORY_RECALL_LANGUAGE",
            ),
            (
                AgentIntent.COPING_EXERCISE,
                self._exercise_terms,
                "COPING_EXERCISE_LANGUAGE",
            ),
            (
                AgentIntent.PSYCHOEDUCATION,
                self._education_terms,
                "EDUCATION_LANGUAGE",
            ),
            (
                AgentIntent.EMOTIONAL_DISCLOSURE,
                self._disclosure_terms,
                "EMOTIONAL_DISCLOSURE_LANGUAGE",
            ),
        )
        for intent, terms, reason in ordered:
            if self._contains_any(normalized, terms):
                return IntentAssessment(
                    intent=intent,
                    confidence=0.95,
                    reason_codes=[reason],
                )
        return IntentAssessment(
            intent=AgentIntent.GENERAL_SUPPORT,
            confidence=0.70,
            reason_codes=["DEFAULT_SUPPORT_INTENT"],
        )

    def draft(
        self,
        transcript: str,
        risk: RiskAssessment,
    ) -> AgentDirective:
        assessment = self.classify(transcript)
        if risk.level in {RiskLevel.RED, RiskLevel.EMERGENCY}:
            return AgentDirective(
                intent=assessment.intent,
                risk_level=risk.level,
                response_strategy=ResponseStrategy.DETERMINISTIC_CRISIS,
                evidence_requirement=EvidenceRequirement.NONE,
                memory_access=MemoryAccessMode.FORBIDDEN,
                allowed_capabilities=[
                    "request_clinician_handoff",
                    "draft_emergency_contact_message",
                    "show_reviewed_resource",
                ],
                max_tool_proposals=1,
                reason_codes=[
                    "RISK_OVERRIDE",
                    *assessment.reason_codes,
                    *risk.reasons,
                ],
            )
        return self._normal_directive(assessment, risk.level)

    def finalize(
        self,
        directive: AgentDirective,
        context: dict[str, object],
    ) -> tuple[AgentDirective, AgentDecisionTrace]:
        reviewed = context.get("reviewed_evidence")
        memory = context.get("long_term_memory")
        reviewed_available = bool(reviewed) if isinstance(reviewed, list) else False
        memory_available = bool(memory) if isinstance(memory, list) else False
        degraded = bool(context.get("memory_retrieval_degraded", False))
        final = directive.model_copy(deep=True)
        reasons = list(final.reason_codes)
        missing_required = (
            final.evidence_requirement is EvidenceRequirement.REVIEWED_KNOWLEDGE
            and not reviewed_available
        ) or (
            final.evidence_requirement is EvidenceRequirement.CONFIRMED_MEMORY
            and (not memory_available or degraded)
        )
        if missing_required:
            reasons.append("REQUIRED_CONTEXT_UNAVAILABLE")
            final = final.model_copy(
                update={
                    "response_strategy": ResponseStrategy.DETERMINISTIC_ABSTAIN,
                    "allowed_capabilities": [],
                    "max_tool_proposals": 0,
                    "reason_codes": reasons,
                }
            )
        trace = AgentDecisionTrace(
            policy_version=final.policy_version,
            intent=final.intent,
            risk_level=final.risk_level,
            response_strategy=final.response_strategy,
            evidence_requirement=final.evidence_requirement,
            memory_access=final.memory_access,
            allowed_capabilities=final.allowed_capabilities,
            max_tool_proposals=final.max_tool_proposals,
            reviewed_evidence_available=reviewed_available,
            confirmed_memory_available=memory_available,
            retrieval_degraded=degraded,
            reason_codes=final.reason_codes,
        )
        return final, trace

    def validate_response(
        self,
        response: AgentResponse,
        directive: AgentDirective,
    ) -> AgentResponse:
        if response.risk_level is not directive.risk_level:
            raise AgentPolicyViolation("response changed deterministic risk level")
        if (
            directive.response_strategy is ResponseStrategy.DETERMINISTIC_CRISIS
            and response.support_mode != "handoff"
        ):
            raise AgentPolicyViolation("crisis response must use handoff support mode")
        if len(response.action_proposals) > directive.max_tool_proposals:
            raise AgentPolicyViolation("response exceeded tool proposal budget")
        return response

    def _normal_directive(
        self,
        assessment: IntentAssessment,
        risk_level: RiskLevel,
    ) -> AgentDirective:
        if assessment.intent is AgentIntent.MEMORY_CONTROL:
            return AgentDirective(
                intent=assessment.intent,
                risk_level=risk_level,
                reason_codes=assessment.reason_codes,
                response_strategy=ResponseStrategy.MODEL_CAPABILITY_PROPOSAL,
                evidence_requirement=EvidenceRequirement.NONE,
                memory_access=MemoryAccessMode.FORBIDDEN,
                allowed_capabilities=["export_user_memories", "delete_user_memory"],
                max_tool_proposals=1,
            )
        if assessment.intent is AgentIntent.CLINICIAN_HANDOFF:
            return AgentDirective(
                intent=assessment.intent,
                risk_level=risk_level,
                reason_codes=assessment.reason_codes,
                response_strategy=ResponseStrategy.MODEL_CAPABILITY_PROPOSAL,
                evidence_requirement=EvidenceRequirement.NONE,
                memory_access=MemoryAccessMode.FORBIDDEN,
                allowed_capabilities=["request_clinician_handoff"],
                max_tool_proposals=1,
            )
        if assessment.intent is AgentIntent.MEMORY_RECALL:
            return AgentDirective(
                intent=assessment.intent,
                risk_level=risk_level,
                reason_codes=assessment.reason_codes,
                response_strategy=ResponseStrategy.MODEL_SUPPORT,
                evidence_requirement=EvidenceRequirement.CONFIRMED_MEMORY,
                memory_access=MemoryAccessMode.REQUIRED,
            )
        if assessment.intent is AgentIntent.COPING_EXERCISE:
            return AgentDirective(
                intent=assessment.intent,
                risk_level=risk_level,
                reason_codes=assessment.reason_codes,
                response_strategy=ResponseStrategy.MODEL_SUPPORT,
                evidence_requirement=EvidenceRequirement.NONE,
                memory_access=MemoryAccessMode.OPTIONAL,
                allowed_capabilities=[
                    "start_breathing_exercise",
                    "show_reviewed_resource",
                ],
                max_tool_proposals=1,
            )
        if assessment.intent is AgentIntent.PSYCHOEDUCATION:
            return AgentDirective(
                intent=assessment.intent,
                risk_level=risk_level,
                reason_codes=assessment.reason_codes,
                response_strategy=ResponseStrategy.MODEL_EDUCATION,
                evidence_requirement=EvidenceRequirement.REVIEWED_KNOWLEDGE,
                memory_access=MemoryAccessMode.FORBIDDEN,
                allowed_capabilities=["show_reviewed_resource"],
                max_tool_proposals=1,
            )
        return AgentDirective(
            intent=assessment.intent,
            risk_level=risk_level,
            reason_codes=assessment.reason_codes,
            response_strategy=ResponseStrategy.MODEL_SUPPORT,
            evidence_requirement=EvidenceRequirement.NONE,
            memory_access=MemoryAccessMode.OPTIONAL,
        )

    @staticmethod
    def _contains_any(text: str, terms: Iterable[str]) -> bool:
        return any(term.casefold() in text for term in terms)
