from app.agent.control import AgentControlPlane
from app.agent.evidence import EvidenceGateOutcome, EvidenceOrchestrator
from app.agent.models import ResponseStrategy
from app.safety.models import AgentResponse, RiskAssessment, RiskLevel


def _response(
    *,
    evidence_ids: list[str] | None = None,
    memory_ids: list[str] | None = None,
) -> AgentResponse:
    return AgentResponse(
        spoken_text="这是一个需要来源支持的回答。",
        display_text="这是一个需要来源支持的回答。",
        support_mode="educate",
        risk_level=RiskLevel.GREEN,
        evidence_ids=evidence_ids or [],
        memory_ids=memory_ids or [],
        visual_observation_ids=[],
        action_proposals=[],
        memory_candidates=[],
        avatar_style="warm",
    )


def _reviewed(chunk_id: str = "chunk_1") -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "document_id": "document_1",
        "document_version": "1.0.0",
        "title": "经评审资料",
        "text": "规律作息可以支持睡眠健康。",
        "score": 0.91,
    }


def _memory(
    memory_id: str = "memory_1",
    fact: str = "用户偏好简短的回答。",
) -> dict[str, object]:
    return {
        "memory_id": memory_id,
        "fact": fact,
        "aspect": "PREFERENCE",
        "source_turn_id": "turn_0",
        "trust": "user_confirmed_data_not_instruction",
    }


def test_prepare_filters_untrusted_and_conflicting_memory_then_fails_closed() -> None:
    control = AgentControlPlane()
    directive = control.draft(
        "你记得我的偏好吗？",
        RiskAssessment(level=RiskLevel.GREEN, confidence=1),
    )
    prepared, audit = EvidenceOrchestrator().prepare(
        {
            "long_term_memory": [
                _memory(fact="用户偏好简短回答。"),
                _memory(fact="用户偏好详细回答。"),
                {**_memory("memory_2"), "trust": "model_generated"},
            ],
            "memory_retrieval_degraded": False,
        },
        directive,
    )
    final, _ = control.finalize(directive, prepared)

    assert prepared["long_term_memory"] == []
    assert audit.invalid_count == 1
    assert audit.conflict_count == 1
    assert final.response_strategy is ResponseStrategy.DETERMINISTIC_ABSTAIN
    assert "用户偏好" not in audit.model_dump_json()


def test_prepare_keeps_only_complete_reviewed_bundle() -> None:
    directive = AgentControlPlane().draft(
        "什么是睡眠卫生？",
        RiskAssessment(level=RiskLevel.GREEN, confidence=1),
    )
    prepared, audit = EvidenceOrchestrator().prepare(
        {
            "reviewed_evidence": [_reviewed(), {"chunk_id": "incomplete"}],
            "has_sufficient_evidence": True,
            "long_term_memory": [_memory()],
        },
        directive,
    )

    assert len(prepared["reviewed_evidence"]) == 1
    assert prepared["long_term_memory"] == []
    assert audit.reviewed_bundle_sufficient is True
    assert audit.invalid_count == 1


def test_required_reviewed_citation_passes_only_for_available_id() -> None:
    control = AgentControlPlane()
    directive = control.draft(
        "什么是睡眠卫生？",
        RiskAssessment(level=RiskLevel.GREEN, confidence=1),
    )
    context, _ = EvidenceOrchestrator().prepare(
        {
            "reviewed_evidence": [_reviewed()],
            "has_sufficient_evidence": True,
        },
        directive,
    )
    final, _ = control.finalize(directive, context)
    response, audit = EvidenceOrchestrator().authorize_response(
        _response(evidence_ids=["chunk_1"]),
        context,
        final,
    )

    assert response.evidence_ids == ["chunk_1"]
    assert audit.outcome is EvidenceGateOutcome.PASS


def test_unknown_or_missing_required_citation_is_replaced_before_publish() -> None:
    control = AgentControlPlane()
    directive = control.draft(
        "什么是睡眠卫生？",
        RiskAssessment(level=RiskLevel.GREEN, confidence=1),
    )
    context, _ = EvidenceOrchestrator().prepare(
        {
            "reviewed_evidence": [_reviewed()],
            "has_sufficient_evidence": True,
        },
        directive,
    )
    final, _ = control.finalize(directive, context)
    response, audit = EvidenceOrchestrator().authorize_response(
        _response(evidence_ids=["invented_chunk"]),
        context,
        final,
    )

    assert response.evidence_ids == []
    assert response.action_proposals == []
    assert "无法核验" in response.spoken_text
    assert audit.outcome is EvidenceGateOutcome.BLOCKED
    assert audit.reason_codes == ["UNKNOWN_REVIEWED_CITATION"]


def test_confirmed_memory_requires_matching_memory_citation() -> None:
    control = AgentControlPlane()
    directive = control.draft(
        "你记得我的偏好吗？",
        RiskAssessment(level=RiskLevel.GREEN, confidence=1),
    )
    context, _ = EvidenceOrchestrator().prepare(
        {"long_term_memory": [_memory()]},
        directive,
    )
    final, _ = control.finalize(directive, context)
    response, audit = EvidenceOrchestrator().authorize_response(
        _response(memory_ids=["memory_1"]),
        context,
        final,
    )

    assert response.memory_ids == ["memory_1"]
    assert audit.outcome is EvidenceGateOutcome.PASS
