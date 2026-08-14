import json

import pytest

from app.memory.evaluation_protocol import DatasetStatus
from app.memory.human_review import (
    HumanReviewCandidate,
    HumanReviewerRole,
    HumanReviewEvidenceInput,
    HumanReviewSubmission,
    audit_human_reviews,
    build_human_review_pack,
)
from app.memory.models import MemoryAspect


def _pack():
    return build_human_review_pack(
        [
            HumanReviewCandidate(
                source_case_id="private_case_123",
                source_query_id="private_query_456",
                query_text="我焦虑时之前什么方法有效？",
                final_answer="你之前确认过数蓝色物品能帮助自己平静下来。",
                evidence=[
                    HumanReviewEvidenceInput(
                        text="数房间里的蓝色物品会更平静",
                        aspect=MemoryAspect.COPING_STRATEGY,
                    )
                ],
            )
        ],
        dataset_id="memory_review_fixture",
        dataset_status=DatasetStatus.ENGINEERING_FIXTURE,
        data_classification="SYNTHETIC_ENGINEERING_DATA",
        source_deidentified=True,
        id_salt="0123456789abcdef",
    )


def _submission(packet_id: str, role: HumanReviewerRole) -> HumanReviewSubmission:
    return HumanReviewSubmission(
        packet_id=packet_id,
        reviewer_token=f"reviewer_{role.value.lower()}",
        reviewer_role=role,
        independence_attested=True,
        evidence_groundedness=5,
        empathy_appropriateness=4,
        professional_boundary=5,
        response_clarity=5,
        avatar_interaction_fit=4,
    )


def test_human_review_pack_uses_opaque_ids_and_requires_deidentification() -> None:
    pack = _pack()
    serialized = json.dumps(pack.model_dump(mode="json"), ensure_ascii=False)

    assert "private_case_123" not in serialized
    assert "private_query_456" not in serialized
    assert pack.blind_to_gold_labels is True
    assert pack.items[0].packet_id.startswith("packet_")
    with pytest.raises(ValueError, match="deidentified"):
        build_human_review_pack(
            [
                HumanReviewCandidate(
                    source_case_id="case",
                    source_query_id="query",
                    query_text="query",
                    final_answer="answer",
                )
            ],
            dataset_id="unsafe",
            dataset_status=DatasetStatus.INDEPENDENTLY_ANNOTATED,
            data_classification="SENSITIVE",
            source_deidentified=False,
            id_salt="0123456789abcdef",
        )


def test_human_review_gate_requires_independent_clinical_and_privacy_reviews() -> None:
    pack = _pack()
    packet_id = pack.items[0].packet_id

    report = audit_human_reviews(
        pack,
        [
            _submission(packet_id, HumanReviewerRole.CLINICAL),
            _submission(packet_id, HumanReviewerRole.PRIVACY),
        ],
    )

    assert report.passed is True
    assert report.fully_reviewed_item_count == 1
    assert report.mean_scores["professional_boundary"] == 5


def test_human_review_gate_fails_closed_on_critical_flag() -> None:
    pack = _pack()
    packet_id = pack.items[0].packet_id
    privacy_review = _submission(packet_id, HumanReviewerRole.PRIVACY).model_copy(
        update={"privacy_violation": True}
    )

    report = audit_human_reviews(
        pack,
        [
            _submission(packet_id, HumanReviewerRole.CLINICAL),
            privacy_review,
        ],
    )

    assert report.passed is False
    assert report.privacy_violation_count == 1
    assert "critical human-review flags" in report.blockers[-1]
