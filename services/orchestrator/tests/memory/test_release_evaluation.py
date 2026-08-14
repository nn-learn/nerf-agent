from dataclasses import replace
from types import SimpleNamespace

from app.evals.privacy import (
    DeletionProbeReport,
    MemoryPrivacyAuditReport,
    PrivacyScanReport,
)
from app.memory.evaluation_protocol import DatasetStatus
from app.memory.release_evaluation import MemoryV2ReleaseEvaluator


def _privacy_report() -> MemoryPrivacyAuditReport:
    return MemoryPrivacyAuditReport(
        media=PrivacyScanReport(0, 0, 0, 0),
        deletion_probe=DeletionProbeReport(1, 0, ()),
        observation_owner_mismatch_count=0,
        profile_evidence_owner_mismatch_count=0,
        episode_member_owner_mismatch_count=0,
        retrieval_owner_mismatch_count=0,
        shadow_ranking_owner_mismatch_count=0,
        orphan_observation_count=0,
        orphan_profile_evidence_count=0,
        orphan_episode_member_count=0,
        orphan_retrieval_count=0,
        orphan_shadow_ranking_count=0,
        profile_evidence_link_mismatch_count=0,
        ineligible_episode_member_count=0,
        ineligible_active_profile_evidence_count=0,
    )


def _strategy_report():
    arm = SimpleNamespace(
        answer_stage_evaluated_count=6,
        final_answer_adherence_rate=1.0,
        false_memory_adoption_rate=0.0,
        evidence_sufficiency_rate=0.833,
        retrieval=SimpleNamespace(forbidden_retrieval_rate=0.0),
    )
    return SimpleNamespace(
        recommended_offline_arm="bge-m3-hybrid",
        arms={"bge-m3-hybrid": arm},
        query_count=6,
        dataset_status=DatasetStatus.ENGINEERING_FIXTURE,
        annotation_audit=None,
        production_promotion_ready=False,
    )


def test_release_gate_can_close_v2_engineering_without_overclaiming_production() -> None:
    report = MemoryV2ReleaseEvaluator().evaluate(
        strategy_report=_strategy_report(),
        stress_report=SimpleNamespace(engineering_stress_passed=True),
        privacy_report=_privacy_report(),
        privacy_audit_scope="SYNTHETIC_SCHEMA",
    )

    assert report.engineering_complete is True
    assert report.release_claim == "ENGINEERING_BASELINE_COMPLETE"
    assert report.automated_answer_gate_passed is True
    assert report.graph_research_ready is False
    assert report.production_ready is False
    assert "clinical safety signoff is missing" in report.external_validation_blockers


def test_release_gate_fails_engineering_on_privacy_or_answer_regression() -> None:
    unsafe_privacy = replace(
        _privacy_report(),
        retrieval_owner_mismatch_count=1,
    )
    strategy = _strategy_report()
    strategy.arms["bge-m3-hybrid"].final_answer_adherence_rate = 0.50

    report = MemoryV2ReleaseEvaluator().evaluate(
        strategy_report=strategy,
        stress_report=SimpleNamespace(engineering_stress_passed=True),
        privacy_report=unsafe_privacy,
        privacy_audit_scope="DEPLOYMENT_DATABASE",
    )

    assert report.engineering_complete is False
    assert "memory privacy invariant audit failed" in report.engineering_blockers
    assert "automated final-answer safety gate failed" in report.engineering_blockers
