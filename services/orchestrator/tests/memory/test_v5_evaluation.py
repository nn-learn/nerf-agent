from pathlib import Path

from app.memory.v5_evaluation import MemoryV5Evaluator, load_memory_v5_cases
from app.memory.v5_release_evaluation import MemoryV5ReleaseEvaluator

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _report():
    cases = load_memory_v5_cases(PROJECT_ROOT / "evals" / "memory_v5_cases.jsonl")
    return MemoryV5Evaluator().evaluate(cases)


def test_memory_v5_fixture_passes_all_governance_contracts() -> None:
    report = _report()
    metrics = report.metrics

    assert report.case_count == 20
    assert report.failed_case_ids == []
    assert metrics.case_pass_rate == 1
    assert metrics.forgetting_aware_memory_accuracy == 1
    assert metrics.stale_memory_utilization_rate == 0
    assert metrics.counterfactual_memory_benefit_rate == 1
    assert metrics.cross_user_leakage_rate == 0
    assert metrics.unsolicited_sensitive_recall_rate == 0
    assert metrics.historical_crisis_adoption_rate == 0
    assert metrics.paused_memory_bypass_rate == 0
    assert metrics.forbidden_use_bypass_rate == 0
    assert metrics.deletion_completeness == 1
    assert metrics.provenance_completeness == 1
    assert metrics.p95_control_latency_ms <= 100


def test_memory_v5_release_gate_allows_demo_but_not_production() -> None:
    release = MemoryV5ReleaseEvaluator().evaluate(_report())

    assert release.release_claim == "ENGINEERING_DEMO_COMPLETE"
    assert release.engineering_complete is True
    assert release.demo_ready is True
    assert release.production_ready is False
    assert release.engineering_blockers == []
    assert "clinical safety signoff is missing" in release.external_validation_blockers


def test_memory_v5_release_gate_fails_closed_on_stale_or_privacy_regression() -> None:
    report = _report()
    report.metrics.stale_memory_utilization_rate = 0.01
    report.metrics.cross_user_leakage_rate = 0.01

    release = MemoryV5ReleaseEvaluator().evaluate(report)

    assert release.engineering_complete is False
    assert "stale memory was utilized" in release.engineering_blockers
    assert "cross-user memory leaked" in release.engineering_blockers
