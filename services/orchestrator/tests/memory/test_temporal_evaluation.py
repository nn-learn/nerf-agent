from pathlib import Path

from app.memory.temporal_evaluation import MemoryTemporalEvaluator, load_temporal_cases


def test_v22_temporal_governance_fixture_passes() -> None:
    fixture = Path(__file__).parents[4] / "evals" / "memory_v22_cases.jsonl"

    report = MemoryTemporalEvaluator().evaluate(load_temporal_cases(fixture))

    assert report.case_count == 6
    assert report.temporal_classification_accuracy == 1.0
    assert report.false_change_rate == 0.0
    assert report.predecision_current_preservation_rate == 1.0
    assert report.valid_observed_time_accuracy == 1.0
    assert report.bitemporal_boundary_accuracy == 1.0
    assert report.revoked_change_apply_rate == 0.0
    assert report.failed_case_ids == []
