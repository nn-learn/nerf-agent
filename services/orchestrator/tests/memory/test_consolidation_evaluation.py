from pathlib import Path

from app.memory.consolidation_evaluation import (
    MemoryConsolidationEvaluator,
    load_consolidation_cases,
)


def test_v2_governance_fixture_passes_without_preconfirmation_leakage() -> None:
    fixture = Path(__file__).parents[4] / "evals" / "memory_v2_cases.jsonl"

    report = MemoryConsolidationEvaluator().evaluate(
        load_consolidation_cases(fixture)
    )

    assert report.case_count == 5
    assert report.profile_proposal_accuracy == 1.0
    assert report.conflict_detection_accuracy == 1.0
    assert report.preconfirmation_leakage_rate == 0.0
    assert report.confirmed_profile_recall_rate == 1.0
    assert report.deletion_cascade_rate == 1.0
    assert report.failed_case_ids == []
