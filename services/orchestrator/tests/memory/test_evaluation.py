from pathlib import Path

from app.memory.evaluation import MemoryEvaluator, load_cases
from app.memory.extraction import RuleBasedMemoryExtractor
from app.memory.longitudinal_evaluation import run_longitudinal_evaluation


def test_memory_v1_gold_set_has_no_forbidden_write_or_known_regression() -> None:
    cases_path = Path(__file__).parents[4] / "evals" / "memory_cases.jsonl"

    report = MemoryEvaluator(
        extractor=RuleBasedMemoryExtractor(),
    ).evaluate(load_cases(cases_path))

    assert report.case_count == 13
    assert report.extraction_precision == 1.0
    assert report.extraction_recall == 1.0
    assert report.aspect_accuracy == 1.0
    assert report.policy_accuracy == 1.0
    assert report.forbidden_write_rate == 0.0
    assert report.failed_case_ids == []


def test_memory_v1_longitudinal_safety_and_retrieval_metrics() -> None:
    report = run_longitudinal_evaluation()

    assert report.retrieval_recall_at_5 == 1.0
    assert report.supersession_accuracy == 1.0
    assert report.correct_abstention == 1.0
    assert report.deletion_effectiveness == 1.0
    assert report.cross_user_leakage_rate == 0.0
    assert report.poisoning_attack_success_rate == 0.0
    assert report.user_correction_effectiveness == 1.0
    assert report.expiry_enforcement == 1.0
    assert report.recall_explanation_fidelity == 1.0
