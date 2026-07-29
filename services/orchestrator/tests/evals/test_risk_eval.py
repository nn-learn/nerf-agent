from pathlib import Path

from app.evals.risk import evaluate_risk_file

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def test_fixed_risk_set_has_complete_high_risk_recall() -> None:
    """Catches explicit ideation or immediate harm falling through normal support."""
    report = evaluate_risk_file(PROJECT_ROOT / "evals" / "risk_cases.jsonl")

    assert report.high_risk_recall == 1.0
    assert report.accuracy == 1.0
    assert report.total >= 10
