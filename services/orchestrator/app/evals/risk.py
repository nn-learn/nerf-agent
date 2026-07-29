import json
from dataclasses import dataclass
from pathlib import Path

from app.safety.models import RiskLevel
from app.safety.rules import assess_risk

HIGH_RISK_LEVELS = {RiskLevel.RED, RiskLevel.EMERGENCY}


@dataclass(frozen=True, slots=True)
class RiskEvalReport:
    total: int
    correct: int
    expected_high_risk: int
    recalled_high_risk: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def high_risk_recall(self) -> float:
        if self.expected_high_risk == 0:
            return 0.0
        return self.recalled_high_risk / self.expected_high_risk


def evaluate_risk_file(path: Path) -> RiskEvalReport:
    total = 0
    correct = 0
    expected_high_risk = 0
    recalled_high_risk = 0
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        case = json.loads(line)
        text = case.get("text")
        expected_raw = case.get("expected_level")
        if not isinstance(text, str) or not isinstance(expected_raw, str):
            raise ValueError(f"invalid risk fixture at line {line_number}")
        expected = RiskLevel(expected_raw)
        predicted = assess_risk(text, "").level
        total += 1
        correct += int(predicted is expected)
        if expected in HIGH_RISK_LEVELS:
            expected_high_risk += 1
            recalled_high_risk += int(predicted in HIGH_RISK_LEVELS)
    return RiskEvalReport(
        total=total,
        correct=correct,
        expected_high_risk=expected_high_risk,
        recalled_high_risk=recalled_high_risk,
    )
