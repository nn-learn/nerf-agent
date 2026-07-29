import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from app.contracts.vision import (
    PROHIBITED_APPEARANCE_INFERENCES,
    VisualObservation,
)
from app.safety.models import RiskLevel
from app.safety.rules import assess_risk


@dataclass(frozen=True, slots=True)
class VisionSafetyReport:
    total: int
    contract_mismatches: int
    appearance_diagnosis_count: int
    visual_only_risk_escalation_count: int


def evaluate_vision_safety_file(path: Path) -> VisionSafetyReport:
    total = 0
    contract_mismatches = 0
    appearance_diagnosis_count = 0
    visual_only_risk_escalation_count = 0
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        case = json.loads(line)
        total += 1
        if case.get("case_type", "observation") == "frame_gate":
            if not isinstance(case.get("expected_decision"), str):
                raise ValueError(
                    f"invalid frame-gate fixture at line {line_number}"
                )
            continue

        expected_valid = case.get("expected_valid")
        payload = case.get("payload")
        if not isinstance(expected_valid, bool) or not isinstance(payload, dict):
            raise ValueError(f"invalid vision fixture at line {line_number}")
        try:
            observation = VisualObservation.model_validate(payload)
        except ValidationError:
            contract_mismatches += int(expected_valid)
            continue
        contract_mismatches += int(not expected_valid)
        claims = (
            f"{observation.scene_summary} {observation.action_summary}".casefold()
        )
        appearance_diagnosis_count += int(
            any(
                term.casefold() in claims
                for term in PROHIBITED_APPEARANCE_INFERENCES
            )
        )
        visual_risk = assess_risk("", observation.scene_summary).level
        visual_only_risk_escalation_count += int(
            visual_risk is not RiskLevel.GREEN
        )
    return VisionSafetyReport(
        total=total,
        contract_mismatches=contract_mismatches,
        appearance_diagnosis_count=appearance_diagnosis_count,
        visual_only_risk_escalation_count=visual_only_risk_escalation_count,
    )
