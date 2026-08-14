import json
from pathlib import Path

from pydantic import BaseModel

from app.memory.extraction import MemoryExtractor
from app.memory.models import (
    MemoryAspect,
    MemoryCandidate,
    MemoryDecision,
    MemoryMessage,
    MemoryWindow,
    MessageRole,
)
from app.memory.policy import MemoryPolicy
from app.memory.windowing import estimate_tokens


class MemoryEvalCase(BaseModel):
    case_id: str
    text: str
    expected_aspect: MemoryAspect | None
    expected_decision: MemoryDecision | str


class MemoryEvalReport(BaseModel):
    case_count: int
    extraction_precision: float
    extraction_recall: float
    aspect_accuracy: float
    policy_accuracy: float
    forbidden_write_rate: float
    failed_case_ids: list[str]


def load_cases(path: Path) -> list[MemoryEvalCase]:
    return [
        MemoryEvalCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class MemoryEvaluator:
    def __init__(
        self,
        *,
        extractor: MemoryExtractor,
        policy: MemoryPolicy | None = None,
        batch_windows: int = 8,
    ) -> None:
        if batch_windows <= 0:
            raise ValueError("batch_windows must be positive")
        self._extractor = extractor
        self._policy = policy or MemoryPolicy()
        self._batch_windows = batch_windows

    def evaluate(self, cases: list[MemoryEvalCase]) -> MemoryEvalReport:
        true_positive = 0
        predicted_positive = 0
        gold_positive = 0
        correct_aspect = 0
        correct_policy = 0
        forbidden_cases = 0
        forbidden_writes = 0
        failures: list[str] = []
        windows: list[MemoryWindow] = []
        for index, case in enumerate(cases):
            windows.append(
                MemoryWindow(
                    window_id=f"eval_window_{index}",
                    messages=[
                        MemoryMessage(
                            message_id=f"eval_message_{index}",
                            turn_id=f"eval_turn_{index}",
                            role=MessageRole.USER,
                            text=case.text,
                            sequence=index,
                            timestamp_ms=index,
                        )
                    ],
                    estimated_tokens=estimate_tokens(case.text),
                )
            )
        extracted: list[list[MemoryCandidate]] = []
        for offset in range(0, len(windows), self._batch_windows):
            extracted.extend(
                self._extractor.extract_batch(
                    windows[offset : offset + self._batch_windows]
                )
            )
        if len(extracted) != len(cases):
            raise ValueError("extractor result count does not match evaluation windows")

        for case, candidates in zip(cases, extracted, strict=True):
            candidate = candidates[0] if candidates else None
            expected_decision = (
                case.expected_decision.value
                if isinstance(case.expected_decision, MemoryDecision)
                else case.expected_decision
            )
            expected_positive = (
                case.expected_aspect is not None
                and expected_decision != MemoryDecision.REJECT.value
            )
            actual_decision: str = "NONE"
            if candidate is not None:
                actual_decision = self._policy.evaluate(
                    candidate,
                    consent_granted=False,
                ).value
            predicted_is_persistable = (
                candidate is not None and actual_decision != MemoryDecision.REJECT.value
            )
            predicted_positive += int(predicted_is_persistable)
            gold_positive += int(expected_positive)
            if predicted_is_persistable and expected_positive:
                true_positive += 1
                assert candidate is not None
                if candidate.aspect is case.expected_aspect:
                    correct_aspect += 1

            policy_ok = actual_decision == expected_decision
            if expected_decision == MemoryDecision.REJECT.value:
                policy_ok = actual_decision in {
                    "NONE",
                    MemoryDecision.REJECT.value,
                }
            if policy_ok:
                correct_policy += 1
            if expected_decision == MemoryDecision.REJECT.value:
                forbidden_cases += 1
                forbidden_writes += int(
                    actual_decision
                    not in {"NONE", MemoryDecision.REJECT.value}
                )

            aspect_ok = not predicted_is_persistable and not expected_positive
            if candidate is not None and predicted_is_persistable and expected_positive:
                aspect_ok = candidate.aspect is case.expected_aspect
            if not aspect_ok or not policy_ok:
                failures.append(case.case_id)

        return MemoryEvalReport(
            case_count=len(cases),
            extraction_precision=(
                true_positive / predicted_positive if predicted_positive else 1.0
            ),
            extraction_recall=true_positive / gold_positive if gold_positive else 1.0,
            aspect_accuracy=correct_aspect / gold_positive if gold_positive else 1.0,
            policy_accuracy=correct_policy / len(cases) if cases else 1.0,
            forbidden_write_rate=(
                forbidden_writes / forbidden_cases if forbidden_cases else 0.0
            ),
            failed_case_ids=failures,
        )


def render_report(report: MemoryEvalReport) -> str:
    return json.dumps(report.model_dump(), ensure_ascii=False, indent=2)
