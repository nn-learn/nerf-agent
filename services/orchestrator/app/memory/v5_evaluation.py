import tempfile
from pathlib import Path
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, Field

from app.memory.evaluation_protocol import DatasetStatus
from app.memory.models import (
    MemoryAllowedUse,
    MemoryCandidate,
    MemoryKind,
    MemorySensitivity,
    MemorySourceType,
)
from app.memory.repository import MemoryRepository
from app.memory.retrieval import GovernedMemoryRetriever
from app.memory.use_policy import MemoryUseContext
from app.safety.models import RiskLevel

MemoryV5CaseType = Literal[
    "current_recall",
    "stale_suppression",
    "cross_user_isolation",
    "unsolicited_health",
    "historical_crisis",
    "pause_enforcement",
    "purpose_enforcement",
    "cascade_deletion",
    "provenance_export",
]


class MemoryV5Case(BaseModel):
    case_id: str = Field(min_length=1)
    case_type: MemoryV5CaseType
    text: str = Field(min_length=1)
    query: str = Field(min_length=1)


class MemoryV5Metrics(BaseModel):
    case_pass_rate: float = Field(ge=0, le=1)
    forgetting_aware_memory_accuracy: float = Field(ge=0, le=1)
    stale_memory_utilization_rate: float = Field(ge=0, le=1)
    counterfactual_memory_benefit_rate: float = Field(ge=0, le=1)
    cross_user_leakage_rate: float = Field(ge=0, le=1)
    unsolicited_sensitive_recall_rate: float = Field(ge=0, le=1)
    historical_crisis_adoption_rate: float = Field(ge=0, le=1)
    paused_memory_bypass_rate: float = Field(ge=0, le=1)
    forbidden_use_bypass_rate: float = Field(ge=0, le=1)
    deletion_completeness: float = Field(ge=0, le=1)
    provenance_completeness: float = Field(ge=0, le=1)
    p95_control_latency_ms: float = Field(ge=0)


class MemoryV5EvaluationReport(BaseModel):
    evaluation_version: str = "memory-v5-eval-1.0"
    dataset_status: DatasetStatus = DatasetStatus.ENGINEERING_FIXTURE
    case_count: int
    metrics: MemoryV5Metrics
    failed_case_ids: list[str]


def load_memory_v5_cases(path: Path) -> list[MemoryV5Case]:
    cases = [
        MemoryV5Case.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    case_ids = [case.case_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("memory V5 case ids must be unique")
    return cases


class MemoryV5Evaluator:
    """Deterministic adversarial evaluation of governed long-term memory."""

    def evaluate(self, cases: list[MemoryV5Case]) -> MemoryV5EvaluationReport:
        if not cases:
            raise ValueError("memory V5 evaluation requires cases")
        passed: dict[str, bool] = {}
        latencies: list[float] = []
        counts: dict[str, list[int]] = {}

        for case in cases:
            result, latency_ms = self._evaluate_case(case)
            latencies.append(latency_ms)
            passed[case.case_id] = result
            bucket = counts.setdefault(case.case_type, [0, 0])
            bucket[0] += int(result)
            bucket[1] += 1

        def rate(case_type: MemoryV5CaseType) -> float:
            successes, total = counts.get(case_type, [0, 0])
            return successes / total if total else 1.0

        current_successes, current_total = counts.get("current_recall", [0, 0])
        stale_successes, stale_total = counts.get("stale_suppression", [0, 0])
        forgetting_total = current_total + stale_total
        fama = (
            (current_successes + stale_successes) / forgetting_total
            if forgetting_total
            else 1.0
        )
        metrics = MemoryV5Metrics(
            case_pass_rate=sum(passed.values()) / len(passed),
            forgetting_aware_memory_accuracy=fama,
            stale_memory_utilization_rate=1 - rate("stale_suppression"),
            counterfactual_memory_benefit_rate=rate("current_recall"),
            cross_user_leakage_rate=1 - rate("cross_user_isolation"),
            unsolicited_sensitive_recall_rate=1 - rate("unsolicited_health"),
            historical_crisis_adoption_rate=1 - rate("historical_crisis"),
            paused_memory_bypass_rate=1 - rate("pause_enforcement"),
            forbidden_use_bypass_rate=1 - rate("purpose_enforcement"),
            deletion_completeness=rate("cascade_deletion"),
            provenance_completeness=rate("provenance_export"),
            p95_control_latency_ms=self._percentile(latencies, 0.95),
        )
        return MemoryV5EvaluationReport(
            case_count=len(cases),
            metrics=metrics,
            failed_case_ids=[case_id for case_id, ok in passed.items() if not ok],
        )

    def _evaluate_case(self, case: MemoryV5Case) -> tuple[bool, float]:
        with tempfile.TemporaryDirectory(prefix="psyavatar-memory-v5-") as directory:
            repository = MemoryRepository(Path(directory) / "memory.sqlite3")
            repository.initialize()
            retriever = GovernedMemoryRetriever(repository, min_score=0)
            now_ms = 10_000

            sensitivity = (
                MemorySensitivity.HEALTH_SENSITIVE
                if case.case_type == "unsolicited_health"
                else MemorySensitivity.CRISIS_SENSITIVE
                if case.case_type == "historical_crisis"
                else MemorySensitivity.GENERAL
            )
            allowed_uses = (
                [MemoryAllowedUse.SAFETY_SUPPORT]
                if case.case_type == "historical_crisis"
                else [MemoryAllowedUse.RESPONSE_CONTEXT]
                if case.case_type in {"unsolicited_health", "purpose_enforcement"}
                else [
                    MemoryAllowedUse.PERSONALIZATION,
                    MemoryAllowedUse.RESPONSE_CONTEXT,
                ]
            )
            owner = "user_2" if case.case_type == "cross_user_isolation" else "user_1"
            candidate = MemoryCandidate(
                source="transcript",
                contains_sensitive_content=sensitivity
                is not MemorySensitivity.GENERAL,
                text=case.text,
                kind=MemoryKind.SEMANTIC,
                source_turn_id=f"turn_{case.case_id}",
                subject_key=f"eval.{case.case_id}",
                user_confirmed=True,
                source_type=MemorySourceType.USER_STATEMENT,
                sensitivity=sensitivity,
                allowed_uses=allowed_uses,
                observed_at_ms=1_000,
                valid_from_ms=900,
                valid_to_ms=(5_000 if case.case_type == "stale_suppression" else None),
            )
            item = repository.add_active(
                user_id=owner,
                candidate=candidate,
                now_ms=1_000,
            )
            started = perf_counter()

            if case.case_type == "current_recall":
                recalled = retriever.retrieve(
                    case.query,
                    user_id="user_1",
                    as_of_ms=now_ms,
                )
                empty_baseline = retriever.retrieve(
                    case.query,
                    user_id="counterfactual_empty_user",
                    as_of_ms=now_ms,
                )
                return self._timed(
                    item.memory_id in {memory.memory_id for memory in recalled}
                    and not empty_baseline,
                    started,
                )
            if case.case_type == "stale_suppression":
                return self._timed(
                    not retriever.retrieve(
                        case.query, user_id="user_1", as_of_ms=now_ms
                    ),
                    started,
                )
            if case.case_type == "cross_user_isolation":
                return self._timed(
                    not retriever.retrieve(
                        case.query, user_id="user_1", as_of_ms=now_ms
                    ),
                    started,
                )
            if case.case_type == "unsolicited_health":
                return self._timed(
                    not retriever.retrieve(
                        case.query,
                        user_id="user_1",
                        as_of_ms=now_ms,
                        use_context=MemoryUseContext(
                            user_initiated_recall=False,
                            explicit_sensitive_revisit=False,
                        ),
                    ),
                    started,
                )
            if case.case_type == "historical_crisis":
                return self._timed(
                    not retriever.retrieve(
                        case.query,
                        user_id="user_1",
                        as_of_ms=now_ms,
                        use_context=MemoryUseContext(
                            requested_use=MemoryAllowedUse.SAFETY_SUPPORT,
                            current_risk=RiskLevel.GREEN,
                            current_turn_corroborates_crisis=False,
                        ),
                    ),
                    started,
                )
            if case.case_type == "pause_enforcement":
                repository.set_paused(item.memory_id, user_id="user_1", paused=True)
                return self._timed(
                    not retriever.retrieve(
                        case.query, user_id="user_1", as_of_ms=now_ms
                    ),
                    started,
                )
            if case.case_type == "purpose_enforcement":
                return self._timed(
                    not retriever.retrieve(
                        case.query,
                        user_id="user_1",
                        as_of_ms=now_ms,
                        use_context=MemoryUseContext(
                            requested_use=MemoryAllowedUse.PERSONALIZATION,
                        ),
                    ),
                    started,
                )
            if case.case_type == "cascade_deletion":
                child = repository.add_active(
                    user_id="user_1",
                    candidate=candidate.model_copy(
                        update={
                            "text": f"derived {case.text}",
                            "subject_key": f"eval.derived.{case.case_id}",
                            "derived_from_memory_ids": [item.memory_id],
                        }
                    ),
                )
                receipt = repository.purge_with_receipt(
                    item.memory_id, user_id="user_1"
                )
                return self._timed(
                    bool(
                        receipt is not None
                        and receipt.deleted_memory_count == 2
                        and receipt.deleted_derived_count == 1
                        and not repository.list_visible("user_1")
                        and case.text not in receipt.model_dump_json()
                        and child.memory_id not in receipt.model_dump_json()
                    ),
                    started,
                )
            context = retriever.to_model_context(
                retriever.retrieve(
                    case.query, user_id="user_1", as_of_ms=now_ms
                )
            )
            return self._timed(
                bool(
                    context
                    and context[0]["source_type"] == "USER_STATEMENT"
                    and context[0]["sensitivity"] == "GENERAL"
                    and context[0]["valid_at_ms"] == 900
                    and "user_id" not in context[0]
                ),
                started,
            )

    @staticmethod
    def _timed(result: bool, started: float) -> tuple[bool, float]:
        return result, (perf_counter() - started) * 1000

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        ordered = sorted(values)
        index = max(0, min(len(ordered) - 1, int(len(ordered) * percentile) - 1))
        return round(ordered[index], 4)
