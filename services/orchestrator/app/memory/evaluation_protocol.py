import random
from collections import Counter, defaultdict
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from app.memory.multisession_evaluation import (
    MultiSessionMemoryCase,
    MultiSessionMemoryEvalReport,
    QueryEvaluation,
)


class DatasetStatus(StrEnum):
    ENGINEERING_FIXTURE = "ENGINEERING_FIXTURE"
    INDEPENDENTLY_ANNOTATED = "INDEPENDENTLY_ANNOTATED"


class EvaluationSplit(StrEnum):
    DEV = "DEV"
    TEST = "TEST"


class CaseProtocol(BaseModel):
    split: EvaluationSplit
    slices: list[str] = Field(min_length=1)


class EvaluationManifest(BaseModel):
    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    dataset_status: DatasetStatus
    label_policy_version: str = Field(min_length=1)
    minimum_annotators: int = Field(default=2, ge=2)
    cases: dict[str, CaseProtocol] = Field(min_length=1)


class RelevanceAnnotation(BaseModel):
    annotation_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    annotator_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    query_id: str = Field(min_length=1)
    relevance_grades: dict[str, int]
    forbidden_memories: dict[str, str] = Field(default_factory=dict)
    notes: str = ""

    @field_validator("relevance_grades")
    @classmethod
    def validate_relevance_grades(cls, value: dict[str, int]) -> dict[str, int]:
        for grade in value.values():
            _validate_grade(grade)
        return value


class AnnotationConflict(BaseModel):
    case_id: str
    query_id: str
    memory_key: str
    annotations: dict[str, int]


class AnnotationAuditReport(BaseModel):
    annotation_count: int
    query_count: int
    exact_agreement_rate: float = Field(ge=0, le=1)
    quadratic_weighted_kappa: float = Field(ge=-1, le=1)
    conflicts: list[AnnotationConflict]
    insufficiently_annotated_queries: list[str]


class AdjudicatedQueryLabel(BaseModel):
    dataset_version: str
    case_id: str
    query_id: str
    relevance_grades: dict[str, int]
    forbidden_memories: dict[str, str]
    adjudicator_id: str = Field(min_length=1)
    source_annotation_ids: list[str] = Field(min_length=2)


class MetricInterval(BaseModel):
    point_estimate: float = Field(ge=0, le=1)
    lower: float = Field(ge=0, le=1)
    upper: float = Field(ge=0, le=1)
    sample_count: int = Field(ge=0)
    reliable: bool
    warning: str | None = None


class BootstrapReport(BaseModel):
    confidence_level: float = Field(gt=0, lt=1)
    iterations: int = Field(gt=0)
    seed: int
    retrieval_recall_at_5: MetricInterval | None = None
    ndcg_at_5: MetricInterval | None = None
    correct_abstention_rate: MetricInterval | None = None


class SliceMetric(BaseModel):
    query_count: int = Field(ge=0)
    retrieval_recall_at_5: float | None = Field(default=None, ge=0, le=1)
    ndcg_at_5: float | None = Field(default=None, ge=0, le=1)
    correct_abstention_rate: float | None = Field(default=None, ge=0, le=1)
    forbidden_retrieval_rate: float = Field(ge=0, le=1)
    failed_query_ids: list[str]


def load_evaluation_manifest(path: Path) -> EvaluationManifest:
    return EvaluationManifest.model_validate_json(path.read_text(encoding="utf-8"))


def validate_manifest_cases(
    manifest: EvaluationManifest,
    cases: list[MultiSessionMemoryCase],
) -> None:
    case_ids = {case.case_id for case in cases}
    manifest_ids = set(manifest.cases)
    if case_ids != manifest_ids:
        missing = sorted(case_ids - manifest_ids)
        unknown = sorted(manifest_ids - case_ids)
        raise ValueError(f"manifest mismatch: missing={missing}, unknown={unknown}")


def cases_for_split(
    cases: list[MultiSessionMemoryCase],
    manifest: EvaluationManifest,
    split: EvaluationSplit,
) -> list[MultiSessionMemoryCase]:
    validate_manifest_cases(manifest, cases)
    return [case for case in cases if manifest.cases[case.case_id].split is split]


def load_annotations(path: Path) -> list[RelevanceAnnotation]:
    return [
        RelevanceAnnotation.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_adjudicated_labels(path: Path) -> list[AdjudicatedQueryLabel]:
    return [
        AdjudicatedQueryLabel.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit_annotations(
    annotations: list[RelevanceAnnotation],
    *,
    minimum_annotators: int = 2,
    expected_dataset_version: str | None = None,
) -> AnnotationAuditReport:
    if minimum_annotators < 2:
        raise ValueError("minimum_annotators must be at least two")
    versions = {annotation.dataset_version for annotation in annotations}
    if expected_dataset_version is not None and versions != {expected_dataset_version}:
        raise ValueError(
            f"annotation dataset versions {sorted(versions)} do not match "
            f"{expected_dataset_version}"
        )
    identities = [annotation.annotation_id for annotation in annotations]
    if len(identities) != len(set(identities)):
        raise ValueError("annotation IDs must be unique")
    grouped: dict[tuple[str, str], list[RelevanceAnnotation]] = defaultdict(list)
    for annotation in annotations:
        grouped[(annotation.case_id, annotation.query_id)].append(annotation)
    conflicts: list[AnnotationConflict] = []
    paired_grades: list[tuple[int, int]] = []
    exact = 0
    compared = 0
    insufficient: list[str] = []
    for (case_id, query_id), group in sorted(grouped.items()):
        annotators = {annotation.annotator_id for annotation in group}
        if len(annotators) < minimum_annotators:
            insufficient.append(f"{case_id}/{query_id}")
            continue
        ordered = sorted(group, key=lambda item: item.annotator_id)
        left, right = ordered[0], ordered[1]
        memory_keys = set(left.relevance_grades) | set(right.relevance_grades)
        for memory_key in sorted(memory_keys):
            left_grade = left.relevance_grades.get(memory_key, 0)
            right_grade = right.relevance_grades.get(memory_key, 0)
            _validate_grade(left_grade)
            _validate_grade(right_grade)
            paired_grades.append((left_grade, right_grade))
            compared += 1
            exact += int(left_grade == right_grade)
            if left_grade != right_grade:
                conflicts.append(
                    AnnotationConflict(
                        case_id=case_id,
                        query_id=query_id,
                        memory_key=memory_key,
                        annotations={
                            left.annotator_id: left_grade,
                            right.annotator_id: right_grade,
                        },
                    )
                )
    return AnnotationAuditReport(
        annotation_count=len(annotations),
        query_count=len(grouped),
        exact_agreement_rate=exact / compared if compared else 0.0,
        quadratic_weighted_kappa=_quadratic_weighted_kappa(paired_grades),
        conflicts=conflicts,
        insufficiently_annotated_queries=insufficient,
    )


def apply_adjudicated_labels(
    cases: list[MultiSessionMemoryCase],
    labels: list[AdjudicatedQueryLabel],
) -> list[MultiSessionMemoryCase]:
    by_query = {(label.case_id, label.query_id): label for label in labels}
    known = {
        (case.case_id, query.query_id)
        for case in cases
        for query in case.queries
    }
    unknown = set(by_query) - known
    if unknown:
        raise ValueError(f"adjudicated labels reference unknown queries: {sorted(unknown)}")
    updated: list[MultiSessionMemoryCase] = []
    for case in cases:
        memory_keys = {memory.memory_key for memory in case.memories}
        queries = []
        for query in case.queries:
            label = by_query.get((case.case_id, query.query_id))
            if label is None:
                queries.append(query.model_copy(deep=True))
                continue
            referenced = set(label.relevance_grades) | set(label.forbidden_memories)
            if referenced - memory_keys:
                raise ValueError(f"adjudicated labels contain unknown memory in {case.case_id}")
            for grade in label.relevance_grades.values():
                _validate_grade(grade)
            queries.append(
                query.model_copy(
                    update={
                        "relevance_grades": dict(label.relevance_grades),
                        "forbidden_memories": dict(label.forbidden_memories),
                    },
                    deep=True,
                )
            )
        updated.append(case.model_copy(update={"queries": queries}, deep=True))
    return updated


def evaluate_slices(
    report: MultiSessionMemoryEvalReport,
    manifest: EvaluationManifest,
) -> dict[str, SliceMetric]:
    results_by_slice: dict[str, list[QueryEvaluation]] = defaultdict(list)
    for result in report.query_results:
        protocol = manifest.cases[result.case_id]
        for slice_name in protocol.slices:
            results_by_slice[slice_name].append(result)
    return {
        name: _slice_metric(results)
        for name, results in sorted(results_by_slice.items())
    }


def bootstrap_report(
    report: MultiSessionMemoryEvalReport,
    *,
    iterations: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 15,
) -> BootstrapReport:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")
    randomizer = random.Random(seed)
    return BootstrapReport(
        confidence_level=confidence_level,
        iterations=iterations,
        seed=seed,
        retrieval_recall_at_5=_bootstrap_metric(
            [item.recall_at_5 for item in report.query_results if item.recall_at_5 is not None],
            iterations=iterations,
            confidence_level=confidence_level,
            randomizer=randomizer,
        ),
        ndcg_at_5=_bootstrap_metric(
            [item.ndcg_at_5 for item in report.query_results if item.ndcg_at_5 is not None],
            iterations=iterations,
            confidence_level=confidence_level,
            randomizer=randomizer,
        ),
        correct_abstention_rate=_bootstrap_metric(
            [
                float(item.correct_abstention)
                for item in report.query_results
                if item.correct_abstention is not None
            ],
            iterations=iterations,
            confidence_level=confidence_level,
            randomizer=randomizer,
        ),
    )


def _slice_metric(results: list[QueryEvaluation]) -> SliceMetric:
    recall = [item.recall_at_5 for item in results if item.recall_at_5 is not None]
    ndcg = [item.ndcg_at_5 for item in results if item.ndcg_at_5 is not None]
    abstention = [
        item.correct_abstention
        for item in results
        if item.correct_abstention is not None
    ]
    forbidden_attempts = sum(item.forbidden_memory_count for item in results)
    forbidden_leaks = sum(len(item.forbidden_retrieved_keys) for item in results)
    failed = [
        item.query_id
        for item in results
        if item.recall_at_5 == 0
        or item.correct_abstention is False
        or bool(item.forbidden_retrieved_keys)
    ]
    return SliceMetric(
        query_count=len(results),
        retrieval_recall_at_5=_optional_mean(recall),
        ndcg_at_5=_optional_mean(ndcg),
        correct_abstention_rate=_optional_mean(abstention),
        forbidden_retrieval_rate=(
            forbidden_leaks / forbidden_attempts if forbidden_attempts else 0.0
        ),
        failed_query_ids=failed,
    )


def _bootstrap_metric(
    values: list[float],
    *,
    iterations: int,
    confidence_level: float,
    randomizer: random.Random,
) -> MetricInterval | None:
    if not values:
        return None
    estimates = []
    for _ in range(iterations):
        sample = randomizer.choices(values, k=len(values))
        estimates.append(sum(sample) / len(sample))
    estimates.sort()
    alpha = (1 - confidence_level) / 2
    return MetricInterval(
        point_estimate=sum(values) / len(values),
        lower=_quantile(estimates, alpha),
        upper=_quantile(estimates, 1 - alpha),
        sample_count=len(values),
        reliable=len(values) >= 30,
        warning=(
            None
            if len(values) >= 30
            else "Fewer than 30 observations; percentile bootstrap may be degenerate."
        ),
    )


def _quadratic_weighted_kappa(pairs: list[tuple[int, int]]) -> float:
    if not pairs:
        return 0.0
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    count = len(pairs)
    observed = sum(((left - right) / 3) ** 2 for left, right in pairs) / count
    expected = sum(
        left_counts[left] * right_counts[right] / count**2 * ((left - right) / 3) ** 2
        for left in range(4)
        for right in range(4)
    )
    if expected == 0:
        return 1.0 if observed == 0 else 0.0
    return max(-1.0, min(1.0, 1 - observed / expected))


def _quantile(values: list[float], fraction: float) -> float:
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(len(values) - 1, lower + 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _optional_mean(values: list[float] | list[bool]) -> float | None:
    return sum(float(value) for value in values) / len(values) if values else None


def _validate_grade(grade: int) -> None:
    if not 0 <= grade <= 3:
        raise ValueError("relevance grades must be between zero and three")
