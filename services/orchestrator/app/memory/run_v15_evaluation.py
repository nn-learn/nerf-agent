import argparse
import json
from pathlib import Path

from pydantic import BaseModel, Field

from app.memory.evaluation_protocol import (
    AnnotationAuditReport,
    BootstrapReport,
    DatasetStatus,
    EvaluationSplit,
    SliceMetric,
    apply_adjudicated_labels,
    audit_annotations,
    bootstrap_report,
    cases_for_split,
    evaluate_slices,
    load_adjudicated_labels,
    load_annotations,
    load_evaluation_manifest,
    validate_manifest_cases,
)
from app.memory.multisession_evaluation import (
    MemoryQualityGate,
    MultiSessionMemoryCase,
    MultiSessionMemoryEvalReport,
    MultiSessionMemoryEvaluator,
    OllamaMemoryAnswerGenerator,
    load_multisession_cases,
)
from app.memory.retrieval import (
    CachedMemoryEmbeddingProvider,
    GovernedHybridMemoryRetriever,
)
from app.rag.bge import BgeM3EmbeddingProvider


class ThresholdTrial(BaseModel):
    semantic_min_relevance: float = Field(ge=0, le=1)
    objective: float
    gate_violations: list[str]
    report: MultiSessionMemoryEvalReport


class V15EvaluationReport(BaseModel):
    dataset_id: str
    dataset_version: str
    dataset_status: DatasetStatus
    annotation_audit: AnnotationAuditReport | None = None
    selected_on_split: EvaluationSplit
    reported_on_split: EvaluationSplit
    selected_semantic_min_relevance: float = Field(ge=0, le=1)
    development_trials: list[ThresholdTrial]
    held_out_test: MultiSessionMemoryEvalReport
    held_out_gate_violations: list[str]
    held_out_bootstrap: BootstrapReport
    held_out_slices: dict[str, SliceMetric]
    limitations: list[str]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Memory V1.5 hybrid evaluation")
    parser.add_argument(
        "--thresholds",
        default="0.30,0.35,0.40,0.45,0.50",
        help="comma-separated BGE cosine thresholds scanned on DEV only",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--with-ollama-answers", action="store_true")
    parser.add_argument("--ollama-model", default="qwen3.6:latest")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument(
        "--annotations",
        type=Path,
        help="optional independent annotation JSONL to audit",
    )
    parser.add_argument(
        "--adjudicated-labels",
        type=Path,
        help="optional adjudicated JSONL labels to apply before evaluation",
    )
    args = parser.parse_args()

    thresholds = _parse_thresholds(args.thresholds)
    project_root = Path(__file__).parents[4]
    cases = load_multisession_cases(project_root / "evals" / "memory_multisession_cases.jsonl")
    manifest = load_evaluation_manifest(project_root / "evals" / "memory_v15_manifest.json")
    validate_manifest_cases(manifest, cases)
    annotation_audit = None
    if args.annotations is not None:
        annotation_audit = audit_annotations(
            load_annotations(args.annotations),
            minimum_annotators=manifest.minimum_annotators,
            expected_dataset_version=manifest.dataset_version,
        )
    if args.adjudicated_labels is not None:
        cases = apply_adjudicated_labels(
            cases,
            load_adjudicated_labels(args.adjudicated_labels),
        )
    if (
        manifest.dataset_status is DatasetStatus.INDEPENDENTLY_ANNOTATED
        and args.adjudicated_labels is None
    ):
        raise ValueError(
            "INDEPENDENTLY_ANNOTATED datasets require explicit adjudicated labels"
        )
    dev_cases = cases_for_split(cases, manifest, EvaluationSplit.DEV)
    test_cases = cases_for_split(cases, manifest, EvaluationSplit.TEST)
    provider = CachedMemoryEmbeddingProvider(BgeM3EmbeddingProvider())
    gate = MemoryQualityGate()

    trials = [
        _evaluate_trial(
            threshold=threshold,
            cases=dev_cases,
            provider=provider,
            gate=gate,
        )
        for threshold in thresholds
    ]
    selected = _select_trial(trials)
    answer_generator = (
        OllamaMemoryAnswerGenerator(
            model=args.ollama_model,
            base_url=args.ollama_base_url,
        )
        if args.with_ollama_answers
        else None
    )
    try:
        test_report = MultiSessionMemoryEvaluator(
            strategy=f"hybrid-bge-m3@{selected.semantic_min_relevance:.2f}",
            retriever_factory=lambda repository: GovernedHybridMemoryRetriever(
                repository,
                embedding_provider=provider,
                semantic_min_relevance=selected.semantic_min_relevance,
            ),
            answer_generator=answer_generator,
        ).evaluate(test_cases)
    finally:
        if answer_generator is not None:
            answer_generator.close()
    report = V15EvaluationReport(
        dataset_id=manifest.dataset_id,
        dataset_version=manifest.dataset_version,
        dataset_status=manifest.dataset_status,
        annotation_audit=annotation_audit,
        selected_on_split=EvaluationSplit.DEV,
        reported_on_split=EvaluationSplit.TEST,
        selected_semantic_min_relevance=selected.semantic_min_relevance,
        development_trials=trials,
        held_out_test=test_report,
        held_out_gate_violations=gate.evaluate(
            test_report,
            require_answers=args.with_ollama_answers,
        ),
        held_out_bootstrap=bootstrap_report(
            test_report,
            iterations=args.bootstrap_iterations,
            seed=15,
        ),
        held_out_slices=evaluate_slices(test_report, manifest),
        limitations=[
            "ENGINEERING_FIXTURE means this is not independently annotated participant data.",
            "The held-out split has only two cases; percentile bootstrap intervals "
            "can be degenerate and are marked unreliable.",
            "A production threshold requires at least 500 independently labelled queries.",
            "The hybrid retriever remains offline-only and is not enabled in the live Agent.",
        ],
    )
    print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))


def _evaluate_trial(
    *,
    threshold: float,
    cases: list[MultiSessionMemoryCase],
    provider: CachedMemoryEmbeddingProvider,
    gate: MemoryQualityGate,
) -> ThresholdTrial:
    report = MultiSessionMemoryEvaluator(
        strategy=f"hybrid-bge-m3@{threshold:.2f}",
        retriever_factory=lambda repository: GovernedHybridMemoryRetriever(
            repository,
            embedding_provider=provider,
            semantic_min_relevance=threshold,
        ),
    ).evaluate(cases)
    violations = gate.evaluate(report, require_answers=False)
    objective = (
        0.45 * report.retrieval_recall_at_5
        + 0.35 * report.ndcg_at_5
        + 0.20 * report.correct_abstention_rate
        - 2.0 * report.forbidden_retrieval_rate
    )
    return ThresholdTrial(
        semantic_min_relevance=threshold,
        objective=round(objective, 6),
        gate_violations=violations,
        report=report,
    )
def _select_trial(trials: list[ThresholdTrial]) -> ThresholdTrial:
    passing = [trial for trial in trials if not trial.gate_violations]
    candidates = passing or trials
    return max(
        candidates,
        key=lambda trial: (trial.objective, trial.semantic_min_relevance),
    )


def _parse_thresholds(value: str) -> list[float]:
    try:
        thresholds = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as error:
        raise ValueError("thresholds must be comma-separated numbers") from error
    if not thresholds or any(not 0 <= threshold <= 1 for threshold in thresholds):
        raise ValueError("thresholds must contain values between zero and one")
    return thresholds


if __name__ == "__main__":
    main()
