from pathlib import Path

import pytest

from app.memory.evaluation_protocol import (
    AdjudicatedQueryLabel,
    DatasetStatus,
    EvaluationSplit,
    RelevanceAnnotation,
    apply_adjudicated_labels,
    audit_annotations,
    bootstrap_report,
    cases_for_split,
    evaluate_slices,
    load_evaluation_manifest,
    validate_manifest_cases,
)
from app.memory.multisession_evaluation import (
    MultiSessionMemoryEvaluator,
    load_multisession_cases,
)
from app.memory.retrieval import (
    CachedMemoryEmbeddingProvider,
    GovernedHybridMemoryRetriever,
)
from app.memory.run_v15_evaluation import ThresholdTrial, _parse_thresholds, _select_trial


class TopicEmbeddingProvider:
    dimensions = 5

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, text: str) -> list[float]:
        self.calls += 1
        vector = [0.0] * self.dimensions
        groups = [
            ("详细", "展开", "回答"),
            ("慌", "焦虑", "办法", "蓝色", "平静"),
            ("毛孩子", "猫", "豆豆"),
            ("城市", "居住", "苏州", "生活"),
            ("语气", "压力", "追问", "温柔"),
        ]
        for index, terms in enumerate(groups):
            vector[index] = float(any(term in text for term in terms))
        return vector


def _paths() -> tuple[Path, Path]:
    root = Path(__file__).parents[4]
    return (
        root / "evals" / "memory_multisession_cases.jsonl",
        root / "evals" / "memory_v15_manifest.json",
    )


def test_manifest_enforces_dev_test_separation_and_fixture_disclosure() -> None:
    cases_path, manifest_path = _paths()
    cases = load_multisession_cases(cases_path)
    manifest = load_evaluation_manifest(manifest_path)

    validate_manifest_cases(manifest, cases)
    dev = cases_for_split(cases, manifest, EvaluationSplit.DEV)
    test = cases_for_split(cases, manifest, EvaluationSplit.TEST)

    assert manifest.dataset_status is DatasetStatus.ENGINEERING_FIXTURE
    assert {case.case_id for case in dev}.isdisjoint(case.case_id for case in test)
    assert len(dev) == 3
    assert len(test) == 2


def test_hybrid_fuses_candidates_and_preserves_governance() -> None:
    cases_path, _ = _paths()
    provider = CachedMemoryEmbeddingProvider(TopicEmbeddingProvider())
    report = MultiSessionMemoryEvaluator(
        strategy="hybrid-test-double",
        retriever_factory=lambda repository: GovernedHybridMemoryRetriever(
            repository,
            embedding_provider=provider,
        ),
    ).evaluate(load_multisession_cases(cases_path))

    assert report.retrieval_recall_at_5 == 1.0
    assert report.correct_abstention_rate == 1.0
    assert report.forbidden_retrieval_rate == 0.0
    assert report.failed_query_ids == []
    style = next(item for item in report.query_results if item.query_id == "style_q1")
    assert style.retrieved_memory_keys[0] == "style_current"


def test_embedding_cache_avoids_reencoding_identical_text() -> None:
    underlying = TopicEmbeddingProvider()
    cached = CachedMemoryEmbeddingProvider(underlying)

    assert cached.embed("毛孩子") == cached.embed("毛孩子")
    assert underlying.calls == 1
    assert cached.cache_size == 1


def test_annotation_audit_reports_agreement_kappa_and_conflicts() -> None:
    annotations = [
        RelevanceAnnotation(
            annotation_id="a1",
            dataset_version="1.5.0",
            annotator_id="ann_a",
            case_id="case",
            query_id="q1",
            relevance_grades={"m1": 3, "m2": 0},
        ),
        RelevanceAnnotation(
            annotation_id="a2",
            dataset_version="1.5.0",
            annotator_id="ann_b",
            case_id="case",
            query_id="q1",
            relevance_grades={"m1": 2, "m2": 0},
        ),
    ]

    audit = audit_annotations(annotations)

    assert audit.annotation_count == 2
    assert audit.exact_agreement_rate == 0.5
    assert 0 < audit.quadratic_weighted_kappa < 1
    assert [conflict.memory_key for conflict in audit.conflicts] == ["m1"]
    assert audit.insufficiently_annotated_queries == []


def test_annotation_audit_rejects_version_drift_and_duplicate_ids() -> None:
    annotation = RelevanceAnnotation(
        annotation_id="duplicate",
        dataset_version="1.4.0",
        annotator_id="ann_a",
        case_id="case",
        query_id="q1",
        relevance_grades={"m1": 3},
    )

    with pytest.raises(ValueError, match="versions"):
        audit_annotations([annotation], expected_dataset_version="1.5.0")
    with pytest.raises(ValueError, match="unique"):
        audit_annotations([annotation, annotation.model_copy()])


def test_annotation_grade_must_be_on_four_point_scale() -> None:
    with pytest.raises(ValueError, match="between zero and three"):
        RelevanceAnnotation(
            annotation_id="a1",
            dataset_version="1.5.0",
            annotator_id="ann_a",
            case_id="case",
            query_id="q1",
            relevance_grades={"m1": 4},
        )


def test_adjudication_replaces_only_label_fields() -> None:
    cases_path, _ = _paths()
    cases = load_multisession_cases(cases_path)
    original = next(query for query in cases[0].queries if query.query_id == "style_q1")
    labels = [
        AdjudicatedQueryLabel(
            dataset_version="1.5.0",
            case_id=cases[0].case_id,
            query_id="style_q1",
            relevance_grades={"style_current": 2},
            forbidden_memories={"style_old": "superseded"},
            adjudicator_id="adjudicator",
            source_annotation_ids=["a1", "a2"],
        )
    ]

    updated = apply_adjudicated_labels(cases, labels)
    query = next(item for item in updated[0].queries if item.query_id == "style_q1")

    assert query.relevance_grades == {"style_current": 2}
    assert query.text == original.text
    assert original.relevance_grades == {"style_current": 3}


def test_bootstrap_and_slice_reports_are_deterministic() -> None:
    cases_path, manifest_path = _paths()
    cases = load_multisession_cases(cases_path)
    manifest = load_evaluation_manifest(manifest_path)
    report = MultiSessionMemoryEvaluator(
        strategy="hybrid-test-double",
        retriever_factory=lambda repository: GovernedHybridMemoryRetriever(
            repository,
            embedding_provider=TopicEmbeddingProvider(),
        ),
    ).evaluate(cases)

    first = bootstrap_report(report, iterations=200, seed=15)
    second = bootstrap_report(report, iterations=200, seed=15)
    slices = evaluate_slices(report, manifest)

    assert first == second
    assert first.retrieval_recall_at_5 is not None
    assert first.retrieval_recall_at_5.point_estimate == 1.0
    assert slices["paraphrase"].retrieval_recall_at_5 == 1.0
    assert slices["cross_user"].forbidden_retrieval_rate == 0.0


def test_threshold_parser_and_selector_prefer_passing_conservative_tie() -> None:
    assert _parse_thresholds("0.4,0.3,0.40") == [0.3, 0.4]
    with pytest.raises(ValueError):
        _parse_thresholds("1.1")

    # Selection uses objective first, then the higher (more conservative) threshold.
    cases_path, _ = _paths()
    sample = MultiSessionMemoryEvaluator(
        strategy="sample",
        retriever_factory=lambda repository: GovernedHybridMemoryRetriever(
            repository,
            embedding_provider=TopicEmbeddingProvider(),
        ),
    ).evaluate(load_multisession_cases(cases_path))
    selected = _select_trial(
        [
            ThresholdTrial(
                semantic_min_relevance=0.35,
                objective=1.0,
                gate_violations=[],
                report=sample,
            ),
            ThresholdTrial(
                semantic_min_relevance=0.40,
                objective=1.0,
                gate_violations=[],
                report=sample,
            ),
        ]
    )
    assert selected.semantic_min_relevance == 0.40
