from pathlib import Path

from app.memory.evaluation_protocol import DatasetStatus, validate_manifest_cases
from app.memory.multisession_evaluation import load_multisession_cases
from app.memory.retrieval import RetrievedMemory
from app.memory.strategy_evaluation import MemoryStrategyEvaluator, load_v24_manifest


class TopicEmbeddingProvider:
    dimensions = 6

    def embed(self, text: str) -> list[float]:
        groups = [
            ("回答", "详细", "展开", "细节"),
            ("出差", "旅行", "猫", "豆豆", "照顾"),
            ("焦虑", "慌", "镇定", "蓝色", "平静"),
            ("城市", "杭州", "苏州", "居住"),
            ("电影", "饮料", "咖啡"),
            ("健康", "规则", "同事", "处方"),
        ]
        return [float(any(term in text for term in group)) for group in groups]


class EvidenceEchoAnswerGenerator:
    def generate(self, *, query: str, memories: list[RetrievedMemory]) -> str:
        del query
        if not memories:
            return "没有相关的已确认记忆。"
        return "；".join(memory.text for memory in memories)


def _fixture() -> tuple[Path, Path]:
    root = Path(__file__).parents[4]
    return (
        root / "evals" / "memory_v24_cases.jsonl",
        root / "evals" / "memory_v24_manifest.json",
    )


def test_v24_matrix_separates_retrieval_and_answer_stage_without_overclaiming() -> None:
    cases_path, manifest_path = _fixture()
    cases = load_multisession_cases(cases_path)
    manifest = load_v24_manifest(manifest_path)
    validate_manifest_cases(manifest, cases)

    report = MemoryStrategyEvaluator(
        embedding_provider=TopicEmbeddingProvider(),
    ).evaluate(cases, manifest=manifest)

    assert manifest.dataset_status is DatasetStatus.ENGINEERING_FIXTURE
    assert set(report.arms) == {
        "flat-evidence",
        "profile-first-confirmed",
        "episode-freshness-v2.3",
        "bge-m3-hybrid",
    }
    assert all(
        arm.retrieval.forbidden_retrieval_rate == 0 for arm in report.arms.values()
    )
    assert all(arm.answer_stage_evaluated_count == 0 for arm in report.arms.values())
    assert report.production_promotion_ready is False
    assert "dataset is not independently annotated" in report.promotion_blockers
    assert "final answers were not evaluated" in report.promotion_blockers


def test_episode_arm_improves_complete_evidence_coverage_on_event_case() -> None:
    cases_path, manifest_path = _fixture()
    report = MemoryStrategyEvaluator().evaluate(
        load_multisession_cases(cases_path),
        manifest=load_v24_manifest(manifest_path),
    )

    assert report.arms["episode-freshness-v2.3"].complete_evidence_recall_at_5 > (
        report.arms["flat-evidence"].complete_evidence_recall_at_5
    )
    assert report.omitted_arms == ["bge-m3-hybrid (embedding provider not requested)"]


def test_answer_stage_runs_only_for_the_selected_arm() -> None:
    cases_path, manifest_path = _fixture()
    report = MemoryStrategyEvaluator(
        embedding_provider=TopicEmbeddingProvider(),
        answer_generator=EvidenceEchoAnswerGenerator(),
        answer_arm="bge-m3-hybrid",
    ).evaluate(
        load_multisession_cases(cases_path),
        manifest=load_v24_manifest(manifest_path),
    )

    assert report.answer_evaluated_arms == ["bge-m3-hybrid"]
    assert report.arms["bge-m3-hybrid"].answer_stage_evaluated_count == 6
    assert all(
        arm.answer_stage_evaluated_count == 0
        for name, arm in report.arms.items()
        if name != "bge-m3-hybrid"
    )
    assert "final answers were not evaluated" not in report.promotion_blockers
