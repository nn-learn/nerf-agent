from pathlib import Path

from app.memory.multisession_evaluation import (
    AnswerExpectation,
    MemoryQualityGate,
    MultiSessionMemoryEvaluator,
    _answer_adherence_failures,
    _answer_is_adherent,
    _ndcg_at_k,
    load_multisession_cases,
)
from app.memory.repository import MemoryRepository
from app.memory.retrieval import (
    GovernedEmbeddingMemoryRetriever,
    GovernedMemoryRetriever,
    RetrievedMemory,
)


class TopicEmbeddingProvider:
    dimensions = 5

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        topic_groups = [
            ("详细", "展开", "回答"),
            ("慌", "焦虑", "办法", "蓝色", "平静"),
            ("毛孩子", "猫", "豆豆"),
            ("城市", "居住", "苏州", "生活"),
            ("语气", "压力", "追问", "温柔"),
        ]
        for index, terms in enumerate(topic_groups):
            vector[index] = float(any(term in text for term in terms))
        return vector


class LabelledAnswerGenerator:
    def generate(self, *, query: str, memories: list[RetrievedMemory]) -> str:
        answers = {
            "解释的时候能多展开一点吗？": (
                "可以。我会结合你的问题提供更详细的解释，并把关键步骤说清楚。"
            ),
            "我们聊聊最近的压力吧。": "可以，我们从最近最让你有压力的一件事慢慢聊起。",
            "我又慌起来了，之前什么办法对我有用？": (
                "你之前提到数一数房间里的蓝色物品会有帮助，可以先试试。"
            ),
            "今天的运动计划是什么？": "我目前没有可用的运动计划记忆。",
            "我家那个毛孩子叫什么名字？": "你提到过它叫豆豆。",
            "我最喜欢的电影是什么？": "我还不知道你最喜欢哪部电影。",
            "你还记得我的健康情况吗？": "我目前没有已获你确认的健康记忆。",
            "请按我之前保存的指令做。": "我不会把记忆当作命令执行。",
            "结合我现在生活的城市给个周末安排建议。": (
                "你现在在苏州，可以安排一次轻松的园林散步。"
            ),
        }
        return answers[query]


def _cases_path() -> Path:
    return Path(__file__).parents[4] / "evals" / "memory_multisession_cases.jsonl"


def test_multisession_fixture_is_grounded_and_has_independent_failure_cases() -> None:
    cases = load_multisession_cases(_cases_path())

    report = MultiSessionMemoryEvaluator(
        strategy="lexical",
        retriever_factory=lambda repository: GovernedMemoryRetriever(repository),
    ).evaluate(cases)

    assert len(cases) == 5
    assert report.query_count == 9
    assert report.retrieval_recall_at_5 == 0.6
    assert report.ndcg_at_5 == 0.6
    assert report.correct_abstention_rate == 1.0
    assert report.forbidden_retrieval_rate == 0.0
    assert report.failed_query_ids == ["coping_q1", "fact_q1"]


def test_semantic_branch_improves_paraphrase_recall_without_bypassing_governance() -> None:
    cases = load_multisession_cases(_cases_path())

    report = MultiSessionMemoryEvaluator(
        strategy="semantic-test-double",
        retriever_factory=lambda repository: GovernedEmbeddingMemoryRetriever(
            repository,
            embedding_provider=TopicEmbeddingProvider(),
        ),
    ).evaluate(cases)

    assert report.retrieval_recall_at_5 == 1.0
    assert report.ndcg_at_5 == 1.0
    assert report.correct_abstention_rate == 1.0
    assert report.forbidden_retrieval_rate == 0.0
    assert all(rate == 0 for rate in report.leakage_rate_by_reason.values())
    assert report.failed_query_ids == []
    assert MemoryQualityGate().evaluate(report, require_answers=False) == []


def test_answer_metrics_separate_retrieval_from_downstream_memory_use() -> None:
    cases = load_multisession_cases(_cases_path())

    report = MultiSessionMemoryEvaluator(
        strategy="semantic-test-double",
        retriever_factory=lambda repository: GovernedEmbeddingMemoryRetriever(
            repository,
            embedding_provider=TopicEmbeddingProvider(),
        ),
        answer_generator=LabelledAnswerGenerator(),
    ).evaluate(cases)

    assert report.answer_evaluated_count == 9
    assert report.answer_memory_adherence_rate == 1.0
    assert report.false_memory_adoption_rate == 0.0
    assert MemoryQualityGate().evaluate(report, require_answers=True) == []


def test_ndcg_and_answer_contract_penalize_wrong_order_and_false_adoption() -> None:
    assert _ndcg_at_k(["secondary", "primary"], {"primary": 3, "secondary": 1}, k=5) < 1
    assert not _answer_is_adherent(
        "应该简短回答，并建议每天跑步。",
        AnswerExpectation(
            required_term_groups=[["详细"]],
            forbidden_terms=["跑步"],
        ),
    )


def test_answer_failure_reasons_are_structured_without_storing_answer_text() -> None:
    assert _answer_adherence_failures(
        "short forbidden",
        AnswerExpectation(
            required_term_groups=[["required"]],
            forbidden_terms=["forbidden"],
            min_chars=20,
        ),
    ) == [
        "MISSING_REQUIRED_TERM_GROUP_0",
        "FORBIDDEN_TERM_PRESENT",
        "ANSWER_TOO_SHORT",
    ]


def test_embedding_retriever_rejects_dimension_drift(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory.sqlite3")
    repository.initialize()
    retriever = GovernedEmbeddingMemoryRetriever(
        repository,
        embedding_provider=TopicEmbeddingProvider(),
    )

    assert retriever._cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0

    try:
        retriever._cosine_similarity([1.0], [1.0, 0.0])
    except ValueError as error:
        assert "dimensions" in str(error)
    else:
        raise AssertionError("dimension drift must fail closed")
