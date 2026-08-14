from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from app.memory.consolidation import MemoryConsolidator, MemoryProfileRepository
from app.memory.episodes import (
    EpisodeAugmentedMemoryRetriever,
    FreshnessAwareMemoryRetriever,
    MemoryEpisodeRepository,
    MemoryRetrieverPort,
)
from app.memory.evaluation_protocol import (
    AnnotationAuditReport,
    DatasetStatus,
    EvaluationManifest,
)
from app.memory.models import MemoryProfileState
from app.memory.multisession_evaluation import (
    MemoryAnswerGenerator,
    MultiSessionMemoryCase,
    MultiSessionMemoryEvalReport,
    MultiSessionMemoryEvaluator,
)
from app.memory.repository import MemoryRepository
from app.memory.retrieval import (
    GovernedHybridMemoryRetriever,
    GovernedMemoryRetriever,
    GovernedProfileRetriever,
    LayeredMemoryRetriever,
    MemoryEmbeddingProvider,
    RetrievedMemory,
)


class StrategyStageMetrics(BaseModel):
    strategy: str
    retrieval: MultiSessionMemoryEvalReport
    complete_evidence_recall_at_5: float = Field(ge=0, le=1)
    precision_at_5: float = Field(ge=0, le=1)
    mean_reciprocal_rank: float = Field(ge=0, le=1)
    evidence_sufficiency_rate: float = Field(ge=0, le=1)
    irrelevant_context_rate: float = Field(ge=0, le=1)
    answer_stage_evaluated_count: int = Field(ge=0)
    final_answer_adherence_rate: float | None = Field(default=None, ge=0, le=1)
    false_memory_adoption_rate: float | None = Field(default=None, ge=0, le=1)


class MemoryStrategyEvaluationReport(BaseModel):
    evaluation_version: str = "memory-v2.4"
    dataset_id: str
    dataset_version: str
    dataset_status: DatasetStatus
    annotation_audit: AnnotationAuditReport | None = None
    case_count: int = Field(ge=1)
    query_count: int = Field(ge=1)
    arms: dict[str, StrategyStageMetrics]
    omitted_arms: list[str]
    answer_evaluated_arms: list[str]
    recommended_offline_arm: str
    recommendation_scope: str = "retrieval-evidence-only"
    production_promotion_ready: bool
    promotion_blockers: list[str]
    limitations: list[str]


class EvaluationLayeredRetriever:
    """Lazily materialize confirmed profiles/episodes for an offline arm.

    The fixture's source memories are already labelled as user-confirmed. Profile
    proposals are confirmed inside this evaluation harness so profile-first can be
    measured as a counterfactual arm. This never changes the live confirmation
    policy and the report discloses the simulation.
    """

    def __init__(
        self,
        repository: MemoryRepository,
        *,
        include_episode_index: bool,
        include_freshness_rerank: bool,
    ) -> None:
        self._repository = repository
        self._profiles = MemoryProfileRepository(repository.database_path)
        evidence = GovernedMemoryRetriever(repository)
        layered: MemoryRetrieverPort = LayeredMemoryRetriever(
            evidence,
            GovernedProfileRetriever(self._profiles),
        )
        self._episodes = (
            MemoryEpisodeRepository(repository.database_path)
            if include_episode_index
            else None
        )
        if self._episodes is not None:
            layered = EpisodeAugmentedMemoryRetriever(
                layered,
                episodes=self._episodes,
                memories=repository,
            )
        if include_freshness_rerank:
            layered = FreshnessAwareMemoryRetriever(layered)
        self._retriever = layered
        self._prepared_users: set[str] = set()

    def retrieve(
        self,
        query: str,
        *,
        user_id: str,
        purpose_scope: str = "personalization",
        k: int = 5,
        as_of_ms: int | None = None,
    ) -> list[RetrievedMemory]:
        self.prepare_for_evaluation(
            user_id=user_id,
            purpose_scope=purpose_scope,
            as_of_ms=as_of_ms,
        )
        return self._retriever.retrieve(
            query,
            user_id=user_id,
            purpose_scope=purpose_scope,
            k=k,
            as_of_ms=as_of_ms,
        )

    def to_model_context(
        self,
        items: list[RetrievedMemory],
    ) -> list[dict[str, object]]:
        return self._retriever.to_model_context(items)

    def prepare_for_evaluation(
        self,
        *,
        user_id: str,
        purpose_scope: str = "personalization",
        as_of_ms: int | None,
    ) -> None:
        if user_id in self._prepared_users:
            return
        timestamp = as_of_ms if as_of_ms is not None else 0
        session_ids = {
            observation.session_id
            for observation, _ in self._profiles.evidence_for_user(user_id)
        }
        proposals = MemoryConsolidator(self._profiles).rebuild_user(
            user_id,
            now_ms=timestamp,
        )
        for profile in proposals:
            if profile.state is MemoryProfileState.AWAITING_CONFIRMATION:
                self._profiles.confirm_profile(
                    profile.profile_id,
                    user_id=user_id,
                    now_ms=timestamp,
                )
        if self._episodes is not None:
            for session_id in sorted(session_ids):
                self._episodes.rebuild_session(
                    user_id=user_id,
                    session_id=session_id,
                    memory_repository=self._repository,
                    purpose_scope=purpose_scope,
                    now_ms=timestamp,
                )
        self._prepared_users.add(user_id)


class MemoryStrategyEvaluator:
    """Compare retrieval arms, then report answer-stage evidence readiness."""

    def __init__(
        self,
        *,
        embedding_provider: MemoryEmbeddingProvider | None = None,
        answer_generator: MemoryAnswerGenerator | None = None,
        answer_arm: str = "episode-freshness-v2.3",
    ) -> None:
        self._embedding_provider = embedding_provider
        self._answer_generator = answer_generator
        self._answer_arm = answer_arm

    def evaluate(
        self,
        cases: list[MultiSessionMemoryCase],
        *,
        manifest: EvaluationManifest,
        annotation_audit: AnnotationAuditReport | None = None,
    ) -> MemoryStrategyEvaluationReport:
        if not cases:
            raise ValueError("at least one V2.4 case is required")
        factories: dict[str, Callable[[MemoryRepository], MemoryRetrieverPort]] = {
            "flat-evidence": lambda repository: GovernedMemoryRetriever(repository),
            "profile-first-confirmed": lambda repository: EvaluationLayeredRetriever(
                repository,
                include_episode_index=False,
                include_freshness_rerank=False,
            ),
            "episode-freshness-v2.3": lambda repository: EvaluationLayeredRetriever(
                repository,
                include_episode_index=True,
                include_freshness_rerank=True,
            ),
        }
        omitted: list[str] = []
        embedding_provider = self._embedding_provider
        if embedding_provider is None:
            omitted.append("bge-m3-hybrid (embedding provider not requested)")
        else:
            factories["bge-m3-hybrid"] = lambda repository: (
                GovernedHybridMemoryRetriever(
                    repository,
                    embedding_provider=embedding_provider,
                    semantic_min_relevance=0.40,
                )
            )
        if self._answer_generator is not None and self._answer_arm not in factories:
            raise ValueError(f"answer arm is unavailable: {self._answer_arm}")
        arms = {
            strategy: self._evaluate_arm(strategy, factory, cases)
            for strategy, factory in factories.items()
        }
        recommended = max(
            arms.values(),
            key=lambda arm: (
                arm.evidence_sufficiency_rate,
                arm.complete_evidence_recall_at_5,
                arm.retrieval.ndcg_at_5,
                -arm.retrieval.forbidden_retrieval_rate,
                -arm.retrieval.retrieval_latency_ms_p95,
            ),
        ).strategy
        query_count = sum(len(case.queries) for case in cases)
        blockers = self._promotion_blockers(
            manifest=manifest,
            query_count=query_count,
            arms=arms,
            annotation_audit=annotation_audit,
        )
        return MemoryStrategyEvaluationReport(
            dataset_id=manifest.dataset_id,
            dataset_version=manifest.dataset_version,
            dataset_status=manifest.dataset_status,
            annotation_audit=annotation_audit,
            case_count=len(cases),
            query_count=query_count,
            arms=arms,
            omitted_arms=omitted,
            answer_evaluated_arms=[
                name
                for name, arm in arms.items()
                if arm.answer_stage_evaluated_count > 0
            ],
            recommended_offline_arm=recommended,
            production_promotion_ready=not blockers,
            promotion_blockers=blockers,
            limitations=[
                "The bundled V2.4 set is an engineering fixture, not independent gold data.",
                "Profile confirmation is simulated only inside the profile-first evaluation arms.",
                "Evidence readiness is not a substitute for human review of "
                "final response quality.",
                "BGE-M3 and local Qwen answer evaluation are optional CPU-capable arms.",
            ],
        )

    def _evaluate_arm(
        self,
        strategy: str,
        factory: Callable[[MemoryRepository], MemoryRetrieverPort],
        cases: list[MultiSessionMemoryCase],
    ) -> StrategyStageMetrics:
        report = MultiSessionMemoryEvaluator(
            strategy=strategy,
            retriever_factory=factory,
            answer_generator=(
                self._answer_generator if strategy == self._answer_arm else None
            ),
        ).evaluate(cases)
        complete_recall: list[float] = []
        precision: list[float] = []
        reciprocal_rank: list[float] = []
        sufficiency: list[float] = []
        irrelevant = 0
        retrieved_total = 0
        for result in report.query_results:
            relevant = set(result.relevant_memory_keys)
            retrieved = result.retrieved_memory_keys
            hits = relevant & set(retrieved)
            if relevant:
                complete_recall.append(len(hits) / len(relevant))
                precision.append(len(hits) / len(retrieved) if retrieved else 0.0)
                reciprocal_rank.append(
                    next(
                        (1 / rank for rank, key in enumerate(retrieved, 1) if key in relevant),
                        0.0,
                    )
                )
                sufficiency.append(float(relevant <= set(retrieved)))
            else:
                sufficiency.append(float(not retrieved))
            irrelevant += sum(key not in relevant for key in retrieved)
            retrieved_total += len(retrieved)
        return StrategyStageMetrics(
            strategy=strategy,
            retrieval=report,
            complete_evidence_recall_at_5=_mean(complete_recall),
            precision_at_5=_mean(precision),
            mean_reciprocal_rank=_mean(reciprocal_rank),
            evidence_sufficiency_rate=_mean(sufficiency),
            irrelevant_context_rate=(
                irrelevant / retrieved_total if retrieved_total else 0.0
            ),
            answer_stage_evaluated_count=report.answer_evaluated_count,
            final_answer_adherence_rate=report.answer_memory_adherence_rate,
            false_memory_adoption_rate=report.false_memory_adoption_rate,
        )

    def _promotion_blockers(
        self,
        *,
        manifest: EvaluationManifest,
        query_count: int,
        arms: dict[str, StrategyStageMetrics],
        annotation_audit: AnnotationAuditReport | None,
    ) -> list[str]:
        blockers: list[str] = []
        if manifest.dataset_status is not DatasetStatus.INDEPENDENTLY_ANNOTATED:
            blockers.append("dataset is not independently annotated")
        elif annotation_audit is None:
            blockers.append("independent annotation audit was not supplied")
        elif annotation_audit.insufficiently_annotated_queries:
            blockers.append("some queries have fewer than two independent annotations")
        if query_count < 50:
            blockers.append("fewer than 50 held-out queries")
        if not any(arm.answer_stage_evaluated_count > 0 for arm in arms.values()):
            blockers.append("final answers were not evaluated")
        leaking = [
            name
            for name, arm in arms.items()
            if arm.retrieval.forbidden_retrieval_rate > 0
        ]
        if leaking:
            blockers.append(f"forbidden retrieval detected in: {', '.join(leaking)}")
        return blockers


def load_v24_manifest(path: Path) -> EvaluationManifest:
    return EvaluationManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 1.0
