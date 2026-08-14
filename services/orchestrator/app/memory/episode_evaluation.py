import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from app.memory.consolidation import MemoryProfileRepository
from app.memory.episodes import EpisodeAugmentedMemoryRetriever, MemoryEpisodeRepository
from app.memory.models import MemoryAspect, MemoryCandidate, MemoryKind
from app.memory.repository import MemoryRepository
from app.memory.retrieval import GovernedMemoryRetriever


class EpisodeEvalMemory(BaseModel):
    alias: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    subject_key: str = Field(min_length=1)
    aspect: MemoryAspect = MemoryAspect.FACT


class EpisodeEvalCase(BaseModel):
    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    memories: list[EpisodeEvalMemory] = Field(min_length=2)
    expected_aliases: list[str] = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=20)


class EpisodeCaseResult(BaseModel):
    case_id: str
    expected_count: int = Field(ge=1)
    flat_recalled_aliases: list[str]
    episode_recalled_aliases: list[str]
    flat_recall: float = Field(ge=0, le=1)
    episode_recall: float = Field(ge=0, le=1)
    episode_precision: float = Field(ge=0, le=1)
    summary_text_entered_context: bool


class EpisodeEvalReport(BaseModel):
    strategy: str = "memory-episode-v2.3"
    case_count: int = Field(ge=1)
    flat_multihop_recall_at_k: float = Field(ge=0, le=1)
    episode_multihop_recall_at_k: float = Field(ge=0, le=1)
    episode_precision_at_k: float = Field(ge=0, le=1)
    summary_context_leakage_rate: float = Field(ge=0, le=1)
    graph_gate_minimum_cases: int = Field(default=50, ge=1)
    graph_gate_ready: bool
    graph_activation_recommended: bool
    failed_case_ids: list[str]
    case_results: list[EpisodeCaseResult]


def load_episode_cases(path: Path) -> list[EpisodeEvalCase]:
    return [
        EpisodeEvalCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class MemoryEpisodeEvaluator:
    """Evaluate source expansion and keep graph activation behind a data gate."""

    def __init__(self, *, graph_gate_minimum_cases: int = 50) -> None:
        if graph_gate_minimum_cases <= 0:
            raise ValueError("graph gate minimum must be positive")
        self._graph_gate_minimum_cases = graph_gate_minimum_cases

    def evaluate(self, cases: list[EpisodeEvalCase]) -> EpisodeEvalReport:
        if not cases:
            raise ValueError("at least one episode case is required")
        results = [self._evaluate_case(case) for case in cases]
        ready = len(results) >= self._graph_gate_minimum_cases
        failed = [
            result.case_id
            for result in results
            if result.episode_recall < result.flat_recall
            or result.episode_precision < 0.8
            or result.summary_text_entered_context
        ]
        return EpisodeEvalReport(
            case_count=len(results),
            flat_multihop_recall_at_k=sum(result.flat_recall for result in results)
            / len(results),
            episode_multihop_recall_at_k=sum(
                result.episode_recall for result in results
            )
            / len(results),
            episode_precision_at_k=sum(
                result.episode_precision for result in results
            )
            / len(results),
            summary_context_leakage_rate=sum(
                result.summary_text_entered_context for result in results
            )
            / len(results),
            graph_gate_minimum_cases=self._graph_gate_minimum_cases,
            graph_gate_ready=ready,
            # No graph arm is allowed into production until an independent set is
            # large enough and demonstrates a material end-to-end gain.
            graph_activation_recommended=False,
            failed_case_ids=failed,
            case_results=results,
        )

    @staticmethod
    def _evaluate_case(case: EpisodeEvalCase) -> EpisodeCaseResult:
        with tempfile.TemporaryDirectory(prefix="psyavatar-memory-v23-") as directory:
            repository = MemoryRepository(Path(directory) / "memory.sqlite3")
            repository.initialize()
            profiles = MemoryProfileRepository(repository.database_path)
            episodes = MemoryEpisodeRepository(repository.database_path)
            alias_by_id: dict[str, str] = {}
            for index, memory in enumerate(case.memories):
                item = repository.add_active(
                    user_id="user_eval",
                    candidate=MemoryCandidate(
                        source="evaluation",
                        contains_sensitive_content=False,
                        text=memory.text,
                        kind=MemoryKind.EPISODIC,
                        source_turn_id=f"turn_{index}",
                        aspect=memory.aspect,
                        subject_key=memory.subject_key,
                        confidence=0.9,
                        valid_from_ms=1_000 + index,
                    ),
                    now_ms=2_000 + index,
                )
                alias_by_id[item.memory_id] = memory.alias
                profiles.record_observation(
                    item,
                    session_id=memory.session_id,
                    now_ms=2_000 + index,
                )
            for session_id in sorted({memory.session_id for memory in case.memories}):
                episodes.rebuild_session(
                    user_id="user_eval",
                    session_id=session_id,
                    memory_repository=repository,
                    now_ms=3_000,
                )
            flat_retriever = GovernedMemoryRetriever(repository)
            episode_retriever = EpisodeAugmentedMemoryRetriever(
                flat_retriever,
                episodes=episodes,
                memories=repository,
            )
            flat = flat_retriever.retrieve(
                case.query,
                user_id="user_eval",
                k=case.k,
                as_of_ms=4_000,
            )
            augmented = episode_retriever.retrieve(
                case.query,
                user_id="user_eval",
                k=case.k,
                as_of_ms=4_000,
            )
            flat_aliases = [alias_by_id[item.memory_id] for item in flat]
            episode_aliases = [alias_by_id[item.memory_id] for item in augmented]
            expected = set(case.expected_aliases)
            flat_hits = expected & set(flat_aliases)
            episode_hits = expected & set(episode_aliases)
            episode_precision = (
                len(episode_hits) / len(episode_aliases) if episode_aliases else 1.0
            )
            source_text_by_id = {
                item.memory_id: item.candidate.text
                for item in repository.list_active("user_eval", as_of_ms=4_000)
            }
            model_context = episode_retriever.to_model_context(augmented)
            # A one-member episode can legitimately have the same text as its
            # source memory. Track provenance instead of comparing strings: every
            # model-context row must resolve to a governed source memory and carry
            # that source's exact text.
            summary_entered_context = any(
                str(row["memory_id"]) not in source_text_by_id
                or str(row["fact"])
                != source_text_by_id.get(str(row["memory_id"]))
                for row in model_context
            )
            return EpisodeCaseResult(
                case_id=case.case_id,
                expected_count=len(expected),
                flat_recalled_aliases=flat_aliases,
                episode_recalled_aliases=episode_aliases,
                flat_recall=len(flat_hits) / len(expected),
                episode_recall=len(episode_hits) / len(expected),
                episode_precision=episode_precision,
                summary_text_entered_context=summary_entered_context,
            )
