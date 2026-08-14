import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from app.memory.consolidation import MemoryConsolidator, MemoryProfileRepository
from app.memory.models import MemoryAspect, MemoryCandidate, MemoryKind, MemoryProfileState
from app.memory.repository import MemoryRepository
from app.memory.retrieval import GovernedProfileRetriever


class ConsolidationEvidence(BaseModel):
    session_id: str = Field(min_length=1)
    turn_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    subject_key: str = Field(min_length=1)
    confirmed: bool = True


class ConsolidationEvalCase(BaseModel):
    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    evidence: list[ConsolidationEvidence] = Field(min_length=1)
    expect_profile: bool
    expect_conflict: bool


class ConsolidationCaseResult(BaseModel):
    case_id: str
    profile_expected: bool
    profile_observed: bool
    conflict_expected: bool
    conflict_observed: bool
    preconfirmation_retrieval_count: int = Field(ge=0)
    confirmed_profile_recalled: bool | None = None
    deletion_cascade_passed: bool | None = None


class ConsolidationEvalReport(BaseModel):
    strategy: str = "memory-profile-v2.0"
    case_count: int = Field(ge=1)
    profile_proposal_accuracy: float = Field(ge=0, le=1)
    conflict_detection_accuracy: float = Field(ge=0, le=1)
    preconfirmation_leakage_rate: float = Field(ge=0, le=1)
    confirmed_profile_recall_rate: float = Field(ge=0, le=1)
    deletion_cascade_rate: float = Field(ge=0, le=1)
    failed_case_ids: list[str]
    case_results: list[ConsolidationCaseResult]


def load_consolidation_cases(path: Path) -> list[ConsolidationEvalCase]:
    return [
        ConsolidationEvalCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class MemoryConsolidationEvaluator:
    """Deterministic governance eval; it does not claim clinical validity."""

    def evaluate(
        self,
        cases: list[ConsolidationEvalCase],
    ) -> ConsolidationEvalReport:
        if not cases:
            raise ValueError("at least one consolidation case is required")
        results = [self._evaluate_case(case) for case in cases]
        stable_results = [
            result
            for result in results
            if result.profile_expected and not result.conflict_expected
        ]
        profile_correct = sum(
            result.profile_observed == result.profile_expected for result in results
        )
        conflict_correct = sum(
            result.conflict_observed == result.conflict_expected for result in results
        )
        leaked = sum(result.preconfirmation_retrieval_count > 0 for result in results)
        confirmed_recalled = sum(
            result.confirmed_profile_recalled is True for result in stable_results
        )
        deletion_passed = sum(
            result.deletion_cascade_passed is True for result in stable_results
        )
        failed = [
            result.case_id
            for result in results
            if result.profile_observed != result.profile_expected
            or result.conflict_observed != result.conflict_expected
            or result.preconfirmation_retrieval_count > 0
            or result.confirmed_profile_recalled is False
            or result.deletion_cascade_passed is False
        ]
        stable_count = len(stable_results)
        return ConsolidationEvalReport(
            case_count=len(results),
            profile_proposal_accuracy=profile_correct / len(results),
            conflict_detection_accuracy=conflict_correct / len(results),
            preconfirmation_leakage_rate=leaked / len(results),
            confirmed_profile_recall_rate=(
                confirmed_recalled / stable_count if stable_count else 1.0
            ),
            deletion_cascade_rate=(
                deletion_passed / stable_count if stable_count else 1.0
            ),
            failed_case_ids=failed,
            case_results=results,
        )

    @staticmethod
    def _evaluate_case(case: ConsolidationEvalCase) -> ConsolidationCaseResult:
        with tempfile.TemporaryDirectory(prefix="psyavatar-memory-v2-") as directory:
            repository = MemoryRepository(Path(directory) / "memory.sqlite3")
            repository.initialize()
            profiles = MemoryProfileRepository(repository.database_path)
            consolidator = MemoryConsolidator(profiles)
            user_id = f"user_{case.case_id}"
            for evidence in case.evidence:
                candidate = MemoryCandidate(
                    source="transcript",
                    contains_sensitive_content=False,
                    text=evidence.text,
                    kind=MemoryKind.SEMANTIC,
                    source_turn_id=evidence.turn_id,
                    aspect=MemoryAspect.PREFERENCE,
                    subject_key=evidence.subject_key,
                    confidence=0.9,
                )
                item = (
                    repository.add_active(user_id=user_id, candidate=candidate)
                    if evidence.confirmed
                    else repository.add_awaiting_consent(
                        user_id=user_id,
                        candidate=candidate,
                    )
                )
                profiles.record_observation(item, session_id=evidence.session_id)
            consolidator.rebuild_user(user_id)
            proposed = profiles.list_profiles(user_id)
            conflicts = profiles.list_conflicts(user_id)
            retriever = GovernedProfileRetriever(profiles)
            before = retriever.retrieve(case.query, user_id=user_id)
            confirmed_recalled: bool | None = None
            deletion_passed: bool | None = None
            if case.expect_profile and not case.expect_conflict and proposed:
                profile = next(
                    profile
                    for profile in proposed
                    if profile.state is MemoryProfileState.AWAITING_CONFIRMATION
                )
                profiles.confirm_profile(profile.profile_id, user_id=user_id)
                confirmed_recalled = any(
                    result.memory_id == profile.profile_id
                    for result in retriever.retrieve(case.query, user_id=user_id)
                )
                source_memory_id = profile.evidence[0].memory_id
                repository.purge(source_memory_id, user_id=user_id)
                deletion_passed = profiles.list_active_profiles(user_id) == []
            return ConsolidationCaseResult(
                case_id=case.case_id,
                profile_expected=case.expect_profile,
                profile_observed=bool(proposed),
                conflict_expected=case.expect_conflict,
                conflict_observed=bool(conflicts),
                preconfirmation_retrieval_count=len(before),
                confirmed_profile_recalled=confirmed_recalled,
                deletion_cascade_passed=deletion_passed,
            )
