import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from app.memory.consolidation import MemoryConsolidator, MemoryProfileRepository
from app.memory.models import (
    MemoryAspect,
    MemoryCandidate,
    MemoryChangeState,
    MemoryKind,
    MemoryProfileState,
)
from app.memory.repository import MemoryRepository

TemporalOutcome = Literal["CHANGE", "CONFLICT", "STABLE"]


class TemporalEvalCase(BaseModel):
    case_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    valid_at_ms: int = Field(ge=0)
    observed_at_ms: int = Field(ge=0)
    expected_outcome: TemporalOutcome
    revoke_before_decision: bool = False


class TemporalCaseResult(BaseModel):
    case_id: str
    expected_outcome: TemporalOutcome
    observed_outcome: TemporalOutcome
    current_profile_preserved_before_decision: bool | None = None
    valid_and_observed_time_preserved: bool
    bitemporal_boundary_passed: bool | None = None
    revoked_change_apply_succeeded: bool | None = None


class TemporalEvalReport(BaseModel):
    strategy: str = "memory-profile-v2.2-bitemporal"
    case_count: int = Field(ge=1)
    temporal_classification_accuracy: float = Field(ge=0, le=1)
    false_change_rate: float = Field(ge=0, le=1)
    predecision_current_preservation_rate: float = Field(ge=0, le=1)
    valid_observed_time_accuracy: float = Field(ge=0, le=1)
    bitemporal_boundary_accuracy: float = Field(ge=0, le=1)
    revoked_change_apply_rate: float = Field(ge=0, le=1)
    failed_case_ids: list[str]
    case_results: list[TemporalCaseResult]


def load_temporal_cases(path: Path) -> list[TemporalEvalCase]:
    return [
        TemporalEvalCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class MemoryTemporalEvaluator:
    """Synthetic temporal-governance regression; not a clinical-validity claim."""

    def evaluate(self, cases: list[TemporalEvalCase]) -> TemporalEvalReport:
        if not cases:
            raise ValueError("at least one temporal case is required")
        results = [self._evaluate_case(case) for case in cases]
        non_change = [
            result for result in results if result.expected_outcome != "CHANGE"
        ]
        change = [result for result in results if result.expected_outcome == "CHANGE"]
        applicable = [
            result
            for result, case in zip(results, cases, strict=True)
            if case.expected_outcome == "CHANGE" and not case.revoke_before_decision
        ]
        revoked = [
            result
            for result, case in zip(results, cases, strict=True)
            if case.revoke_before_decision
        ]
        failed = [
            result.case_id
            for result in results
            if result.observed_outcome != result.expected_outcome
            or not result.valid_and_observed_time_preserved
            or result.current_profile_preserved_before_decision is False
            or result.bitemporal_boundary_passed is False
            or result.revoked_change_apply_succeeded is True
        ]
        return TemporalEvalReport(
            case_count=len(results),
            temporal_classification_accuracy=sum(
                result.observed_outcome == result.expected_outcome for result in results
            )
            / len(results),
            false_change_rate=(
                sum(result.observed_outcome == "CHANGE" for result in non_change)
                / len(non_change)
                if non_change
                else 0.0
            ),
            predecision_current_preservation_rate=(
                sum(
                    result.current_profile_preserved_before_decision is True
                    for result in change
                )
                / len(change)
                if change
                else 1.0
            ),
            valid_observed_time_accuracy=sum(
                result.valid_and_observed_time_preserved for result in results
            )
            / len(results),
            bitemporal_boundary_accuracy=(
                sum(result.bitemporal_boundary_passed is True for result in applicable)
                / len(applicable)
                if applicable
                else 1.0
            ),
            revoked_change_apply_rate=(
                sum(result.revoked_change_apply_succeeded is True for result in revoked)
                / len(revoked)
                if revoked
                else 0.0
            ),
            failed_case_ids=failed,
            case_results=results,
        )

    @staticmethod
    def _candidate(text: str, turn_id: str, valid_at_ms: int) -> MemoryCandidate:
        return MemoryCandidate(
            source="transcript",
            contains_sensitive_content=False,
            text=text,
            kind=MemoryKind.SEMANTIC,
            source_turn_id=turn_id,
            aspect=MemoryAspect.PREFERENCE,
            subject_key="communication.response_style",
            confidence=0.9,
            valid_from_ms=valid_at_ms,
        )

    @classmethod
    def _evaluate_case(cls, case: TemporalEvalCase) -> TemporalCaseResult:
        with tempfile.TemporaryDirectory(prefix="psyavatar-memory-v22-") as directory:
            repository = MemoryRepository(Path(directory) / "memory.sqlite3")
            repository.initialize()
            profiles = MemoryProfileRepository(repository.database_path)
            consolidator = MemoryConsolidator(profiles)
            user_id = f"user_{case.case_id}"
            baseline = repository.add_active(
                user_id=user_id,
                candidate=cls._candidate("用户偏好：回答时简短", "turn_1", 1_000),
                now_ms=10_000,
            )
            profiles.record_observation(
                baseline,
                session_id="session_1",
                now_ms=10_000,
            )
            profiles.record_observation(
                baseline,
                session_id="session_2",
                now_ms=20_000,
            )
            proposal = consolidator.rebuild_user(user_id, now_ms=30_000)[0]
            old = profiles.confirm_profile(
                proposal.profile_id,
                user_id=user_id,
                now_ms=30_000,
            )
            evidence = repository.add_active(
                user_id=user_id,
                candidate=cls._candidate(case.text, "turn_3", case.valid_at_ms),
                now_ms=case.observed_at_ms,
            )
            observation = profiles.record_observation(
                evidence,
                session_id="session_3",
                now_ms=case.observed_at_ms,
            )
            consolidator.rebuild_user(user_id, now_ms=case.observed_at_ms + 1)
            changes = profiles.list_changes(user_id, state=MemoryChangeState.OPEN)
            conflicts = profiles.list_conflicts(user_id)
            observed: TemporalOutcome = (
                "CHANGE" if changes else "CONFLICT" if conflicts else "STABLE"
            )
            current_preserved: bool | None = None
            boundary: bool | None = None
            revoked_apply: bool | None = None
            if case.expected_outcome == "CHANGE":
                current_preserved = (
                    profiles.get_profile(old.profile_id, user_id=user_id).state
                    is MemoryProfileState.ACTIVE
                )
            if changes and case.revoke_before_decision:
                repository.revoke(evidence.memory_id, user_id=user_id)
                try:
                    profiles.apply_change(changes[0].change_id, user_id=user_id)
                except ValueError:
                    revoked_apply = False
                else:
                    revoked_apply = True
            elif changes:
                applied = profiles.apply_change(
                    changes[0].change_id,
                    user_id=user_id,
                    now_ms=case.observed_at_ms + 2,
                )
                before = profiles.list_active_profiles(
                    user_id,
                    as_of_ms=case.valid_at_ms - 1,
                )
                at_boundary = profiles.list_active_profiles(
                    user_id,
                    as_of_ms=case.valid_at_ms,
                )
                boundary = (
                    [profile.profile_id for profile in before] == [old.profile_id]
                    and [profile.profile_id for profile in at_boundary]
                    == [applied.proposed_profile_id]
                )
            return TemporalCaseResult(
                case_id=case.case_id,
                expected_outcome=case.expected_outcome,
                observed_outcome=observed,
                current_profile_preserved_before_decision=current_preserved,
                valid_and_observed_time_preserved=(
                    observation is not None
                    and observation.valid_at_ms == case.valid_at_ms
                    and observation.observed_at_ms == case.observed_at_ms
                ),
                bitemporal_boundary_passed=boundary,
                revoked_change_apply_succeeded=revoked_apply,
            )
