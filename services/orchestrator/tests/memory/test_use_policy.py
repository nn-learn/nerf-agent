from pathlib import Path

from app.memory.models import (
    MemoryAllowedUse,
    MemoryCandidate,
    MemoryKind,
    MemorySensitivity,
    MemorySourceType,
)
from app.memory.repository import MemoryRepository
from app.memory.retrieval import GovernedMemoryRetriever
from app.memory.use_policy import (
    MemoryUseContext,
    MemoryUseDecision,
    MemoryUsePolicy,
)
from app.safety.models import RiskLevel


def _item(
    tmp_path: Path,
    *,
    sensitivity: MemorySensitivity,
    allowed_uses: list[MemoryAllowedUse],
    source_type: MemorySourceType = MemorySourceType.USER_STATEMENT,
):
    repository = MemoryRepository(tmp_path / "memory.sqlite3")
    repository.initialize()
    item = repository.add_active(
        user_id="user_1",
        candidate=MemoryCandidate(
            source="transcript",
            contains_sensitive_content=(
                sensitivity is not MemorySensitivity.GENERAL
            ),
            text="用户最近睡眠不好",
            kind=MemoryKind.SEMANTIC,
            source_turn_id="turn_1",
            user_confirmed=True,
            source_type=source_type,
            sensitivity=sensitivity,
            allowed_uses=allowed_uses,
        ),
    )
    return repository, item


def test_health_memory_requires_user_led_explicit_revisit(tmp_path: Path) -> None:
    _, item = _item(
        tmp_path,
        sensitivity=MemorySensitivity.HEALTH_SENSITIVE,
        allowed_uses=[MemoryAllowedUse.RESPONSE_CONTEXT],
    )
    policy = MemoryUsePolicy()

    unsolicited = policy.evaluate(
        item,
        context=MemoryUseContext(user_initiated_recall=False),
    )
    vague = policy.evaluate(item, context=MemoryUseContext())
    explicit = policy.evaluate(
        item,
        context=MemoryUseContext(explicit_sensitive_revisit=True),
    )

    assert unsolicited.decision is MemoryUseDecision.BLOCK
    assert unsolicited.reason_codes == ["UNSOLICITED_HEALTH_MEMORY"]
    assert vague.reason_codes == ["EXPLICIT_SENSITIVE_REVISIT_REQUIRED"]
    assert explicit.decision is MemoryUseDecision.ALLOW


def test_historical_crisis_never_establishes_current_risk(tmp_path: Path) -> None:
    _, item = _item(
        tmp_path,
        sensitivity=MemorySensitivity.CRISIS_SENSITIVE,
        allowed_uses=[MemoryAllowedUse.SAFETY_SUPPORT],
    )
    policy = MemoryUsePolicy()

    low_risk = policy.evaluate(
        item,
        context=MemoryUseContext(
            requested_use=MemoryAllowedUse.SAFETY_SUPPORT,
            current_risk=RiskLevel.GREEN,
            current_turn_corroborates_crisis=False,
        ),
    )
    current_crisis = policy.evaluate(
        item,
        context=MemoryUseContext(
            requested_use=MemoryAllowedUse.SAFETY_SUPPORT,
            current_risk=RiskLevel.RED,
            current_turn_corroborates_crisis=True,
        ),
    )

    assert low_risk.reason_codes == ["HISTORICAL_CRISIS_IS_NOT_CURRENT_RISK"]
    assert current_crisis.decision is MemoryUseDecision.ALLOW


def test_retriever_applies_policy_after_candidate_selection(tmp_path: Path) -> None:
    repository, item = _item(
        tmp_path,
        sensitivity=MemorySensitivity.HEALTH_SENSITIVE,
        allowed_uses=[MemoryAllowedUse.RESPONSE_CONTEXT],
    )
    retriever = GovernedMemoryRetriever(repository, min_score=0)

    blocked = retriever.retrieve(
        "最近睡眠怎么样",
        user_id="user_1",
        use_context=MemoryUseContext(explicit_sensitive_revisit=False),
    )
    allowed = retriever.retrieve(
        "最近睡眠怎么样",
        user_id="user_1",
        use_context=MemoryUseContext(explicit_sensitive_revisit=True),
    )

    assert blocked == []
    assert [memory.memory_id for memory in allowed] == [item.memory_id]
    assert allowed[0].sensitivity == "HEALTH_SENSITIVE"
