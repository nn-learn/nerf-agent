from pathlib import Path

from app.memory.models import (
    MemoryCandidate,
    MemoryDecision,
    MemoryKind,
)
from app.memory.policy import MemoryPolicy
from app.memory.repository import MemoryRepository


def test_sensitive_visual_observation_never_becomes_memory() -> None:
    """Catches addresses, faces or other sensitive camera context entering memory."""
    candidate = MemoryCandidate(
        source="visual",
        contains_sensitive_content=True,
        text="画面中出现家庭住址",
        kind=MemoryKind.SEMANTIC,
        source_turn_id="turn_1",
    )

    decision = MemoryPolicy().evaluate(candidate, consent_granted=True)

    assert decision is MemoryDecision.REJECT


def test_personalization_memory_requires_explicit_consent() -> None:
    """Catches ordinary preferences being persisted before user authorization."""
    candidate = MemoryCandidate(
        source="transcript",
        contains_sensitive_content=False,
        text="用户更喜欢简短回答",
        kind=MemoryKind.SEMANTIC,
        source_turn_id="turn_1",
    )

    decision = MemoryPolicy().evaluate(candidate, consent_granted=False)

    assert decision is MemoryDecision.REQUEST_CONSENT


def test_revoked_memory_is_not_returned_for_personalization(tmp_path: Path) -> None:
    """Catches revoked memory remaining visible to later Agent turns."""
    repository = MemoryRepository(tmp_path / "memory.sqlite3")
    repository.initialize()
    item = repository.add_active(
        user_id="user_1",
        candidate=MemoryCandidate(
            source="transcript",
            contains_sensitive_content=False,
            text="用户更喜欢简短回答",
            kind=MemoryKind.SEMANTIC,
            source_turn_id="turn_1",
        ),
    )

    repository.revoke(item.memory_id)

    assert repository.list_active("user_1") == []

