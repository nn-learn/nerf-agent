from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.rag.ingest import load_manifest
from app.rag.models import KnowledgeDocument, load_reviewed_documents


def reviewed_document(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "document_id": "sleep-1",
        "title": "睡眠卫生演示",
        "version": "1.0.0",
        "reviewer": "demo-clinical-reviewer",
        "reviewed_at": date(2026, 7, 28),
        "expires_at": date(2027, 7, 28),
        "audience": ["adult"],
        "allowed_risk_levels": ["GREEN", "AMBER"],
        "body": "固定作息和睡前减少刺激有助于形成稳定的睡眠节律。",
    }
    payload.update(overrides)
    return payload


def test_unreviewed_document_is_rejected() -> None:
    """Catches knowledge entering the index without an accountable reviewer."""
    with pytest.raises(ValidationError, match="reviewer"):
        KnowledgeDocument.model_validate(reviewed_document(reviewer=""))


def test_expired_document_is_excluded_from_reviewed_load() -> None:
    """Catches expired medical education content remaining retrievable."""
    active = KnowledgeDocument.model_validate(reviewed_document())
    expired = KnowledgeDocument.model_validate(
        reviewed_document(
            document_id="sleep-old",
            expires_at=date(2026, 7, 27),
        )
    )

    loaded = load_reviewed_documents([active, expired], as_of=date(2026, 7, 28))

    assert [document.document_id for document in loaded] == ["sleep-1"]


def test_document_version_must_be_semantic() -> None:
    """Catches untraceable free-form document versions."""
    with pytest.raises(ValidationError, match="semantic version"):
        KnowledgeDocument.model_validate(reviewed_document(version="latest"))


def test_repository_manifest_loads_reviewed_markdown_body() -> None:
    """Catches manifest metadata and reviewed source files drifting apart."""
    project_root = Path(__file__).parents[4]

    documents = load_manifest(project_root / "knowledge" / "manifest.yaml")

    assert len(documents) == 1
    assert documents[0].document_id == "demo-sleep-hygiene"
    assert "固定作息" in documents[0].body
