from datetime import date

import pytest

from app.rag.bge import BgeM3EmbeddingProvider
from app.rag.evidence import EvidencePolicy, MissingEvidenceError
from app.rag.models import KnowledgeDocument
from app.rag.retriever import (
    EmbeddingDimensionError,
    HybridRetriever,
)
from app.safety.models import RiskLevel


class KeywordEmbeddingProvider:
    dimensions = 1024

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        vector[0] = 1.0 if "睡眠" in text else 0.0
        vector[1] = 1.0 if "呼吸" in text else 0.0
        vector[2] = 1.0 if "压力" in text else 0.0
        return vector


class WrongDimensionEmbeddingProvider:
    dimensions = 3

    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class RecordingSentenceEncoder:
    def __init__(self, dimensions: int = 1024) -> None:
        self.dimensions = dimensions
        self.calls: list[tuple[list[str], bool]] = []

    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool,
    ) -> list[list[float]]:
        self.calls.append((sentences, normalize_embeddings))
        return [[1.0] * self.dimensions]


def document(
    document_id: str,
    title: str,
    body: str,
    allowed_risk_levels: list[str],
) -> KnowledgeDocument:
    return KnowledgeDocument(
        document_id=document_id,
        title=title,
        version="1.0.0",
        reviewer="demo-clinical-reviewer",
        reviewed_at=date(2026, 7, 28),
        expires_at=date(2027, 7, 28),
        audience=["adult"],
        allowed_risk_levels=allowed_risk_levels,
        body=body,
    )


def test_hybrid_retrieval_returns_versioned_sleep_evidence() -> None:
    """Catches dense/lexical fusion losing the relevant reviewed source."""
    retriever = HybridRetriever(
        embedding_provider=KeywordEmbeddingProvider(),
        evidence_threshold=0.01,
    )
    retriever.index(
        [
            document(
                "sleep-1",
                "睡眠卫生",
                "固定作息有助于形成稳定的睡眠节律。",
                ["GREEN", "AMBER"],
            ),
            document(
                "breathing-1",
                "呼吸练习",
                "缓慢呼吸可以作为短时放松练习。",
                ["GREEN", "AMBER"],
            ),
        ],
        as_of=date(2026, 7, 28),
    )

    bundle = retriever.retrieve(
        "怎样改善睡眠",
        risk_level=RiskLevel.GREEN,
        k=2,
    )

    assert bundle.has_sufficient_evidence
    assert bundle.items[0].document_id == "sleep-1"
    assert bundle.items[0].document_version == "1.0.0"
    assert bundle.items[0].chunk_id == "sleep-1:0"


def test_risk_filter_removes_content_not_approved_for_red_state() -> None:
    """Catches normal self-help content leaking into a crisis response path."""
    retriever = HybridRetriever(
        embedding_provider=KeywordEmbeddingProvider(),
        evidence_threshold=0.01,
    )
    retriever.index(
        [
            document(
                "sleep-1",
                "睡眠卫生",
                "固定作息有助于形成稳定的睡眠节律。",
                ["GREEN", "AMBER"],
            )
        ],
        as_of=date(2026, 7, 28),
    )

    bundle = retriever.retrieve(
        "怎样改善睡眠",
        risk_level=RiskLevel.RED,
        k=2,
    )

    assert bundle.items == []
    assert not bundle.has_sufficient_evidence


def test_dense_embedding_dimension_is_fixed_to_bge_m3_contract() -> None:
    """Catches an embedding-model swap that silently corrupts the vector index."""
    with pytest.raises(EmbeddingDimensionError, match="1024"):
        HybridRetriever(
            embedding_provider=WrongDimensionEmbeddingProvider(),
            evidence_threshold=0.01,
        )


def test_bge_provider_requests_normalized_cpu_compatible_vectors() -> None:
    """Catches BGE calls that return unnormalized or non-1024-dimensional vectors."""
    encoder = RecordingSentenceEncoder()
    provider = BgeM3EmbeddingProvider(encoder=encoder)

    vector = provider.embed("睡眠规律")

    assert len(vector) == 1024
    assert encoder.calls == [(["睡眠规律"], True)]


def test_evidence_policy_rejects_unknown_citation_ids() -> None:
    """Catches generated medical claims citing evidence absent from this turn."""
    retriever = HybridRetriever(
        embedding_provider=KeywordEmbeddingProvider(),
        evidence_threshold=0.01,
    )
    retriever.index(
        [
            document(
                "sleep-1",
                "睡眠卫生",
                "固定作息有助于形成稳定的睡眠节律。",
                ["GREEN", "AMBER"],
            )
        ],
        as_of=date(2026, 7, 28),
    )
    bundle = retriever.retrieve(
        "怎样改善睡眠",
        risk_level=RiskLevel.GREEN,
        k=2,
    )

    with pytest.raises(MissingEvidenceError, match="unknown:0"):
        EvidencePolicy().validate_claim_ids(bundle, ["unknown:0"])
