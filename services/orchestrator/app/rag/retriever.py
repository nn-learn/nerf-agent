import math
import re
from collections import Counter
from datetime import date
from typing import Protocol

from pydantic import BaseModel, Field

from app.rag.models import (
    KnowledgeChunk,
    KnowledgeDocument,
    chunk_document,
    load_reviewed_documents,
)
from app.safety.models import RiskLevel


class EmbeddingProvider(Protocol):
    dimensions: int

    def embed(self, text: str) -> list[float]: ...


class EmbeddingDimensionError(ValueError):
    pass


class EvidenceItem(BaseModel):
    chunk_id: str
    document_id: str
    document_version: str
    title: str
    text: str
    score: float = Field(ge=0, le=1)


class EvidenceBundle(BaseModel):
    query: str
    items: list[EvidenceItem]
    has_sufficient_evidence: bool


class _IndexedChunk:
    def __init__(
        self,
        chunk: KnowledgeChunk,
        vector: list[float],
        tokens: list[str],
    ) -> None:
        self.chunk = chunk
        self.vector = vector
        self.tokens = tokens


class HybridRetriever:
    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        evidence_threshold: float,
    ) -> None:
        if embedding_provider.dimensions != 1024:
            raise EmbeddingDimensionError(
                "BGE-M3 embedding contract requires exactly 1024 dimensions"
            )
        self.embedding_provider = embedding_provider
        self.evidence_threshold = evidence_threshold
        self._chunks: list[_IndexedChunk] = []

    def index(
        self,
        documents: list[KnowledgeDocument],
        *,
        as_of: date,
    ) -> None:
        indexed: list[_IndexedChunk] = []
        for document in load_reviewed_documents(documents, as_of=as_of):
            for chunk in chunk_document(document):
                vector = self.embedding_provider.embed(
                    f"{chunk.title}\n{chunk.text}"
                )
                self._assert_vector(vector)
                indexed.append(
                    _IndexedChunk(
                        chunk=chunk,
                        vector=vector,
                        tokens=_tokenize(f"{chunk.title} {chunk.text}"),
                    )
                )
        self._chunks = indexed

    def retrieve(
        self,
        query: str,
        *,
        risk_level: RiskLevel,
        k: int,
    ) -> EvidenceBundle:
        if k <= 0:
            return EvidenceBundle(
                query=query,
                items=[],
                has_sufficient_evidence=False,
            )
        candidates = [
            indexed
            for indexed in self._chunks
            if risk_level in indexed.chunk.allowed_risk_levels
        ]
        if not candidates:
            return EvidenceBundle(
                query=query,
                items=[],
                has_sufficient_evidence=False,
            )

        query_vector = self.embedding_provider.embed(query)
        self._assert_vector(query_vector)
        query_tokens = _tokenize(query)
        dense_scores = [
            _cosine_similarity(query_vector, candidate.vector)
            for candidate in candidates
        ]
        lexical_scores = _bm25_scores(query_tokens, candidates)
        dense_ranks = _ranks(dense_scores)
        lexical_ranks = _ranks(lexical_scores)
        max_lexical = max(lexical_scores, default=0.0)

        scored: list[tuple[float, float, _IndexedChunk]] = []
        for index, candidate in enumerate(candidates):
            rrf_score = _rrf(dense_ranks[index]) + _rrf(lexical_ranks[index])
            normalized_lexical = (
                lexical_scores[index] / max_lexical if max_lexical > 0 else 0.0
            )
            confidence = min(
                1.0,
                0.65 * max(0.0, dense_scores[index])
                + 0.35 * normalized_lexical,
            )
            scored.append((rrf_score, confidence, candidate))

        scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
        items = [
            EvidenceItem(
                chunk_id=candidate.chunk.chunk_id,
                document_id=candidate.chunk.document_id,
                document_version=candidate.chunk.document_version,
                title=candidate.chunk.title,
                text=candidate.chunk.text,
                score=confidence,
            )
            for _, confidence, candidate in scored[:k]
            if confidence > 0
        ]
        sufficient = bool(
            items and items[0].score >= self.evidence_threshold
        )
        return EvidenceBundle(
            query=query,
            items=items,
            has_sufficient_evidence=sufficient,
        )

    @staticmethod
    def _assert_vector(vector: list[float]) -> None:
        if len(vector) != 1024:
            raise EmbeddingDimensionError(
                f"expected 1024 embedding dimensions, got {len(vector)}"
            )


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.casefold())


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _bm25_scores(
    query_tokens: list[str],
    candidates: list[_IndexedChunk],
) -> list[float]:
    if not query_tokens:
        return [0.0] * len(candidates)
    document_frequency = Counter(
        token
        for token in set(query_tokens)
        for candidate in candidates
        if token in candidate.tokens
    )
    average_length = sum(len(candidate.tokens) for candidate in candidates) / len(
        candidates
    )
    scores: list[float] = []
    for candidate in candidates:
        frequencies = Counter(candidate.tokens)
        score = 0.0
        for token in set(query_tokens):
            frequency = frequencies[token]
            if frequency == 0:
                continue
            df = document_frequency[token]
            inverse_document_frequency = math.log(
                1 + (len(candidates) - df + 0.5) / (df + 0.5)
            )
            denominator = frequency + 1.5 * (
                1 - 0.75 + 0.75 * len(candidate.tokens) / average_length
            )
            score += inverse_document_frequency * (
                frequency * 2.5 / denominator
            )
        scores.append(score)
    return scores


def _ranks(scores: list[float]) -> list[int]:
    ordering = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )
    ranks = [0] * len(scores)
    for rank, index in enumerate(ordering, start=1):
        ranks[index] = rank
    return ranks


def _rrf(rank: int, constant: int = 60) -> float:
    return 1.0 / (constant + rank)
