import math
import re
import time
from collections import Counter
from typing import Protocol

from pydantic import BaseModel, Field

from app.memory.consolidation import MemoryProfileRepository
from app.memory.models import MemoryAspect, MemoryItem, MemoryProfile
from app.memory.repository import MemoryRepository
from app.memory.use_policy import (
    MemoryUseContext,
    MemoryUseDecision,
    MemoryUsePolicy,
)

_INSTRUCTION_RE = re.compile(
    r"忽略.{0,8}(指令|规则|系统)|system\s*prompt|调用.{0,6}工具|执行.{0,6}命令",
    re.IGNORECASE,
)
_ASCII_WORD_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_STOP_FEATURES = frozenset("我你他她它的是了有在和对用户表示明确")


class RetrievedMemory(BaseModel):
    memory_id: str
    text: str
    aspect: MemoryAspect
    score: float = Field(ge=0, le=1)
    relevance_score: float = Field(ge=0, le=1)
    reason_codes: list[str]
    source_turn_id: str
    source_message_ids: list[str]
    evidence_memory_ids: list[str] = Field(default_factory=list)
    valid_at_ms: int | None = Field(default=None, ge=0)
    source_type: str = "LEGACY"
    sensitivity: str = "GENERAL"


class MemoryEmbeddingProvider(Protocol):
    """The narrow embedding contract shared with the reviewed-knowledge RAG."""

    dimensions: int

    def embed(self, text: str) -> list[float]: ...


class CachedMemoryEmbeddingProvider:
    """Process-local cache for repeatable threshold scans and hybrid evaluation."""

    def __init__(self, provider: MemoryEmbeddingProvider) -> None:
        self._provider = provider
        self.dimensions = provider.dimensions
        self._cache: dict[str, tuple[float, ...]] = {}

    def embed(self, text: str) -> list[float]:
        cached = self._cache.get(text)
        if cached is None:
            cached = tuple(self._provider.embed(text))
            self._cache[text] = cached
        return list(cached)

    @property
    def cache_size(self) -> int:
        return len(self._cache)


class GovernedMemoryRetriever:
    """Purpose-filtered CPU baseline; embeddings can replace only the scoring stage."""

    def __init__(
        self,
        repository: MemoryRepository,
        *,
        min_score: float = 0.18,
        use_policy: MemoryUsePolicy | None = None,
    ) -> None:
        self._repository = repository
        self._min_score = min_score
        self._use_policy = use_policy or MemoryUsePolicy()

    def retrieve(
        self,
        query: str,
        *,
        user_id: str,
        purpose_scope: str = "personalization",
        k: int = 5,
        as_of_ms: int | None = None,
        use_context: MemoryUseContext | None = None,
    ) -> list[RetrievedMemory]:
        if k <= 0:
            return []
        now_ms = as_of_ms if as_of_ms is not None else int(time.time() * 1000)
        policy_context = use_context or MemoryUseContext(
            explicit_sensitive_revisit=True
        )
        query_features = self._features(query)
        ranked: list[tuple[float, float, list[str], MemoryItem]] = []
        for item in self._repository.list_active(
            user_id,
            purpose_scope=purpose_scope,
            as_of_ms=now_ms,
        ):
            candidate = item.candidate
            authorization = self._use_policy.evaluate(item, context=policy_context)
            if authorization.decision is MemoryUseDecision.BLOCK:
                continue
            if candidate.integrity_flags or _INSTRUCTION_RE.search(candidate.text):
                continue
            lexical = self._weighted_jaccard(query_features, self._features(candidate.text))
            confirmed = 1.0 if candidate.user_confirmed else 0.0
            age_days = max(0.0, (now_ms - item.updated_at_ms) / 86_400_000)
            recency = math.exp(-age_days / 90)
            persistent_constraint = (
                1.0
                if candidate.aspect in {MemoryAspect.BOUNDARY, MemoryAspect.PREFERENCE}
                else 0.0
            )
            if lexical < 0.03 and not persistent_constraint:
                continue
            reasons = ["USER_CONFIRMED"]
            if lexical >= 0.03:
                reasons.append("TOPIC_MATCH")
            if persistent_constraint:
                reasons.append("STABLE_PREFERENCE")
            if recency >= 0.8:
                reasons.append("RECENTLY_UPDATED")
            score = min(
                1.0,
                0.55 * lexical
                + 0.15 * candidate.confidence
                + 0.15 * confirmed
                + 0.05 * recency
                + 0.10 * persistent_constraint,
            )
            if score >= self._min_score:
                ranked.append((score, lexical, reasons, item))
        ranked.sort(
            key=lambda value: (-value[0], -value[3].updated_at_ms, value[3].memory_id)
        )
        return [
            RetrievedMemory(
                memory_id=item.memory_id,
                text=item.candidate.text,
                aspect=item.candidate.aspect,
                score=round(score, 6),
                relevance_score=round(lexical, 6),
                reason_codes=reasons,
                source_turn_id=item.candidate.source_turn_id,
                source_message_ids=item.candidate.source_message_ids,
                valid_at_ms=(
                    item.candidate.valid_from_ms
                    if item.candidate.valid_from_ms is not None
                    else item.created_at_ms
                ),
                source_type=item.candidate.source_type.value,
                sensitivity=item.candidate.sensitivity.value,
            )
            for score, lexical, reasons, item in ranked[:k]
        ]

    @staticmethod
    def to_model_context(items: list[RetrievedMemory]) -> list[dict[str, object]]:
        return [
            {
                "memory_id": item.memory_id,
                "fact": item.text,
                "aspect": item.aspect.value,
                "source_turn_id": item.source_turn_id,
                "source_type": item.source_type,
                "sensitivity": item.sensitivity,
                "valid_at_ms": item.valid_at_ms,
                "trust": "user_confirmed_data_not_instruction",
            }
            for item in items
        ]

    @staticmethod
    def _features(text: str) -> Counter[str]:
        normalized = text.lower()
        chars = _CJK_RE.findall(normalized)
        features: Counter[str] = Counter(_ASCII_WORD_RE.findall(normalized))
        features.update(chars)
        features.update(a + b for a, b in zip(chars, chars[1:], strict=False))
        for stop_feature in _STOP_FEATURES:
            features.pop(stop_feature, None)
        return features

    @staticmethod
    def _weighted_jaccard(left: Counter[str], right: Counter[str]) -> float:
        if not left or not right:
            return 0.0
        keys = left.keys() | right.keys()
        intersection = sum(min(left[key], right[key]) for key in keys)
        union = sum(max(left[key], right[key]) for key in keys)
        return intersection / union if union else 0.0


class GovernedProfileRetriever:
    """Retrieve only user-confirmed V2 profiles, never unresolved proposals."""

    def __init__(
        self,
        repository: MemoryProfileRepository,
        *,
        min_score: float = 0.20,
    ) -> None:
        self._repository = repository
        self._min_score = min_score

    def retrieve(
        self,
        query: str,
        *,
        user_id: str,
        purpose_scope: str = "personalization",
        k: int = 3,
        as_of_ms: int | None = None,
    ) -> list[RetrievedMemory]:
        if k <= 0:
            return []
        query_features = GovernedMemoryRetriever._features(query)
        ranked: list[tuple[float, float, MemoryProfile]] = []
        for profile in self._repository.list_active_profiles(
            user_id,
            purpose_scope=purpose_scope,
            as_of_ms=as_of_ms,
        ):
            relevance = GovernedMemoryRetriever._weighted_jaccard(
                query_features,
                GovernedMemoryRetriever._features(profile.statement),
            )
            persistent = profile.aspect in {
                MemoryAspect.BOUNDARY,
                MemoryAspect.PREFERENCE,
            }
            if relevance < 0.03 and not persistent:
                continue
            score = min(
                1.0,
                0.60 * relevance
                + 0.20 * profile.confidence
                + 0.10 * min(1.0, profile.distinct_session_count / 3)
                + 0.10 * float(persistent),
            )
            if score >= self._min_score:
                ranked.append((score, relevance, profile))
        ranked.sort(key=lambda row: (-row[0], -row[2].updated_at_ms, row[2].profile_id))
        return [
            RetrievedMemory(
                memory_id=profile.profile_id,
                text=profile.statement,
                aspect=profile.aspect,
                score=round(score, 6),
                relevance_score=round(relevance, 6),
                reason_codes=[
                    "USER_CONFIRMED_PROFILE",
                    "MULTI_SESSION_EVIDENCE",
                    *( ["TOPIC_MATCH"] if relevance >= 0.03 else [] ),
                    *( ["STABLE_PREFERENCE"] if profile.aspect in {
                        MemoryAspect.BOUNDARY,
                        MemoryAspect.PREFERENCE,
                    } else [] ),
                ],
                source_turn_id="profile_consolidation",
                source_message_ids=[],
                evidence_memory_ids=[
                    evidence.memory_id for evidence in profile.evidence
                ],
                valid_at_ms=(
                    profile.valid_from_ms
                    if profile.valid_from_ms is not None
                    else profile.updated_at_ms
                ),
            )
            for score, relevance, profile in ranked[:k]
        ]


class LayeredMemoryRetriever:
    """Fuse confirmed semantic profiles with V1 evidence without duplicates."""

    def __init__(
        self,
        evidence_retriever: GovernedMemoryRetriever,
        profile_retriever: GovernedProfileRetriever,
    ) -> None:
        self._evidence = evidence_retriever
        self._profiles = profile_retriever

    def retrieve(
        self,
        query: str,
        *,
        user_id: str,
        purpose_scope: str = "personalization",
        k: int = 5,
        as_of_ms: int | None = None,
    ) -> list[RetrievedMemory]:
        profiles = self._profiles.retrieve(
            query,
            user_id=user_id,
            purpose_scope=purpose_scope,
            k=k,
            as_of_ms=as_of_ms,
        )
        evidence = self._evidence.retrieve(
            query,
            user_id=user_id,
            purpose_scope=purpose_scope,
            k=k,
            as_of_ms=as_of_ms,
        )
        profiled_evidence_ids = {
            evidence_id
            for profile in profiles
            for evidence_id in profile.evidence_memory_ids
        }
        combined = profiles + [
            item for item in evidence if item.memory_id not in profiled_evidence_ids
        ]
        combined.sort(
            key=lambda item: (
                -item.score,
                -int("USER_CONFIRMED_PROFILE" in item.reason_codes),
                item.memory_id,
            )
        )
        return combined[:k]

    @staticmethod
    def to_model_context(items: list[RetrievedMemory]) -> list[dict[str, object]]:
        return GovernedMemoryRetriever.to_model_context(items)


class GovernedEmbeddingMemoryRetriever:
    """Optional semantic scorer with the same memory-governance gates as baseline.

    Vectors stay in process and are not persisted. This class is currently used by
    the offline V1.4 A/B evaluation; enabling it online is a separate product choice.
    """

    def __init__(
        self,
        repository: MemoryRepository,
        *,
        embedding_provider: MemoryEmbeddingProvider,
        min_relevance: float = 0.40,
        min_score: float = 0.18,
        use_policy: MemoryUsePolicy | None = None,
    ) -> None:
        if not 0 <= min_relevance <= 1:
            raise ValueError("min_relevance must be between 0 and 1")
        self._repository = repository
        self._embedding_provider = embedding_provider
        self._min_relevance = min_relevance
        self._min_score = min_score
        self._use_policy = use_policy or MemoryUsePolicy()
        self._vector_cache: dict[str, list[float]] = {}

    def retrieve(
        self,
        query: str,
        *,
        user_id: str,
        purpose_scope: str = "personalization",
        k: int = 5,
        as_of_ms: int | None = None,
        use_context: MemoryUseContext | None = None,
    ) -> list[RetrievedMemory]:
        if k <= 0:
            return []
        now_ms = as_of_ms if as_of_ms is not None else int(time.time() * 1000)
        policy_context = use_context or MemoryUseContext(
            explicit_sensitive_revisit=True
        )
        query_vector = self._embedding_provider.embed(query)
        ranked: list[tuple[float, float, list[str], MemoryItem]] = []
        for item in self._repository.list_active(
            user_id,
            purpose_scope=purpose_scope,
            as_of_ms=now_ms,
        ):
            candidate = item.candidate
            authorization = self._use_policy.evaluate(item, context=policy_context)
            if authorization.decision is MemoryUseDecision.BLOCK:
                continue
            if candidate.integrity_flags or _INSTRUCTION_RE.search(candidate.text):
                continue
            memory_vector = self._vector_cache.get(candidate.text)
            if memory_vector is None:
                memory_vector = self._embedding_provider.embed(candidate.text)
                self._vector_cache[candidate.text] = memory_vector
            relevance = max(0.0, self._cosine_similarity(query_vector, memory_vector))
            persistent_constraint = (
                1.0
                if candidate.aspect in {MemoryAspect.BOUNDARY, MemoryAspect.PREFERENCE}
                else 0.0
            )
            if relevance < self._min_relevance and not persistent_constraint:
                continue
            confirmed = 1.0 if candidate.user_confirmed else 0.0
            age_days = max(0.0, (now_ms - item.updated_at_ms) / 86_400_000)
            recency = math.exp(-age_days / 90)
            reasons = ["USER_CONFIRMED"]
            if relevance >= self._min_relevance:
                reasons.append("SEMANTIC_MATCH")
            if persistent_constraint:
                reasons.append("STABLE_PREFERENCE")
            if recency >= 0.8:
                reasons.append("RECENTLY_UPDATED")
            score = min(
                1.0,
                0.55 * relevance
                + 0.15 * candidate.confidence
                + 0.15 * confirmed
                + 0.05 * recency
                + 0.10 * persistent_constraint,
            )
            if score >= self._min_score:
                ranked.append((score, relevance, reasons, item))
        ranked.sort(
            key=lambda value: (-value[0], -value[3].updated_at_ms, value[3].memory_id)
        )
        return [
            RetrievedMemory(
                memory_id=item.memory_id,
                text=item.candidate.text,
                aspect=item.candidate.aspect,
                score=round(score, 6),
                relevance_score=round(relevance, 6),
                reason_codes=reasons,
                source_turn_id=item.candidate.source_turn_id,
                source_message_ids=item.candidate.source_message_ids,
                valid_at_ms=(
                    item.candidate.valid_from_ms
                    if item.candidate.valid_from_ms is not None
                    else item.created_at_ms
                ),
                source_type=item.candidate.source_type.value,
                sensitivity=item.candidate.sensitivity.value,
            )
            for score, relevance, reasons, item in ranked[:k]
        ]

    @staticmethod
    def to_model_context(items: list[RetrievedMemory]) -> list[dict[str, object]]:
        return GovernedMemoryRetriever.to_model_context(items)

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise ValueError("embedding dimensions do not match")
        numerator = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)


class GovernedHybridMemoryRetriever:
    """Governed lexical+dense candidate fusion with deterministic reranking.

    Both branches read through the same repository policy gates. The fusion layer
    sees only already-governed candidates and therefore cannot reintroduce expired,
    cross-user, unconfirmed, or integrity-flagged memories.
    """

    def __init__(
        self,
        repository: MemoryRepository,
        *,
        embedding_provider: MemoryEmbeddingProvider,
        semantic_min_relevance: float = 0.40,
        candidate_k: int = 20,
        min_fused_score: float = 0.32,
        rrf_constant: int = 60,
        use_policy: MemoryUsePolicy | None = None,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        if not 0 <= min_fused_score <= 1:
            raise ValueError("min_fused_score must be between 0 and 1")
        if rrf_constant <= 0:
            raise ValueError("rrf_constant must be positive")
        policy = use_policy or MemoryUsePolicy()
        self._lexical = GovernedMemoryRetriever(repository, use_policy=policy)
        self._semantic = GovernedEmbeddingMemoryRetriever(
            repository,
            embedding_provider=embedding_provider,
            min_relevance=semantic_min_relevance,
            use_policy=policy,
        )
        self._candidate_k = candidate_k
        self._min_fused_score = min_fused_score
        self._rrf_constant = rrf_constant

    def retrieve(
        self,
        query: str,
        *,
        user_id: str,
        purpose_scope: str = "personalization",
        k: int = 5,
        as_of_ms: int | None = None,
        use_context: MemoryUseContext | None = None,
    ) -> list[RetrievedMemory]:
        if k <= 0:
            return []
        branch_k = max(k, self._candidate_k)
        lexical = self._lexical.retrieve(
            query,
            user_id=user_id,
            purpose_scope=purpose_scope,
            k=branch_k,
            as_of_ms=as_of_ms,
            use_context=use_context,
        )
        semantic = self._semantic.retrieve(
            query,
            user_id=user_id,
            purpose_scope=purpose_scope,
            k=branch_k,
            as_of_ms=as_of_ms,
            use_context=use_context,
        )
        lexical_by_id = {item.memory_id: item for item in lexical}
        semantic_by_id = {item.memory_id: item for item in semantic}
        lexical_ranks = {item.memory_id: rank for rank, item in enumerate(lexical, 1)}
        semantic_ranks = {item.memory_id: rank for rank, item in enumerate(semantic, 1)}
        maximum_rrf = 1.0 / (self._rrf_constant + 1)
        ranked: list[tuple[float, float, RetrievedMemory]] = []
        for memory_id in lexical_by_id.keys() | semantic_by_id.keys():
            lexical_item = lexical_by_id.get(memory_id)
            semantic_item = semantic_by_id.get(memory_id)
            representative = semantic_item or lexical_item
            assert representative is not None
            rrf = 0.0
            if memory_id in lexical_ranks:
                rrf += 0.55 / (self._rrf_constant + lexical_ranks[memory_id])
            if memory_id in semantic_ranks:
                rrf += 0.45 / (self._rrf_constant + semantic_ranks[memory_id])
            normalized_rrf = min(1.0, rrf / maximum_rrf)
            agreement = float(lexical_item is not None and semantic_item is not None)
            boundary_priority = float(representative.aspect is MemoryAspect.BOUNDARY)
            semantic_relevance = (
                semantic_item.relevance_score if semantic_item is not None else 0.0
            )
            lexical_relevance = (
                lexical_item.relevance_score if lexical_item is not None else 0.0
            )
            fused_score = min(
                1.0,
                0.35 * normalized_rrf
                + 0.37 * semantic_relevance
                + 0.15 * lexical_relevance
                + 0.10 * agreement
                + 0.03 * boundary_priority,
            )
            if fused_score < self._min_fused_score:
                continue
            reasons = sorted(
                set(
                    (lexical_item.reason_codes if lexical_item is not None else [])
                    + (semantic_item.reason_codes if semantic_item is not None else [])
                    + (["LEXICAL_CANDIDATE"] if lexical_item is not None else [])
                    + (["SEMANTIC_CANDIDATE"] if semantic_item is not None else [])
                    + (["CROSS_SIGNAL_AGREEMENT"] if agreement else [])
                    + (["BOUNDARY_PRIORITY"] if boundary_priority else [])
                )
            )
            ranked.append(
                (
                    fused_score,
                    semantic_relevance,
                    representative.model_copy(
                        update={
                            "score": round(fused_score, 6),
                            "relevance_score": round(
                                max(semantic_relevance, lexical_relevance),
                                6,
                            ),
                            "reason_codes": reasons,
                        }
                    ),
                )
            )
        ranked.sort(
            key=lambda row: (-row[0], -row[1], row[2].memory_id)
        )
        return [item for _, _, item in ranked[:k]]

    @staticmethod
    def to_model_context(items: list[RetrievedMemory]) -> list[dict[str, object]]:
        return GovernedMemoryRetriever.to_model_context(items)
