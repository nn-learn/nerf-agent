import hashlib
import json
import math
import sqlite3
import time
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from typing import Protocol

from app.memory.models import (
    MemoryAspect,
    MemoryEpisodeSummary,
    MemoryItem,
)
from app.memory.repository import MemoryRepository
from app.memory.retrieval import GovernedMemoryRetriever, RetrievedMemory

EPISODE_SUMMARY_VERSION = "memory-episode-v2.3"


class MemoryRetrieverPort(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        user_id: str,
        purpose_scope: str = "personalization",
        k: int = 5,
        as_of_ms: int | None = None,
    ) -> list[RetrievedMemory]: ...

    def to_model_context(
        self,
        items: list[RetrievedMemory],
    ) -> list[dict[str, object]]: ...


class MemoryEpisodeRepository:
    """Build bounded, deterministic episode indexes off the realtime path."""

    def __init__(
        self,
        database_path: Path,
        *,
        max_members: int = 6,
        max_summary_chars: int = 800,
        time_gap_ms: int = 30 * 60 * 1000,
    ) -> None:
        self.database_path = database_path
        if max_members <= 0 or max_summary_chars <= 0 or time_gap_ms <= 0:
            raise ValueError("episode summary limits must be positive")
        self.max_members = max_members
        self.max_summary_chars = max_summary_chars
        self.time_gap_ms = time_gap_ms

    def rebuild_session(
        self,
        *,
        user_id: str,
        session_id: str,
        memory_repository: MemoryRepository,
        purpose_scope: str = "personalization",
        now_ms: int | None = None,
    ) -> list[MemoryEpisodeSummary]:
        timestamp = now_ms if now_ms is not None else int(time.time() * 1000)
        active = {
            item.memory_id: item
            for item in memory_repository.list_active(
                user_id,
                purpose_scope=purpose_scope,
                as_of_ms=timestamp,
            )
            if not item.candidate.integrity_flags
        }
        with closing(sqlite3.connect(self.database_path)) as connection:
            rows = connection.execute(
                """
                SELECT memory_id, MIN(valid_at_ms), MAX(valid_at_ms)
                FROM memory_observations
                WHERE user_id = ? AND session_id = ?
                GROUP BY memory_id
                ORDER BY MIN(valid_at_ms), MIN(observed_at_ms), memory_id
                """,
                (user_id, session_id),
            ).fetchall()
        episode_items = [
            (active[str(row[0])], int(row[1]), int(row[2]))
            for row in rows
            if str(row[0]) in active
        ]
        groups = self._partition(episode_items)
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.execute("PRAGMA secure_delete = ON")
            old_ids = connection.execute(
                """
                SELECT summary_id FROM memory_episode_summaries
                WHERE user_id = ? AND session_id = ? AND purpose_scope = ?
                """,
                (user_id, session_id, purpose_scope),
            ).fetchall()
            for (summary_id,) in old_ids:
                connection.execute(
                    "DELETE FROM memory_episode_members WHERE summary_id = ?",
                    (summary_id,),
                )
            connection.execute(
                """
                DELETE FROM memory_episode_summaries
                WHERE user_id = ? AND session_id = ? AND purpose_scope = ?
                """,
                (user_id, session_id, purpose_scope),
            )
            for group in groups:
                source_digest = self._source_digest(group)
                summary_id = "episode_" + hashlib.sha256(
                    f"{user_id}:{session_id}:{purpose_scope}:{source_digest}".encode()
                ).hexdigest()[:32]
                text = "；".join(item.candidate.text for item, _, _ in group)
                if len(text) > self.max_summary_chars:
                    text = text[: self.max_summary_chars - 1].rstrip() + "…"
                connection.execute(
                    """
                    INSERT INTO memory_episode_summaries(
                        summary_id, user_id, session_id, purpose_scope,
                        summary_text, started_at_ms, ended_at_ms,
                        contains_sensitive_content, source_digest, summary_version,
                        created_at_ms, updated_at_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        summary_id,
                        user_id,
                        session_id,
                        purpose_scope,
                        text,
                        min(valid_from for _, valid_from, _ in group),
                        max(valid_to for _, _, valid_to in group),
                        int(
                            any(
                                item.candidate.contains_sensitive_content
                                for item, _, _ in group
                            )
                        ),
                        source_digest,
                        EPISODE_SUMMARY_VERSION,
                        timestamp,
                        timestamp,
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO memory_episode_members(summary_id, memory_id, ordinal)
                    VALUES (?, ?, ?)
                    """,
                    [
                        (summary_id, item.memory_id, ordinal)
                        for ordinal, (item, _, _) in enumerate(group)
                    ],
                )
        return self.list_for_user(user_id, purpose_scope=purpose_scope)

    def rebuild_for_memory(
        self,
        *,
        user_id: str,
        memory_id: str,
        memory_repository: MemoryRepository,
        now_ms: int | None = None,
    ) -> list[MemoryEpisodeSummary]:
        with closing(sqlite3.connect(self.database_path)) as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT session_id FROM memory_observations
                WHERE user_id = ? AND memory_id = ?
                ORDER BY session_id
                """,
                (user_id, memory_id),
            ).fetchall()
        rebuilt: list[MemoryEpisodeSummary] = []
        for (session_id,) in rows:
            rebuilt.extend(
                self.rebuild_session(
                    user_id=user_id,
                    session_id=str(session_id),
                    memory_repository=memory_repository,
                    now_ms=now_ms,
                )
            )
        return rebuilt

    def list_for_user(
        self,
        user_id: str,
        *,
        purpose_scope: str = "personalization",
    ) -> list[MemoryEpisodeSummary]:
        with closing(sqlite3.connect(self.database_path)) as connection:
            rows = connection.execute(
                """
                SELECT summary_id, user_id, session_id, summary_text,
                       purpose_scope, started_at_ms, ended_at_ms,
                       contains_sensitive_content, source_digest, summary_version,
                       created_at_ms, updated_at_ms
                FROM memory_episode_summaries
                WHERE user_id = ? AND purpose_scope = ?
                ORDER BY ended_at_ms DESC, summary_id
                """,
                (user_id, purpose_scope),
            ).fetchall()
            members = {
                str(row[0]): [
                    str(member[0])
                    for member in connection.execute(
                        """
                        SELECT memory_id FROM memory_episode_members
                        WHERE summary_id = ? ORDER BY ordinal, memory_id
                        """,
                        (str(row[0]),),
                    ).fetchall()
                ]
                for row in rows
            }
        return [
            MemoryEpisodeSummary(
                summary_id=str(row[0]),
                user_id=str(row[1]),
                session_id=str(row[2]),
                text=str(row[3]),
                purpose_scope=str(row[4]),
                member_memory_ids=members[str(row[0])],
                started_at_ms=int(row[5]),
                ended_at_ms=int(row[6]),
                contains_sensitive_content=bool(row[7]),
                source_digest=str(row[8]),
                summary_version=str(row[9]),
                created_at_ms=int(row[10]),
                updated_at_ms=int(row[11]),
            )
            for row in rows
        ]

    def _partition(
        self,
        items: Sequence[tuple[MemoryItem, int, int]],
    ) -> list[list[tuple[MemoryItem, int, int]]]:
        groups: list[list[tuple[MemoryItem, int, int]]] = []
        current: list[tuple[MemoryItem, int, int]] = []
        current_chars = 0
        for row in items:
            item, valid_from, _ = row
            item_chars = min(len(item.candidate.text), self.max_summary_chars)
            should_flush = bool(current) and (
                len(current) >= self.max_members
                or current_chars + item_chars + 1 > self.max_summary_chars
                or valid_from - current[-1][2] > self.time_gap_ms
            )
            if should_flush:
                groups.append(current)
                current = []
                current_chars = 0
            current.append(row)
            current_chars += item_chars + int(bool(current_chars))
        if current:
            groups.append(current)
        return groups

    @staticmethod
    def _source_digest(group: Sequence[tuple[MemoryItem, int, int]]) -> str:
        payload = [
            {
                "memory_id": item.memory_id,
                "updated_at_ms": item.updated_at_ms,
                "valid_from_ms": valid_from,
                "valid_to_ms": valid_to,
            }
            for item, valid_from, valid_to in group
        ]
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class EpisodeAugmentedMemoryRetriever:
    """Use summaries only for navigation, then return governed source memories."""

    def __init__(
        self,
        base: MemoryRetrieverPort,
        *,
        episodes: MemoryEpisodeRepository,
        memories: MemoryRepository,
        min_episode_relevance: float = 0.03,
        candidate_episodes: int = 3,
    ) -> None:
        self._base = base
        self._episodes = episodes
        self._memories = memories
        self._min_episode_relevance = min_episode_relevance
        self._candidate_episodes = candidate_episodes

    def retrieve(
        self,
        query: str,
        *,
        user_id: str,
        purpose_scope: str = "personalization",
        k: int = 5,
        as_of_ms: int | None = None,
    ) -> list[RetrievedMemory]:
        if k <= 0:
            return []
        base = self._base.retrieve(
            query,
            user_id=user_id,
            purpose_scope=purpose_scope,
            k=max(k, 8),
            as_of_ms=as_of_ms,
        )
        active = {
            item.memory_id: item
            for item in self._memories.list_active(
                user_id,
                purpose_scope=purpose_scope,
                as_of_ms=as_of_ms,
            )
            if not item.candidate.integrity_flags
        }
        profiled = {
            member_id for item in base for member_id in item.evidence_memory_ids
        }
        by_id = {item.memory_id: item for item in base}
        query_features = GovernedMemoryRetriever._features(query)
        ranked_episodes: list[tuple[float, MemoryEpisodeSummary]] = []
        for episode in self._episodes.list_for_user(
            user_id,
            purpose_scope=purpose_scope,
        ):
            relevance = GovernedMemoryRetriever._weighted_jaccard(
                query_features,
                GovernedMemoryRetriever._features(episode.text),
            )
            if relevance >= self._min_episode_relevance:
                ranked_episodes.append((relevance, episode))
        ranked_episodes.sort(key=lambda row: (-row[0], -row[1].ended_at_ms))
        for episode_relevance, episode in ranked_episodes[: self._candidate_episodes]:
            for memory_id in episode.member_memory_ids:
                item = active.get(memory_id)
                if item is None or memory_id in by_id or memory_id in profiled:
                    continue
                direct_relevance = GovernedMemoryRetriever._weighted_jaccard(
                    query_features,
                    GovernedMemoryRetriever._features(item.candidate.text),
                )
                score = min(
                    1.0,
                    0.40 * episode_relevance
                    + 0.25 * direct_relevance
                    + 0.20 * item.candidate.confidence
                    + 0.15,
                )
                by_id[memory_id] = RetrievedMemory(
                    memory_id=memory_id,
                    text=item.candidate.text,
                    aspect=item.candidate.aspect,
                    score=round(score, 6),
                    relevance_score=round(
                        max(episode_relevance, direct_relevance),
                        6,
                    ),
                    reason_codes=[
                        "USER_CONFIRMED",
                        "EPISODE_SUMMARY_EXPANSION",
                        "GOVERNED_SOURCE_EXPANSION",
                    ],
                    source_turn_id=item.candidate.source_turn_id,
                    source_message_ids=item.candidate.source_message_ids,
                    valid_at_ms=(
                        item.candidate.valid_from_ms
                        if item.candidate.valid_from_ms is not None
                        else item.created_at_ms
                    ),
                )
        ranked = list(by_id.values())
        ranked.sort(
            key=lambda item: (
                -item.score,
                -int("USER_CONFIRMED_PROFILE" in item.reason_codes),
                item.memory_id,
            )
        )
        return ranked[:k]

    @staticmethod
    def to_model_context(items: list[RetrievedMemory]) -> list[dict[str, object]]:
        return GovernedMemoryRetriever.to_model_context(items)


class FreshnessAwareMemoryRetriever:
    """Reorder eligible memories without mutating or filtering governance state."""

    _HALF_LIFE_DAYS = {
        MemoryAspect.FACT: 365.0,
        MemoryAspect.GOAL: 180.0,
        MemoryAspect.COPING_STRATEGY: 90.0,
    }
    _EXEMPT = {MemoryAspect.BOUNDARY, MemoryAspect.PREFERENCE}

    def __init__(
        self,
        base: MemoryRetrieverPort,
        *,
        minimum_factor: float = 0.90,
        candidate_multiplier: int = 3,
    ) -> None:
        if not 0 < minimum_factor <= 1:
            raise ValueError("minimum_factor must be in (0, 1]")
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be positive")
        self._base = base
        self._minimum_factor = minimum_factor
        self._candidate_multiplier = candidate_multiplier

    def retrieve(
        self,
        query: str,
        *,
        user_id: str,
        purpose_scope: str = "personalization",
        k: int = 5,
        as_of_ms: int | None = None,
    ) -> list[RetrievedMemory]:
        if k <= 0:
            return []
        timestamp = as_of_ms if as_of_ms is not None else int(time.time() * 1000)
        candidates = self._base.retrieve(
            query,
            user_id=user_id,
            purpose_scope=purpose_scope,
            k=k * self._candidate_multiplier,
            as_of_ms=as_of_ms,
        )
        reranked: list[RetrievedMemory] = []
        for item in candidates:
            if item.aspect in self._EXEMPT:
                reranked.append(
                    item.model_copy(
                        update={
                            "reason_codes": sorted(
                                set(item.reason_codes + ["FRESHNESS_EXEMPT"])
                            )
                        }
                    )
                )
                continue
            anchor = item.valid_at_ms if item.valid_at_ms is not None else timestamp
            age_days = max(0.0, (timestamp - anchor) / 86_400_000)
            half_life = self._HALF_LIFE_DAYS.get(item.aspect, 365.0)
            freshness = math.exp(-math.log(2) * age_days / half_life)
            factor = self._minimum_factor + (1 - self._minimum_factor) * freshness
            reranked.append(
                item.model_copy(
                    update={
                        "score": round(item.score * factor, 6),
                        "reason_codes": sorted(
                            set(item.reason_codes + ["FRESHNESS_RERANKED"])
                        ),
                    }
                )
            )
        reranked.sort(key=lambda item: (-item.score, item.memory_id))
        return reranked[:k]

    @staticmethod
    def to_model_context(items: list[RetrievedMemory]) -> list[dict[str, object]]:
        return GovernedMemoryRetriever.to_model_context(items)
