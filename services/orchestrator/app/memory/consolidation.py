import hashlib
import json
import re
import sqlite3
import time
from collections import defaultdict
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from app.memory.models import (
    MemoryAspect,
    MemoryChange,
    MemoryChangeState,
    MemoryConflict,
    MemoryConflictState,
    MemoryEvidenceRelation,
    MemoryItem,
    MemoryObservation,
    MemoryProfile,
    MemoryProfileEvidence,
    MemoryProfileState,
    MemoryState,
)

PROFILE_VERSION = "memory-profile-v2.2"


@dataclass(frozen=True, slots=True)
class NormalizedClaim:
    signature: str
    statement: str
    conflict_family: str | None = None
    temporal_transition: bool = False
    historical_only: bool = False


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    observation: MemoryObservation
    item: MemoryItem
    normalized: NormalizedClaim


class MemoryClaimNormalizerPort(Protocol):
    def normalize_many(
        self,
        items: Sequence[MemoryItem],
    ) -> dict[str, NormalizedClaim]: ...


class MemoryClaimNormalizer:
    """Conservative deterministic claim normalizer for the V2.0 baseline."""

    _prefix = re.compile(
        r"^(用户偏好|用户目标|用户事实|对用户有效的支持方式|用户不希望)[:：]\s*"
    )
    _space = re.compile(r"\s+")
    _punctuation = re.compile(r"[，。！？、,.!?;；:：\s]+")
    _transition = re.compile(
        r"(?:以前|过去|曾经|之前).{0,24}(?:现在|目前|以后|最近|从现在)|"
        r"(?:现在|目前|以后|从现在).{0,16}(?:改成|改为|换成|不再|更喜欢|希望|需要)"
    )
    _historical = re.compile(r"以前|之前|过去|曾经")
    _current_anchor = re.compile(r"现在|目前|如今|最近|以后|今后|从现在")
    _change_anchor = re.compile(r"改成|改为|换成|调整为|更喜欢|希望|需要")

    def normalize_many(
        self,
        items: Sequence[MemoryItem],
    ) -> dict[str, NormalizedClaim]:
        return {item.memory_id: self.normalize(item) for item in items}

    def normalize(self, item: MemoryItem) -> NormalizedClaim:
        text = self._prefix.sub("", item.candidate.text).strip()
        subject_key = item.candidate.subject_key
        transition = bool(self._transition.search(text))
        claim_text = self._current_clause(text) if transition else text
        historical_only = bool(self._historical.search(text)) and not transition
        labelled = self._known_claim(subject_key, claim_text)
        if labelled is not None:
            return NormalizedClaim(
                signature=labelled.signature,
                statement=labelled.statement,
                conflict_family=subject_key,
                temporal_transition=transition,
                historical_only=historical_only,
            )
        canonical = self._space.sub(" ", claim_text).strip()
        compact = self._punctuation.sub("", canonical).lower()
        signature = hashlib.sha256(compact.encode("utf-8")).hexdigest()[:16]
        return NormalizedClaim(
            signature=f"exact.{signature}",
            statement=claim_text,
            temporal_transition=transition,
            historical_only=historical_only,
        )

    def _current_clause(self, text: str) -> str:
        anchor = self._current_anchor.search(text)
        current = text[anchor.start() :] if anchor is not None else text
        change = self._change_anchor.search(current)
        if change is not None:
            return current[change.end() :].strip(" ：:，,") or current
        return current

    @staticmethod
    def _known_claim(subject_key: str, text: str) -> NormalizedClaim | None:
        negative = bool(re.search(r"不喜欢|不希望|不要|不想|避免|没用|无效|不适", text))
        if subject_key == "communication.response_style":
            if re.search(r"详细|展开|多一点|具体|解释清楚", text):
                return NormalizedClaim(
                    "response_style.detailed",
                    "用户偏好：回答时提供更详细的解释",
                )
            if re.search(r"简短|简洁|精简|先给结论", text):
                return NormalizedClaim(
                    "response_style.concise",
                    "用户偏好：回答时先给简短结论",
                )
        if subject_key == "communication.tone":
            if re.search(r"温柔|柔和|温和", text):
                return NormalizedClaim(
                    "tone.gentle",
                    "用户偏好：使用温和的语气交流",
                )
            if re.search(r"直接|坦率|直说", text):
                return NormalizedClaim(
                    "tone.direct",
                    "用户偏好：使用直接坦率的语气交流",
                )
        coping_labels = {
            "coping.breathing": "呼吸练习",
            "coping.counting": "数数练习",
            "coping.meditation": "冥想练习",
        }
        if subject_key in coping_labels:
            label = coping_labels[subject_key]
            return NormalizedClaim(
                f"{subject_key}.{'avoid' if negative else 'helpful'}",
                (
                    f"用户不希望：被建议使用{label}"
                    if negative
                    else f"对用户有效的支持方式：{label}"
                ),
            )
        if subject_key == "wellbeing.sleep":
            return NormalizedClaim(
                "wellbeing.sleep.earlier" if not negative else "wellbeing.sleep.avoid",
                "用户目标：逐步改善睡眠节律" if not negative else "用户不希望：讨论睡眠改善",
            )
        return None


class MemoryProfileRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def record_observation(
        self,
        item: MemoryItem,
        *,
        session_id: str,
        turn_id: str | None = None,
        now_ms: int | None = None,
        resolve_source_session: bool = False,
    ) -> MemoryObservation | None:
        if (
            item.state is not MemoryState.ACTIVE
            or not item.candidate.user_confirmed
            or item.candidate.integrity_flags
            or not item.candidate.subject_key
        ):
            return None
        timestamp = now_ms if now_ms is not None else int(time.time() * 1000)
        valid_at_ms = (
            item.candidate.valid_from_ms
            if item.candidate.valid_from_ms is not None
            else item.created_at_ms or timestamp
        )
        observation_id = f"observation_{uuid4().hex}"
        observed_turn = turn_id or item.candidate.source_turn_id
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            source_session = (
                connection.execute(
                    "SELECT session_id FROM turns WHERE turn_id = ?",
                    (observed_turn,),
                ).fetchone()
                if resolve_source_session
                else None
            )
            observed_session = (
                str(source_session[0]) if source_session is not None else session_id
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO memory_observations(
                    observation_id, memory_id, user_id, session_id,
                    turn_id, valid_at_ms, observed_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observation_id,
                    item.memory_id,
                    item.user_id,
                    observed_session,
                    observed_turn,
                    valid_at_ms,
                    timestamp,
                ),
            )
            row = connection.execute(
                """
                SELECT observation_id, memory_id, user_id, session_id,
                       turn_id, valid_at_ms, observed_at_ms
                FROM memory_observations
                WHERE user_id = ? AND memory_id = ? AND session_id = ?
                """,
                (item.user_id, item.memory_id, observed_session),
            ).fetchone()
        assert row is not None
        return MemoryObservation(
            observation_id=str(row[0]),
            memory_id=str(row[1]),
            user_id=str(row[2]),
            session_id=str(row[3]),
            turn_id=str(row[4]),
            valid_at_ms=int(row[5]),
            observed_at_ms=int(row[6]),
        )

    def evidence_for_user(self, user_id: str) -> list[tuple[MemoryObservation, MemoryItem]]:
        from app.memory.repository import MemoryRepository

        with closing(sqlite3.connect(self.database_path)) as connection:
            rows = connection.execute(
                """
                SELECT observation_id, memory_id, session_id, turn_id,
                       valid_at_ms, observed_at_ms
                FROM memory_observations
                WHERE user_id = ? ORDER BY valid_at_ms ASC, observed_at_ms ASC, rowid ASC
                """,
                (user_id,),
            ).fetchall()
        repository = MemoryRepository(self.database_path)
        evidence: list[tuple[MemoryObservation, MemoryItem]] = []
        for row in rows:
            try:
                item = repository.get(str(row[1]), user_id=user_id)
            except KeyError:
                continue
            if item.state not in {MemoryState.ACTIVE, MemoryState.SUPERSEDED}:
                continue
            if not item.candidate.user_confirmed or item.candidate.integrity_flags:
                continue
            evidence.append(
                (
                    MemoryObservation(
                        observation_id=str(row[0]),
                        memory_id=str(row[1]),
                        user_id=user_id,
                        session_id=str(row[2]),
                        turn_id=str(row[3]),
                        valid_at_ms=int(row[4]),
                        observed_at_ms=int(row[5]),
                    ),
                    item,
                )
            )
        return evidence

    def upsert_profile(
        self,
        *,
        user_id: str,
        subject_key: str,
        aspect: MemoryAspect,
        statement: str,
        signature_hash: str,
        evidence_digest: str,
        evidence: list[tuple[EvidenceRecord, MemoryEvidenceRelation]],
        state: MemoryProfileState,
        now_ms: int,
    ) -> MemoryProfile | None:
        if self.is_rejected(
            user_id=user_id,
            subject_key=subject_key,
            signature_hash=signature_hash,
        ):
            return None
        supporting = [row for row in evidence if row[1] is MemoryEvidenceRelation.SUPPORTS]
        conflicting = [
            row for row in evidence if row[1] is MemoryEvidenceRelation.CONTRADICTS
        ]
        sessions = {record.observation.session_id for record, _ in supporting}
        confidence = min(0.95, 0.55 + 0.12 * len(sessions) + 0.04 * len(supporting))
        sensitive = any(record.item.candidate.contains_sensitive_content for record, _ in evidence)
        valid_values = [record.observation.valid_at_ms for record, _ in supporting]
        expires = [
            record.item.candidate.expires_at_ms
            for record, _ in supporting
            if record.item.candidate.expires_at_ms is not None
        ]
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            existing = connection.execute(
                """
                SELECT profile_id, state, created_at_ms, user_edited, statement,
                       valid_from_ms, valid_to_ms
                FROM memory_profiles
                WHERE user_id = ? AND subject_key = ? AND signature_hash = ?
                """,
                (user_id, subject_key, signature_hash),
            ).fetchone()
            if existing is None:
                profile_id = f"profile_{uuid4().hex}"
                created_at_ms = now_ms
                persisted_state = state
                persisted_statement = statement
                user_edited = False
                connection.execute(
                    """
                    INSERT INTO memory_profiles(
                        profile_id, user_id, subject_key, aspect, statement,
                        state, confidence, evidence_count,
                        supporting_evidence_count, conflicting_evidence_count,
                        distinct_session_count, contains_sensitive_content,
                        purpose_scope, valid_from_ms, valid_to_ms, expires_at_ms,
                        created_at_ms, updated_at_ms, profile_version,
                        signature_hash, evidence_digest, user_edited
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        profile_id,
                        user_id,
                        subject_key,
                        aspect.value,
                        persisted_statement,
                        persisted_state.value,
                        confidence,
                        len(evidence),
                        len(supporting),
                        len(conflicting),
                        len(sessions),
                        int(sensitive),
                        "personalization",
                        max(valid_values) if valid_values else None,
                        None,
                        min(expires) if expires else None,
                        created_at_ms,
                        now_ms,
                        PROFILE_VERSION,
                        signature_hash,
                        evidence_digest,
                        0,
                    ),
                )
            else:
                profile_id = str(existing[0])
                existing_state = MemoryProfileState(str(existing[1]))
                created_at_ms = int(existing[2])
                user_edited = bool(existing[3])
                persisted_statement = str(existing[4]) if user_edited else statement
                existing_valid_from_ms = (
                    int(existing[5]) if existing[5] is not None else None
                )
                existing_valid_to_ms = (
                    int(existing[6]) if existing[6] is not None else None
                )
                persisted_state = (
                    existing_state
                    if existing_state is MemoryProfileState.ACTIVE
                    and state is not MemoryProfileState.STALE
                    else state
                )
                next_valid_from_ms = (
                    existing_valid_from_ms
                    if existing_state
                    in {MemoryProfileState.ACTIVE, MemoryProfileState.HISTORICAL}
                    else max(valid_values) if valid_values else None
                )
                next_valid_to_ms = (
                    existing_valid_to_ms
                    if existing_state is MemoryProfileState.HISTORICAL
                    else None
                )
                connection.execute(
                    """
                    UPDATE memory_profiles SET aspect=?, statement=?, state=?,
                        confidence=?, evidence_count=?, supporting_evidence_count=?,
                        conflicting_evidence_count=?, distinct_session_count=?,
                        contains_sensitive_content=?, valid_from_ms=?, valid_to_ms=?,
                        expires_at_ms=?,
                        updated_at_ms=?, profile_version=?, evidence_digest=?
                    WHERE profile_id=?
                    """,
                    (
                        aspect.value,
                        persisted_statement,
                        persisted_state.value,
                        confidence,
                        len(evidence),
                        len(supporting),
                        len(conflicting),
                        len(sessions),
                        int(sensitive),
                        next_valid_from_ms,
                        next_valid_to_ms,
                        min(expires) if expires else None,
                        now_ms,
                        PROFILE_VERSION,
                        evidence_digest,
                        profile_id,
                    ),
                )
            connection.execute(
                "DELETE FROM memory_profile_evidence WHERE profile_id = ?",
                (profile_id,),
            )
            connection.executemany(
                """
                INSERT INTO memory_profile_evidence(
                    profile_id, observation_id, memory_id, relation
                ) VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        profile_id,
                        record.observation.observation_id,
                        record.item.memory_id,
                        relation.value,
                    )
                    for record, relation in evidence
                ],
            )
        return self.get_profile(profile_id, user_id=user_id)

    def mark_subject_profiles_stale(
        self,
        *,
        user_id: str,
        subject_key: str,
        except_profile_id: str | None = None,
        now_ms: int,
    ) -> None:
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            query = """
                UPDATE memory_profiles SET state = ?, updated_at_ms = ?
                WHERE user_id = ? AND subject_key = ?
                    AND state IN (?, ?, ?)
            """
            values: tuple[object, ...] = (
                MemoryProfileState.STALE.value,
                now_ms,
                user_id,
                subject_key,
                MemoryProfileState.AWAITING_CONFIRMATION.value,
                MemoryProfileState.ACTIVE.value,
                MemoryProfileState.STALE.value,
            )
            if except_profile_id is not None:
                query += " AND profile_id <> ?"
                values += (except_profile_id,)
            connection.execute(query, values)

    def mark_profile_stale(
        self,
        profile_id: str,
        *,
        user_id: str,
        now_ms: int,
    ) -> None:
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            cursor = connection.execute(
                """
                UPDATE memory_profiles SET state = ?, updated_at_ms = ?
                WHERE profile_id = ? AND user_id = ?
                """,
                (
                    MemoryProfileState.STALE.value,
                    now_ms,
                    profile_id,
                    user_id,
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(profile_id)

    def create_or_get_conflict(
        self,
        *,
        user_id: str,
        subject_key: str,
        evidence_digest: str,
        profile_ids: list[str],
        now_ms: int,
    ) -> MemoryConflict:
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            row = connection.execute(
                """
                SELECT conflict_id FROM memory_conflict_groups
                WHERE user_id = ? AND subject_key = ? AND evidence_digest = ?
                """,
                (user_id, subject_key, evidence_digest),
            ).fetchone()
            if row is None:
                conflict_id = f"conflict_{uuid4().hex}"
                connection.execute(
                    """
                    INSERT INTO memory_conflict_groups(
                        conflict_id, user_id, subject_key, state,
                        evidence_digest, created_at_ms, updated_at_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        conflict_id,
                        user_id,
                        subject_key,
                        MemoryConflictState.OPEN.value,
                        evidence_digest,
                        now_ms,
                        now_ms,
                    ),
                )
            else:
                conflict_id = str(row[0])
            connection.executemany(
                """
                INSERT OR IGNORE INTO memory_conflict_options(conflict_id, profile_id)
                VALUES (?, ?)
                """,
                [(conflict_id, profile_id) for profile_id in profile_ids],
            )
        return self.get_conflict(conflict_id, user_id=user_id)

    def list_profiles(
        self,
        user_id: str,
        *,
        states: set[MemoryProfileState] | None = None,
    ) -> list[MemoryProfile]:
        with closing(sqlite3.connect(self.database_path)) as connection:
            rows = connection.execute(
                """
                SELECT profile_id FROM memory_profiles WHERE user_id = ?
                ORDER BY updated_at_ms DESC, rowid DESC
                """,
                (user_id,),
            ).fetchall()
        profiles = [self.get_profile(str(row[0]), user_id=user_id) for row in rows]
        return [profile for profile in profiles if states is None or profile.state in states]

    def open_change_proposal_ids(self, user_id: str) -> set[str]:
        with closing(sqlite3.connect(self.database_path)) as connection:
            rows = connection.execute(
                """
                SELECT proposed_profile_id FROM memory_profile_changes
                WHERE user_id = ? AND state = ?
                """,
                (user_id, MemoryChangeState.OPEN.value),
            ).fetchall()
        return {str(row[0]) for row in rows}

    def list_active_profiles(
        self,
        user_id: str,
        *,
        purpose_scope: str = "personalization",
        as_of_ms: int | None = None,
    ) -> list[MemoryProfile]:
        timestamp = as_of_ms if as_of_ms is not None else int(time.time() * 1000)
        return [
            profile
            for profile in self.list_profiles(
                user_id,
                states={
                    MemoryProfileState.ACTIVE,
                    MemoryProfileState.HISTORICAL,
                },
            )
            if profile.purpose_scope == purpose_scope
            and (profile.valid_from_ms is None or profile.valid_from_ms <= timestamp)
            and (profile.valid_to_ms is None or profile.valid_to_ms > timestamp)
            and (profile.expires_at_ms is None or profile.expires_at_ms > timestamp)
        ]

    def get_profile(self, profile_id: str, *, user_id: str) -> MemoryProfile:
        with closing(sqlite3.connect(self.database_path)) as connection:
            row = connection.execute(
                """
                SELECT profile_id, user_id, subject_key, aspect, statement,
                       state, confidence, evidence_count,
                       supporting_evidence_count, conflicting_evidence_count,
                       distinct_session_count, contains_sensitive_content,
                       purpose_scope, valid_from_ms, valid_to_ms, expires_at_ms,
                       created_at_ms, updated_at_ms, profile_version,
                       signature_hash, evidence_digest, user_edited
                FROM memory_profiles WHERE profile_id = ? AND user_id = ?
                """,
                (profile_id, user_id),
            ).fetchone()
            if row is None:
                raise KeyError(profile_id)
            evidence_rows = connection.execute(
                """
                SELECT link.observation_id, link.memory_id,
                       observation.session_id, observation.turn_id,
                       item.content, link.relation, observation.valid_at_ms,
                       observation.observed_at_ms
                FROM memory_profile_evidence AS link
                JOIN memory_observations AS observation
                  ON observation.observation_id = link.observation_id
                JOIN memory_items AS item ON item.memory_id = link.memory_id
                WHERE link.profile_id = ?
                ORDER BY observation.valid_at_ms ASC, observation.observed_at_ms ASC
                """,
                (profile_id,),
            ).fetchall()
        return MemoryProfile(
            profile_id=str(row[0]),
            user_id=str(row[1]),
            subject_key=str(row[2]),
            aspect=MemoryAspect(str(row[3])),
            statement=str(row[4]),
            state=MemoryProfileState(str(row[5])),
            confidence=float(row[6]),
            evidence_count=int(row[7]),
            supporting_evidence_count=int(row[8]),
            conflicting_evidence_count=int(row[9]),
            distinct_session_count=int(row[10]),
            contains_sensitive_content=bool(row[11]),
            purpose_scope=str(row[12]),
            valid_from_ms=int(row[13]) if row[13] is not None else None,
            valid_to_ms=int(row[14]) if row[14] is not None else None,
            expires_at_ms=int(row[15]) if row[15] is not None else None,
            created_at_ms=int(row[16]),
            updated_at_ms=int(row[17]),
            profile_version=str(row[18]),
            signature_hash=str(row[19]),
            evidence_digest=str(row[20]),
            user_edited=bool(row[21]),
            evidence=[
                MemoryProfileEvidence(
                    observation_id=str(evidence_row[0]),
                    memory_id=str(evidence_row[1]),
                    session_id=str(evidence_row[2]),
                    turn_id=str(evidence_row[3]),
                    text=str(evidence_row[4]),
                    relation=MemoryEvidenceRelation(str(evidence_row[5])),
                    valid_at_ms=int(evidence_row[6]),
                    observed_at_ms=int(evidence_row[7]),
                )
                for evidence_row in evidence_rows
            ],
        )

    def confirm_profile(
        self,
        profile_id: str,
        *,
        user_id: str,
        statement: str | None = None,
        now_ms: int | None = None,
    ) -> MemoryProfile:
        profile = self.get_profile(profile_id, user_id=user_id)
        self._ensure_not_open_change_proposal(profile_id, user_id=user_id)
        timestamp = now_ms if now_ms is not None else int(time.time() * 1000)
        normalized = " ".join(statement.split()).strip() if statement is not None else None
        if normalized == "":
            raise ValueError("profile statement must not be blank")
        self.mark_subject_profiles_stale(
            user_id=user_id,
            subject_key=profile.subject_key,
            except_profile_id=profile_id,
            now_ms=timestamp,
        )
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.execute(
                """
                UPDATE memory_profiles SET state = ?, statement = ?,
                    user_edited = ?, confidence = 1.0, updated_at_ms = ?
                WHERE profile_id = ? AND user_id = ?
                """,
                (
                    MemoryProfileState.ACTIVE.value,
                    normalized or profile.statement,
                    int(normalized is not None),
                    timestamp,
                    profile_id,
                    user_id,
                ),
            )
            connection.execute(
                """
                UPDATE memory_conflict_groups SET state = ?, selected_profile_id = ?,
                    updated_at_ms = ?
                WHERE user_id = ? AND subject_key = ? AND state = ?
                """,
                (
                    MemoryConflictState.RESOLVED.value,
                    profile_id,
                    timestamp,
                    user_id,
                    profile.subject_key,
                    MemoryConflictState.OPEN.value,
                ),
            )
        return self.get_profile(profile_id, user_id=user_id)

    def reject_profile(
        self,
        profile_id: str,
        *,
        user_id: str,
        now_ms: int | None = None,
    ) -> None:
        profile = self.get_profile(profile_id, user_id=user_id)
        self._ensure_not_open_change_proposal(profile_id, user_id=user_id)
        timestamp = now_ms if now_ms is not None else int(time.time() * 1000)
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.execute("PRAGMA secure_delete = ON")
            connection.execute(
                """
                INSERT OR REPLACE INTO memory_profile_rejections(
                    user_id, subject_key, signature_hash, rejected_at_ms
                ) VALUES (?, ?, ?, ?)
                """,
                (user_id, profile.subject_key, profile.signature_hash, timestamp),
            )
            self._delete_profiles(connection, [profile_id])

    def _ensure_not_open_change_proposal(
        self,
        profile_id: str,
        *,
        user_id: str,
    ) -> None:
        with closing(sqlite3.connect(self.database_path)) as connection:
            row = connection.execute(
                """
                SELECT 1 FROM memory_profile_changes
                WHERE user_id = ? AND proposed_profile_id = ? AND state = ?
                """,
                (user_id, profile_id, MemoryChangeState.OPEN.value),
            ).fetchone()
        if row is not None:
            raise ValueError("memory change proposals require a change decision")

    def list_conflicts(
        self,
        user_id: str,
        *,
        state: MemoryConflictState | None = None,
    ) -> list[MemoryConflict]:
        with closing(sqlite3.connect(self.database_path)) as connection:
            query = """
                SELECT conflict_id FROM memory_conflict_groups WHERE user_id = ?
            """
            values: tuple[object, ...] = (user_id,)
            if state is not None:
                query += " AND state = ?"
                values += (state.value,)
            query += " ORDER BY updated_at_ms DESC, rowid DESC"
            rows = connection.execute(query, values).fetchall()
        return [self.get_conflict(str(row[0]), user_id=user_id) for row in rows]

    def get_conflict(self, conflict_id: str, *, user_id: str) -> MemoryConflict:
        with closing(sqlite3.connect(self.database_path)) as connection:
            row = connection.execute(
                """
                SELECT conflict_id, user_id, subject_key, state, evidence_digest,
                       selected_profile_id, created_at_ms, updated_at_ms
                FROM memory_conflict_groups
                WHERE conflict_id = ? AND user_id = ?
                """,
                (conflict_id, user_id),
            ).fetchone()
            if row is None:
                raise KeyError(conflict_id)
            option_rows = connection.execute(
                """
                SELECT profile_id FROM memory_conflict_options
                WHERE conflict_id = ? ORDER BY rowid ASC
                """,
                (conflict_id,),
            ).fetchall()
        return MemoryConflict(
            conflict_id=str(row[0]),
            user_id=str(row[1]),
            subject_key=str(row[2]),
            state=MemoryConflictState(str(row[3])),
            evidence_digest=str(row[4]),
            selected_profile_id=str(row[5]) if row[5] is not None else None,
            created_at_ms=int(row[6]),
            updated_at_ms=int(row[7]),
            options=[
                self.get_profile(str(option[0]), user_id=user_id)
                for option in option_rows
            ],
        )

    def dismiss_conflict(
        self,
        conflict_id: str,
        *,
        user_id: str,
        now_ms: int | None = None,
    ) -> MemoryConflict:
        conflict = self.get_conflict(conflict_id, user_id=user_id)
        timestamp = now_ms if now_ms is not None else int(time.time() * 1000)
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.execute(
                """
                UPDATE memory_conflict_groups SET state = ?, updated_at_ms = ?
                WHERE conflict_id = ? AND user_id = ?
                """,
                (
                    MemoryConflictState.DISMISSED.value,
                    timestamp,
                    conflict_id,
                    user_id,
                ),
            )
            connection.execute(
                """
                UPDATE memory_profiles SET state = ?, updated_at_ms = ?
                WHERE profile_id IN (
                    SELECT profile_id FROM memory_conflict_options
                    WHERE conflict_id = ?
                )
                """,
                (MemoryProfileState.STALE.value, timestamp, conflict_id),
            )
        return self.get_conflict(conflict.conflict_id, user_id=user_id)

    def create_or_get_change(
        self,
        *,
        user_id: str,
        subject_key: str,
        previous_profile_id: str,
        proposed_profile_id: str,
        effective_at_ms: int,
        observed_at_ms: int,
        evidence_digest: str,
        now_ms: int,
    ) -> MemoryChange:
        if previous_profile_id == proposed_profile_id:
            raise ValueError("profile change requires two different profiles")
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            row = connection.execute(
                """
                SELECT change_id FROM memory_profile_changes
                WHERE user_id = ? AND subject_key = ? AND evidence_digest = ?
                """,
                (user_id, subject_key, evidence_digest),
            ).fetchone()
            if row is None:
                superseded = connection.execute(
                    """
                    SELECT proposed_profile_id FROM memory_profile_changes
                    WHERE user_id = ? AND subject_key = ? AND state = ?
                    """,
                    (user_id, subject_key, MemoryChangeState.OPEN.value),
                ).fetchall()
                connection.execute(
                    """
                    UPDATE memory_profile_changes SET state = ?, updated_at_ms = ?
                    WHERE user_id = ? AND subject_key = ? AND state = ?
                    """,
                    (
                        MemoryChangeState.REJECTED.value,
                        now_ms,
                        user_id,
                        subject_key,
                        MemoryChangeState.OPEN.value,
                    ),
                )
                for (profile_id,) in superseded:
                    connection.execute(
                        """
                        UPDATE memory_profiles SET state = ?, updated_at_ms = ?
                        WHERE profile_id = ? AND user_id = ? AND state = ?
                        """,
                        (
                            MemoryProfileState.STALE.value,
                            now_ms,
                            profile_id,
                            user_id,
                            MemoryProfileState.AWAITING_CONFIRMATION.value,
                        ),
                    )
                change_id = f"change_{uuid4().hex}"
                connection.execute(
                    """
                    INSERT INTO memory_profile_changes(
                        change_id, user_id, subject_key, state,
                        previous_profile_id, proposed_profile_id,
                        effective_at_ms, observed_at_ms, evidence_digest,
                        created_at_ms, updated_at_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        change_id,
                        user_id,
                        subject_key,
                        MemoryChangeState.OPEN.value,
                        previous_profile_id,
                        proposed_profile_id,
                        effective_at_ms,
                        observed_at_ms,
                        evidence_digest,
                        now_ms,
                        now_ms,
                    ),
                )
            else:
                change_id = str(row[0])
        return self.get_change(change_id, user_id=user_id)

    def reject_open_changes_for_subject(
        self,
        *,
        user_id: str,
        subject_key: str,
        now_ms: int,
    ) -> None:
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            rows = connection.execute(
                """
                SELECT proposed_profile_id FROM memory_profile_changes
                WHERE user_id = ? AND subject_key = ? AND state = ?
                """,
                (user_id, subject_key, MemoryChangeState.OPEN.value),
            ).fetchall()
            connection.execute(
                """
                UPDATE memory_profile_changes SET state = ?, updated_at_ms = ?
                WHERE user_id = ? AND subject_key = ? AND state = ?
                """,
                (
                    MemoryChangeState.REJECTED.value,
                    now_ms,
                    user_id,
                    subject_key,
                    MemoryChangeState.OPEN.value,
                ),
            )
            for (profile_id,) in rows:
                connection.execute(
                    """
                    UPDATE memory_profiles SET state = ?, updated_at_ms = ?
                    WHERE profile_id = ? AND user_id = ? AND state = ?
                    """,
                    (
                        MemoryProfileState.STALE.value,
                        now_ms,
                        profile_id,
                        user_id,
                        MemoryProfileState.AWAITING_CONFIRMATION.value,
                    ),
                )

    def list_changes(
        self,
        user_id: str,
        *,
        state: MemoryChangeState | None = None,
    ) -> list[MemoryChange]:
        with closing(sqlite3.connect(self.database_path)) as connection:
            query = "SELECT change_id FROM memory_profile_changes WHERE user_id = ?"
            values: tuple[object, ...] = (user_id,)
            if state is not None:
                query += " AND state = ?"
                values += (state.value,)
            query += " ORDER BY updated_at_ms DESC, rowid DESC"
            rows = connection.execute(query, values).fetchall()
        return [self.get_change(str(row[0]), user_id=user_id) for row in rows]

    def get_change(self, change_id: str, *, user_id: str) -> MemoryChange:
        with closing(sqlite3.connect(self.database_path)) as connection:
            row = connection.execute(
                """
                SELECT change_id, user_id, subject_key, state,
                       previous_profile_id, proposed_profile_id,
                       effective_at_ms, observed_at_ms, evidence_digest,
                       created_at_ms, updated_at_ms
                FROM memory_profile_changes
                WHERE change_id = ? AND user_id = ?
                """,
                (change_id, user_id),
            ).fetchone()
        if row is None:
            raise KeyError(change_id)
        return MemoryChange(
            change_id=str(row[0]),
            user_id=str(row[1]),
            subject_key=str(row[2]),
            state=MemoryChangeState(str(row[3])),
            previous_profile_id=str(row[4]),
            proposed_profile_id=str(row[5]),
            effective_at_ms=int(row[6]),
            observed_at_ms=int(row[7]),
            evidence_digest=str(row[8]),
            created_at_ms=int(row[9]),
            updated_at_ms=int(row[10]),
            previous_profile=self.get_profile(str(row[4]), user_id=user_id),
            proposed_profile=self.get_profile(str(row[5]), user_id=user_id),
        )

    def apply_change(
        self,
        change_id: str,
        *,
        user_id: str,
        now_ms: int | None = None,
    ) -> MemoryChange:
        change = self.get_change(change_id, user_id=user_id)
        if change.state is not MemoryChangeState.OPEN:
            raise ValueError("memory change is already closed")
        if change.previous_profile.state is not MemoryProfileState.ACTIVE:
            raise ValueError("previous memory profile is no longer active")
        if change.proposed_profile.state is not MemoryProfileState.AWAITING_CONFIRMATION:
            raise ValueError("proposed memory profile is no longer eligible")
        if (
            change.previous_profile.valid_from_ms is not None
            and change.effective_at_ms <= change.previous_profile.valid_from_ms
        ):
            raise ValueError("memory change must start after the current profile")
        timestamp = now_ms if now_ms is not None else int(time.time() * 1000)
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.execute(
                """
                UPDATE memory_profiles
                SET state = ?, valid_to_ms = ?, updated_at_ms = ?
                WHERE profile_id = ? AND user_id = ?
                """,
                (
                    MemoryProfileState.HISTORICAL.value,
                    change.effective_at_ms,
                    timestamp,
                    change.previous_profile_id,
                    user_id,
                ),
            )
            connection.execute(
                """
                UPDATE memory_profiles
                SET state = ?, valid_from_ms = ?, valid_to_ms = NULL,
                    confidence = 1.0, updated_at_ms = ?
                WHERE profile_id = ? AND user_id = ?
                """,
                (
                    MemoryProfileState.ACTIVE.value,
                    change.effective_at_ms,
                    timestamp,
                    change.proposed_profile_id,
                    user_id,
                ),
            )
            connection.execute(
                """
                UPDATE memory_profile_changes SET state = ?, updated_at_ms = ?
                WHERE change_id = ? AND user_id = ?
                """,
                (
                    MemoryChangeState.APPLIED.value,
                    timestamp,
                    change_id,
                    user_id,
                ),
            )
        return self.get_change(change_id, user_id=user_id)

    def reject_change(
        self,
        change_id: str,
        *,
        user_id: str,
        now_ms: int | None = None,
    ) -> MemoryChange:
        change = self.get_change(change_id, user_id=user_id)
        if change.state is not MemoryChangeState.OPEN:
            raise ValueError("memory change is already closed")
        timestamp = now_ms if now_ms is not None else int(time.time() * 1000)
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.execute(
                """
                UPDATE memory_profile_changes SET state = ?, updated_at_ms = ?
                WHERE change_id = ? AND user_id = ?
                """,
                (
                    MemoryChangeState.REJECTED.value,
                    timestamp,
                    change_id,
                    user_id,
                ),
            )
            connection.execute(
                """
                UPDATE memory_profiles SET state = ?, updated_at_ms = ?
                WHERE profile_id = ? AND user_id = ?
                """,
                (
                    MemoryProfileState.STALE.value,
                    timestamp,
                    change.proposed_profile_id,
                    user_id,
                ),
            )
        return self.get_change(change_id, user_id=user_id)

    def is_rejected(
        self,
        *,
        user_id: str,
        subject_key: str,
        signature_hash: str,
    ) -> bool:
        with closing(sqlite3.connect(self.database_path)) as connection:
            row = connection.execute(
                """
                SELECT 1 FROM memory_profile_rejections
                WHERE user_id = ? AND subject_key = ? AND signature_hash = ?
                """,
                (user_id, subject_key, signature_hash),
            ).fetchone()
        return row is not None

    @staticmethod
    def _delete_profiles(
        connection: sqlite3.Connection,
        profile_ids: list[str],
    ) -> None:
        for profile_id in profile_ids:
            connection.execute(
                """
                DELETE FROM memory_profile_changes
                WHERE previous_profile_id = ? OR proposed_profile_id = ?
                """,
                (profile_id, profile_id),
            )
            conflict_ids = connection.execute(
                """
                SELECT conflict_id FROM memory_conflict_options
                WHERE profile_id = ?
                """,
                (profile_id,),
            ).fetchall()
            connection.execute(
                "DELETE FROM memory_conflict_options WHERE profile_id = ?",
                (profile_id,),
            )
            for (conflict_id,) in conflict_ids:
                connection.execute(
                    "DELETE FROM memory_conflict_options WHERE conflict_id = ?",
                    (conflict_id,),
                )
                connection.execute(
                    "DELETE FROM memory_conflict_groups WHERE conflict_id = ?",
                    (conflict_id,),
                )
            connection.execute(
                "DELETE FROM memory_profile_evidence WHERE profile_id = ?",
                (profile_id,),
            )
            connection.execute(
                "DELETE FROM memory_retrievals WHERE memory_id = ?",
                (profile_id,),
            )
            connection.execute(
                "DELETE FROM memory_profiles WHERE profile_id = ?",
                (profile_id,),
            )


class MemoryConsolidator:
    def __init__(
        self,
        repository: MemoryProfileRepository,
        *,
        minimum_sessions: int = 2,
        normalizer: MemoryClaimNormalizerPort | None = None,
    ) -> None:
        if minimum_sessions < 2:
            raise ValueError("minimum_sessions must be at least two")
        self._repository = repository
        self._minimum_sessions = minimum_sessions
        self._normalizer = normalizer or MemoryClaimNormalizer()

    def rebuild_user(
        self,
        user_id: str,
        *,
        now_ms: int | None = None,
    ) -> list[MemoryProfile]:
        timestamp = now_ms if now_ms is not None else int(time.time() * 1000)
        evidence = self._repository.evidence_for_user(user_id)
        unique_items = list({item.memory_id: item for _, item in evidence}.values())
        normalized_by_memory_id = self._normalizer.normalize_many(unique_items)
        missing = {
            item.memory_id for item in unique_items
        } - normalized_by_memory_id.keys()
        if missing:
            raise ValueError("claim normalizer omitted confirmed evidence")
        by_subject: dict[str, list[EvidenceRecord]] = defaultdict(list)
        for observation, item in evidence:
            by_subject[item.candidate.subject_key].append(
                EvidenceRecord(
                    observation,
                    item,
                    normalized_by_memory_id[item.memory_id],
                )
            )

        # Purely historical claims remain in the auditable evidence layer. They
        # do not alter the current profile or create a present-day conflict.
        by_subject = {
            subject_key: [
                record
                for record in records
                if not record.normalized.historical_only
            ]
            for subject_key, records in by_subject.items()
        }

        changed: list[MemoryProfile] = []
        applied_changes = self._repository.list_changes(
            user_id,
            state=MemoryChangeState.APPLIED,
        )
        for subject_key, records in by_subject.items():
            subject_applied_changes = [
                change
                for change in applied_changes
                if change.subject_key == subject_key
            ]
            if subject_applied_changes:
                latest_effective_at = max(
                    change.effective_at_ms for change in subject_applied_changes
                )
                records = [
                    record
                    for record in records
                    if record.observation.valid_at_ms >= latest_effective_at
                ]
            clusters: dict[str, list[EvidenceRecord]] = defaultdict(list)
            for record in records:
                clusters[record.normalized.signature].append(record)
            transition_records = [
                record for record in records if record.normalized.temporal_transition
            ]
            active_profiles = self._repository.list_profiles(
                user_id,
                states={MemoryProfileState.ACTIVE},
            )
            active_for_subject = [
                profile
                for profile in active_profiles
                if profile.subject_key == subject_key
            ]
            if transition_records and len(active_for_subject) == 1:
                latest_transition = max(
                    transition_records,
                    key=lambda record: (
                        record.observation.valid_at_ms,
                        record.observation.observed_at_ms,
                    ),
                )
                previous = active_for_subject[0]
                next_signature = latest_transition.normalized.signature
                if next_signature == _semantic_signature(previous.signature_hash):
                    has_open_change = any(
                        change.subject_key == subject_key
                        for change in self._repository.list_changes(
                            user_id,
                            state=MemoryChangeState.OPEN,
                        )
                    )
                    if has_open_change:
                        self._repository.reject_open_changes_for_subject(
                            user_id=user_id,
                            subject_key=subject_key,
                            now_ms=timestamp,
                        )
                        continue
                if next_signature != _semantic_signature(previous.signature_hash):
                    versioned_signature = (
                        f"{next_signature}.effective."
                        f"{latest_transition.observation.valid_at_ms}"
                    )
                    proposal = self._repository.upsert_profile(
                        user_id=user_id,
                        subject_key=subject_key,
                        aspect=latest_transition.item.candidate.aspect,
                        statement=latest_transition.normalized.statement,
                        signature_hash=versioned_signature,
                        evidence_digest=_digest(
                            [latest_transition.observation.observation_id]
                        ),
                        evidence=[
                            (
                                latest_transition,
                                MemoryEvidenceRelation.SUPPORTS,
                            )
                        ],
                        state=MemoryProfileState.AWAITING_CONFIRMATION,
                        now_ms=timestamp,
                    )
                    if proposal is not None:
                        change_digest = _digest(
                            [
                                previous.profile_id,
                                proposal.profile_id,
                                latest_transition.observation.observation_id,
                            ]
                        )
                        change = self._repository.create_or_get_change(
                            user_id=user_id,
                            subject_key=subject_key,
                            previous_profile_id=previous.profile_id,
                            proposed_profile_id=proposal.profile_id,
                            effective_at_ms=latest_transition.observation.valid_at_ms,
                            observed_at_ms=latest_transition.observation.observed_at_ms,
                            evidence_digest=change_digest,
                            now_ms=timestamp,
                        )
                        if change.state is MemoryChangeState.OPEN:
                            changed.append(proposal)
                        elif change.state is MemoryChangeState.REJECTED:
                            self._repository.mark_profile_stale(
                                proposal.profile_id,
                                user_id=user_id,
                                now_ms=timestamp,
                            )
                        continue
            qualifying = {
                signature: rows
                for signature, rows in clusters.items()
                if len({row.observation.session_id for row in rows})
                >= self._minimum_sessions
            }
            conflict_families: dict[str, list[EvidenceRecord]] = defaultdict(list)
            for record in records:
                family = record.normalized.conflict_family
                if family is not None:
                    conflict_families[family].append(record)
            known_conflict = any(
                len({record.normalized.signature for record in family_records}) > 1
                and len(
                    {
                        record.observation.session_id
                        for record in family_records
                    }
                )
                >= self._minimum_sessions
                for family_records in conflict_families.values()
            )
            is_conflict = len(qualifying) > 1 or known_conflict
            if not qualifying and not is_conflict:
                continue
            profile_clusters = clusters if is_conflict else qualifying
            profiles: list[MemoryProfile] = []
            for signature, supporting_records in profile_clusters.items():
                storage_signature = signature
                if not is_conflict:
                    matching_active = [
                        profile
                        for profile in active_for_subject
                        if _semantic_signature(profile.signature_hash) == signature
                    ]
                    if len(matching_active) == 1:
                        # An applied temporal change has a versioned storage key.
                        # Later corroborating evidence must enrich that version,
                        # not create a duplicate profile that disables it.
                        storage_signature = matching_active[0].signature_hash
                related_records = records if is_conflict else supporting_records
                all_evidence = [
                    (
                        record,
                        MemoryEvidenceRelation.SUPPORTS
                        if record.normalized.signature == signature
                        else MemoryEvidenceRelation.CONTRADICTS,
                    )
                    for record in related_records
                ]
                evidence_digest = _digest(
                    sorted(
                        f"{record.observation.observation_id}:{relation.value}"
                        for record, relation in all_evidence
                    )
                )
                profile = self._repository.upsert_profile(
                    user_id=user_id,
                    subject_key=subject_key,
                    aspect=supporting_records[-1].item.candidate.aspect,
                    statement=supporting_records[-1].normalized.statement,
                    signature_hash=storage_signature,
                    evidence_digest=evidence_digest,
                    evidence=all_evidence,
                    state=(
                        MemoryProfileState.STALE
                        if is_conflict
                        else MemoryProfileState.AWAITING_CONFIRMATION
                    ),
                    now_ms=timestamp,
                )
                if profile is not None:
                    profiles.append(profile)
                    changed.append(profile)
            if is_conflict and len(profiles) > 1:
                signature_digest = _digest(sorted(profile.signature_hash for profile in profiles))
                conflict = self._repository.create_or_get_conflict(
                    user_id=user_id,
                    subject_key=subject_key,
                    evidence_digest=signature_digest,
                    profile_ids=[profile.profile_id for profile in profiles],
                    now_ms=timestamp,
                )
                if conflict.state is MemoryConflictState.RESOLVED:
                    selected = conflict.selected_profile_id
                    if selected is not None:
                        self._repository.confirm_profile(
                            selected,
                            user_id=user_id,
                            now_ms=timestamp,
                        )
                elif conflict.state is MemoryConflictState.OPEN:
                    self._repository.mark_subject_profiles_stale(
                        user_id=user_id,
                        subject_key=subject_key,
                        now_ms=timestamp,
                    )
            elif len(profiles) == 1:
                # A normalizer/version change may replace an older signature for
                # the same semantic claim. Fail closed: only the current proposal
                # can remain active, and a changed signature requires confirmation.
                self._repository.mark_subject_profiles_stale(
                    user_id=user_id,
                    subject_key=subject_key,
                    except_profile_id=profiles[0].profile_id,
                    now_ms=timestamp,
                )
        return changed


def _digest(values: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _semantic_signature(signature: str) -> str:
    return signature.split(".effective.", maxsplit=1)[0]
