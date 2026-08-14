import sqlite3
import tempfile
import time
from collections.abc import Collection
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from app.memory.consolidation import MemoryConsolidator, MemoryProfileRepository
from app.memory.episodes import MemoryEpisodeRepository
from app.memory.models import MemoryAspect, MemoryCandidate, MemoryKind
from app.memory.repository import MemoryRepository


@dataclass(frozen=True, slots=True)
class PrivacyScanReport:
    raw_audio_blob_count: int
    raw_video_blob_count: int
    camera_frame_blob_count: int
    binary_column_count: int


@dataclass(frozen=True, slots=True)
class DeletionProbeReport:
    checked_file_count: int
    residual_file_count: int
    residual_file_kinds: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.checked_file_count > 0 and self.residual_file_count == 0


@dataclass(frozen=True, slots=True)
class MemoryPrivacyAuditReport:
    media: PrivacyScanReport
    deletion_probe: DeletionProbeReport
    observation_owner_mismatch_count: int
    profile_evidence_owner_mismatch_count: int
    episode_member_owner_mismatch_count: int
    retrieval_owner_mismatch_count: int
    shadow_ranking_owner_mismatch_count: int
    orphan_observation_count: int
    orphan_profile_evidence_count: int
    orphan_episode_member_count: int
    orphan_retrieval_count: int
    orphan_shadow_ranking_count: int
    profile_evidence_link_mismatch_count: int
    ineligible_episode_member_count: int
    ineligible_active_profile_evidence_count: int

    @property
    def owner_mismatch_count(self) -> int:
        return sum(
            (
                self.observation_owner_mismatch_count,
                self.profile_evidence_owner_mismatch_count,
                self.episode_member_owner_mismatch_count,
                self.retrieval_owner_mismatch_count,
                self.shadow_ranking_owner_mismatch_count,
            )
        )

    @property
    def orphan_reference_count(self) -> int:
        return sum(
            (
                self.orphan_observation_count,
                self.orphan_profile_evidence_count,
                self.orphan_episode_member_count,
                self.orphan_retrieval_count,
                self.orphan_shadow_ranking_count,
            )
        )

    @property
    def passed(self) -> bool:
        return (
            self.media.raw_audio_blob_count == 0
            and self.media.raw_video_blob_count == 0
            and self.media.camera_frame_blob_count == 0
            and self.media.binary_column_count == 0
            and self.owner_mismatch_count == 0
            and self.orphan_reference_count == 0
            and self.profile_evidence_link_mismatch_count == 0
            and self.ineligible_episode_member_count == 0
            and self.ineligible_active_profile_evidence_count == 0
            and self.deletion_probe.passed
        )


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def scan_sqlite_media(database_path: Path) -> PrivacyScanReport:
    raw_audio = 0
    raw_video = 0
    camera_frames = 0
    binary_columns = 0
    with closing(sqlite3.connect(database_path)) as connection:
        tables = [
            str(row[0])
            for row in connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                """
            ).fetchall()
        ]
        for table in tables:
            columns = connection.execute(
                f"PRAGMA table_info({_quote_identifier(table)})"
            ).fetchall()
            for column in columns:
                column_name = str(column[1])
                declared_type = str(column[2]).upper()
                binary_columns += int("BLOB" in declared_type)
                blob_count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM "
                        f"{_quote_identifier(table)} "
                        f"WHERE typeof({_quote_identifier(column_name)}) = 'blob'"
                    ).fetchone()[0]
                )
                normalized = column_name.casefold()
                if any(term in normalized for term in ("audio", "pcm", "microphone")):
                    raw_audio += blob_count
                if "video" in normalized:
                    raw_video += blob_count
                if any(
                    term in normalized
                    for term in ("camera", "frame", "image", "jpeg")
                ):
                    camera_frames += blob_count
    return PrivacyScanReport(
        raw_audio_blob_count=raw_audio,
        raw_video_blob_count=raw_video,
        camera_frame_blob_count=camera_frames,
        binary_column_count=binary_columns,
    )


def scan_sqlite_residuals(
    database_path: Path,
    markers: Collection[str | bytes],
) -> DeletionProbeReport:
    encoded = tuple(
        marker.encode("utf-8") if isinstance(marker, str) else marker
        for marker in markers
    )
    residual_kinds: list[str] = []
    checked = 0
    for suffix, kind in (("", "database"), ("-wal", "wal"), ("-shm", "shm")):
        path = database_path.with_name(database_path.name + suffix)
        if not path.exists():
            continue
        checked += 1
        content = path.read_bytes()
        if any(marker in content for marker in encoded):
            residual_kinds.append(kind)
    return DeletionProbeReport(
        checked_file_count=checked,
        residual_file_count=len(residual_kinds),
        residual_file_kinds=tuple(residual_kinds),
    )


def run_synthetic_deletion_probe(parent_directory: Path) -> DeletionProbeReport:
    """Exercise the real purge chain without writing synthetic data to production."""
    parent_directory.mkdir(parents=True, exist_ok=True)
    marker = f"PHYSICAL_DELETE_PROBE_{uuid4().hex}"
    with tempfile.TemporaryDirectory(
        prefix="psyavatar-delete-probe-",
        dir=parent_directory,
    ) as directory:
        database_path = Path(directory) / "probe.sqlite3"
        memories = MemoryRepository(database_path)
        memories.initialize()
        profiles = MemoryProfileRepository(database_path)
        episodes = MemoryEpisodeRepository(database_path)
        item = memories.add_active(
            user_id="synthetic_delete_probe",
            candidate=MemoryCandidate(
                source="synthetic_privacy_probe",
                contains_sensitive_content=True,
                text=marker,
                kind=MemoryKind.EPISODIC,
                source_turn_id="synthetic_delete_turn",
                aspect=MemoryAspect.FACT,
                subject_key="privacy.synthetic_delete_probe",
            ),
            now_ms=1_000,
        )
        profiles.record_observation(item, session_id="probe_session_1", now_ms=1_000)
        profiles.record_observation(item, session_id="probe_session_2", now_ms=2_000)
        MemoryConsolidator(profiles).rebuild_user(
            "synthetic_delete_probe",
            now_ms=3_000,
        )
        episodes.rebuild_session(
            user_id="synthetic_delete_probe",
            session_id="probe_session_1",
            memory_repository=memories,
            now_ms=3_000,
        )
        if not memories.purge(item.memory_id, user_id="synthetic_delete_probe"):
            return DeletionProbeReport(
                checked_file_count=1,
                residual_file_count=1,
                residual_file_kinds=("purge_failed",),
            )
        return scan_sqlite_residuals(database_path, (marker,))


def audit_memory_privacy(
    database_path: Path,
    *,
    as_of_ms: int | None = None,
    run_deletion_probe: bool = True,
) -> MemoryPrivacyAuditReport:
    """Audit privacy invariants without returning user content or identifiers."""
    timestamp = as_of_ms if as_of_ms is not None else int(time.time() * 1000)
    deletion_probe = (
        run_synthetic_deletion_probe(database_path.parent)
        if run_deletion_probe
        else DeletionProbeReport(0, 1, ("not_run",))
    )
    with closing(sqlite3.connect(database_path)) as connection:
        counts = {
            "observation_owner": _count(
                connection,
                """
                SELECT COUNT(*) FROM memory_observations AS observation
                JOIN memory_items AS item ON item.memory_id = observation.memory_id
                WHERE observation.user_id <> item.user_id
                """,
            ),
            "profile_owner": _count(
                connection,
                """
                SELECT COUNT(*) FROM memory_profile_evidence AS evidence
                JOIN memory_profiles AS profile ON profile.profile_id = evidence.profile_id
                JOIN memory_observations AS observation
                  ON observation.observation_id = evidence.observation_id
                JOIN memory_items AS item ON item.memory_id = evidence.memory_id
                WHERE profile.user_id <> item.user_id
                   OR observation.user_id <> item.user_id
                """,
            ),
            "episode_owner": _count(
                connection,
                """
                SELECT COUNT(*) FROM memory_episode_members AS member
                JOIN memory_episode_summaries AS summary
                  ON summary.summary_id = member.summary_id
                JOIN memory_items AS item ON item.memory_id = member.memory_id
                WHERE summary.user_id <> item.user_id
                """,
            ),
            "retrieval_owner": _count(
                connection,
                """
                SELECT COUNT(*) FROM memory_retrievals AS retrieval
                LEFT JOIN memory_items AS item ON item.memory_id = retrieval.memory_id
                LEFT JOIN memory_profiles AS profile
                  ON profile.profile_id = retrieval.memory_id
                WHERE COALESCE(item.user_id, profile.user_id) IS NOT NULL
                  AND retrieval.user_id <> COALESCE(item.user_id, profile.user_id)
                """,
            ),
            "shadow_owner": _count(
                connection,
                """
                SELECT COUNT(*) FROM memory_shadow_rankings AS ranking
                JOIN memory_shadow_runs AS run ON run.shadow_run_id = ranking.shadow_run_id
                LEFT JOIN memory_items AS item ON item.memory_id = ranking.memory_id
                LEFT JOIN memory_profiles AS profile ON profile.profile_id = ranking.memory_id
                WHERE COALESCE(item.user_id, profile.user_id) IS NOT NULL
                  AND run.user_id <> COALESCE(item.user_id, profile.user_id)
                """,
            ),
            "orphan_observation": _count(
                connection,
                """
                SELECT COUNT(*) FROM memory_observations AS observation
                LEFT JOIN memory_items AS item ON item.memory_id = observation.memory_id
                WHERE item.memory_id IS NULL
                """,
            ),
            "orphan_profile": _count(
                connection,
                """
                SELECT COUNT(*) FROM memory_profile_evidence AS evidence
                LEFT JOIN memory_profiles AS profile
                  ON profile.profile_id = evidence.profile_id
                LEFT JOIN memory_observations AS observation
                  ON observation.observation_id = evidence.observation_id
                LEFT JOIN memory_items AS item ON item.memory_id = evidence.memory_id
                WHERE profile.profile_id IS NULL
                   OR observation.observation_id IS NULL
                   OR item.memory_id IS NULL
                """,
            ),
            "orphan_episode": _count(
                connection,
                """
                SELECT COUNT(*) FROM memory_episode_members AS member
                LEFT JOIN memory_episode_summaries AS summary
                  ON summary.summary_id = member.summary_id
                LEFT JOIN memory_items AS item ON item.memory_id = member.memory_id
                WHERE summary.summary_id IS NULL OR item.memory_id IS NULL
                """,
            ),
            "orphan_retrieval": _count(
                connection,
                """
                SELECT COUNT(*) FROM memory_retrievals AS retrieval
                LEFT JOIN memory_items AS item ON item.memory_id = retrieval.memory_id
                LEFT JOIN memory_profiles AS profile
                  ON profile.profile_id = retrieval.memory_id
                WHERE item.memory_id IS NULL AND profile.profile_id IS NULL
                """,
            ),
            "orphan_shadow": _count(
                connection,
                """
                SELECT COUNT(*) FROM memory_shadow_rankings AS ranking
                LEFT JOIN memory_shadow_runs AS run
                  ON run.shadow_run_id = ranking.shadow_run_id
                LEFT JOIN memory_items AS item ON item.memory_id = ranking.memory_id
                LEFT JOIN memory_profiles AS profile ON profile.profile_id = ranking.memory_id
                WHERE run.shadow_run_id IS NULL
                   OR (item.memory_id IS NULL AND profile.profile_id IS NULL)
                """,
            ),
            "profile_link": _count(
                connection,
                """
                SELECT COUNT(*) FROM memory_profile_evidence AS evidence
                JOIN memory_observations AS observation
                  ON observation.observation_id = evidence.observation_id
                WHERE observation.memory_id <> evidence.memory_id
                """,
            ),
            "ineligible_episode": _count(
                connection,
                """
                SELECT COUNT(*) FROM memory_episode_members AS member
                JOIN memory_episode_summaries AS summary
                  ON summary.summary_id = member.summary_id
                JOIN memory_items AS item ON item.memory_id = member.memory_id
                WHERE item.state <> 'ACTIVE'
                   OR item.user_confirmed <> 1
                   OR item.integrity_flags_json <> '[]'
                   OR (item.expires_at_ms IS NOT NULL AND item.expires_at_ms <= ?)
                   OR item.purpose_scope <> summary.purpose_scope
                """,
                (timestamp,),
            ),
            "ineligible_profile": _count(
                connection,
                """
                SELECT COUNT(*) FROM memory_profile_evidence AS evidence
                JOIN memory_profiles AS profile ON profile.profile_id = evidence.profile_id
                JOIN memory_items AS item ON item.memory_id = evidence.memory_id
                WHERE profile.state IN ('ACTIVE', 'AWAITING_CONFIRMATION')
                  AND (
                    item.state <> 'ACTIVE'
                    OR item.user_confirmed <> 1
                    OR item.integrity_flags_json <> '[]'
                    OR (item.expires_at_ms IS NOT NULL AND item.expires_at_ms <= ?)
                    OR item.purpose_scope <> profile.purpose_scope
                  )
                """,
                (timestamp,),
            ),
        }
    return MemoryPrivacyAuditReport(
        media=scan_sqlite_media(database_path),
        deletion_probe=deletion_probe,
        observation_owner_mismatch_count=counts["observation_owner"],
        profile_evidence_owner_mismatch_count=counts["profile_owner"],
        episode_member_owner_mismatch_count=counts["episode_owner"],
        retrieval_owner_mismatch_count=counts["retrieval_owner"],
        shadow_ranking_owner_mismatch_count=counts["shadow_owner"],
        orphan_observation_count=counts["orphan_observation"],
        orphan_profile_evidence_count=counts["orphan_profile"],
        orphan_episode_member_count=counts["orphan_episode"],
        orphan_retrieval_count=counts["orphan_retrieval"],
        orphan_shadow_ranking_count=counts["orphan_shadow"],
        profile_evidence_link_mismatch_count=counts["profile_link"],
        ineligible_episode_member_count=counts["ineligible_episode"],
        ineligible_active_profile_evidence_count=counts["ineligible_profile"],
    )


def _count(
    connection: sqlite3.Connection,
    query: str,
    parameters: tuple[object, ...] = (),
) -> int:
    row = connection.execute(query, parameters).fetchone()
    if row is None:
        raise RuntimeError("privacy audit count query returned no row")
    return int(row[0])
