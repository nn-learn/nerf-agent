import sqlite3

import pytest

from app.evals.privacy import (
    audit_memory_privacy,
    scan_sqlite_media,
    scan_sqlite_residuals,
)
from app.events.store import EventStore
from app.memory.consolidation import MemoryConsolidator, MemoryProfileRepository
from app.memory.episodes import MemoryEpisodeRepository
from app.memory.models import MemoryAspect, MemoryCandidate, MemoryKind
from app.memory.repository import MemoryRepository


@pytest.mark.asyncio
async def test_database_contains_no_raw_media_blobs(tmp_path) -> None:
    """Catches schema or event changes that persist microphone/camera bytes."""
    database_path = tmp_path / "events.sqlite3"
    store = EventStore(database_path)
    await store.initialize()
    await store.append_payload(
        session_id="session_eval",
        event_type="vision.observation.ready",
        payload={"summary": "桌面上可见一张睡眠记录卡"},
    )

    report = scan_sqlite_media(database_path)

    assert report.raw_audio_blob_count == 0
    assert report.raw_video_blob_count == 0
    assert report.camera_frame_blob_count == 0
    assert report.binary_column_count == 0


def test_privacy_scanner_detects_an_accidental_media_blob(tmp_path) -> None:
    database_path = tmp_path / "unsafe.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE unsafe(camera_frame BLOB)")
        connection.execute(
            "INSERT INTO unsafe(camera_frame) VALUES (?)",
            (b"\xff\xd8\xff",),
        )

    report = scan_sqlite_media(database_path)

    assert report.camera_frame_blob_count == 1
    assert report.binary_column_count == 1


def test_memory_purge_removes_sensitive_bytes_from_sqlite_files(tmp_path) -> None:
    database_path = tmp_path / "memory.sqlite3"
    memories = MemoryRepository(database_path)
    memories.initialize()
    profiles = MemoryProfileRepository(database_path)
    episodes = MemoryEpisodeRepository(database_path)
    marker = "PHYSICAL_DELETE_MARKER_7ef4c5bd"
    item = memories.add_active(
        user_id="user_delete",
        candidate=MemoryCandidate(
            source="transcript",
            contains_sensitive_content=True,
            text=marker,
            kind=MemoryKind.EPISODIC,
            source_turn_id="turn_delete",
            aspect=MemoryAspect.FACT,
            subject_key="privacy.deletion_marker",
        ),
        now_ms=1_000,
    )
    profiles.record_observation(item, session_id="session_1", now_ms=1_000)
    profiles.record_observation(item, session_id="session_2", now_ms=2_000)
    MemoryConsolidator(profiles).rebuild_user("user_delete", now_ms=3_000)
    episodes.rebuild_session(
        user_id="user_delete",
        session_id="session_1",
        memory_repository=memories,
        now_ms=3_000,
    )

    assert memories.purge(item.memory_id, user_id="user_delete") is True

    report = scan_sqlite_residuals(database_path, (marker,))

    assert report.passed is True


def test_clean_memory_database_passes_full_privacy_audit(tmp_path) -> None:
    database_path = tmp_path / "memory.sqlite3"
    MemoryRepository(database_path).initialize()

    report = audit_memory_privacy(database_path, as_of_ms=10_000)

    assert report.passed is True
    assert report.owner_mismatch_count == 0
    assert report.orphan_reference_count == 0
    assert report.deletion_probe.passed is True


def test_privacy_audit_detects_cross_owner_and_orphan_references(tmp_path) -> None:
    database_path = tmp_path / "memory.sqlite3"
    memories = MemoryRepository(database_path)
    memories.initialize()
    item = memories.add_active(
        user_id="owner_a",
        candidate=MemoryCandidate(
            source="transcript",
            contains_sensitive_content=False,
            text="Owner A prefers concise replies",
            kind=MemoryKind.SEMANTIC,
            source_turn_id="turn_a",
            aspect=MemoryAspect.PREFERENCE,
            subject_key="communication.response_style",
        ),
        now_ms=1_000,
    )
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO memory_observations(
                observation_id, memory_id, user_id, session_id,
                turn_id, valid_at_ms, observed_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("observation_cross_owner", item.memory_id, "owner_b", "s1", "t1", 1, 1),
        )
        connection.execute(
            """
            INSERT INTO memory_retrievals(
                retrieval_id, user_id, memory_id, session_id, turn_id,
                score, relevance_score, reason_codes_json, used_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("retrieval_cross_owner", "owner_b", item.memory_id, "s1", "t1", 1, 1, "[]", 1),
        )
        connection.execute(
            """
            INSERT INTO memory_retrievals(
                retrieval_id, user_id, memory_id, session_id, turn_id,
                score, relevance_score, reason_codes_json, used_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("retrieval_orphan", "owner_a", "missing", "s1", "t2", 1, 1, "[]", 2),
        )

    report = audit_memory_privacy(database_path, as_of_ms=10_000)

    assert report.passed is False
    assert report.observation_owner_mismatch_count == 1
    assert report.retrieval_owner_mismatch_count == 1
    assert report.orphan_retrieval_count == 1
