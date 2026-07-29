import sqlite3

import pytest

from app.evals.privacy import scan_sqlite_media
from app.events.store import EventStore


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
