import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PrivacyScanReport:
    raw_audio_blob_count: int
    raw_video_blob_count: int
    camera_frame_blob_count: int
    binary_column_count: int


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def scan_sqlite_media(database_path: Path) -> PrivacyScanReport:
    raw_audio = 0
    raw_video = 0
    camera_frames = 0
    binary_columns = 0
    with sqlite3.connect(database_path) as connection:
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
