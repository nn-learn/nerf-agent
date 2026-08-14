import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from app.events.store import EventStore
from app.memory.identity import InvalidMemorySubjectToken, MemorySubjectStore
from app.memory.repository import MemoryRepository


@pytest.mark.asyncio
async def test_subject_token_reuses_user_without_storing_plaintext(tmp_path: Path) -> None:
    database_path = tmp_path / "events.sqlite3"
    event_store = EventStore(database_path)
    await event_store.initialize()
    await event_store.reserve_session("session_1")
    await event_store.reserve_session("session_2")
    repository = MemoryRepository(database_path)
    repository.initialize()
    subjects = MemorySubjectStore(database_path)

    first_user, token = subjects.issue_or_resolve(
        session_id="session_1",
        supplied_token=None,
        now_ms=1_000,
    )
    assert token is not None
    second_user, returned = subjects.issue_or_resolve(
        session_id="session_2",
        supplied_token=token,
        now_ms=2_000,
    )

    assert first_user == second_user
    assert returned is None
    with closing(sqlite3.connect(database_path)) as connection:
        row = connection.execute(
            "SELECT token_digest FROM memory_subjects WHERE user_id = ?",
            (first_user,),
        ).fetchone()
    assert row is not None
    assert row[0] != token
    assert token.encode("utf-8") not in database_path.read_bytes()


@pytest.mark.asyncio
async def test_invalid_subject_token_is_rejected_before_binding(tmp_path: Path) -> None:
    database_path = tmp_path / "events.sqlite3"
    store = EventStore(database_path)
    await store.initialize()
    await store.reserve_session("session_1")
    MemoryRepository(database_path).initialize()
    subjects = MemorySubjectStore(database_path)

    with pytest.raises(InvalidMemorySubjectToken):
        subjects.issue_or_resolve(
            session_id="session_1",
            supplied_token="pms_" + "x" * 40,
        )

    with pytest.raises(KeyError):
        subjects.user_for_session("session_1")
