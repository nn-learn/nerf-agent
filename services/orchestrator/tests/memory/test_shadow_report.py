import json
from pathlib import Path

import pytest

from app.memory.run_shadow_report import main
from app.memory.shadow import MemoryShadowRepository
from tests.memory.test_shadow import initialized_subject


@pytest.mark.asyncio
async def test_report_cli_omits_subject_and_per_turn_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_path = tmp_path / "events.sqlite3"
    user_id = await initialized_subject(database_path)
    MemoryShadowRepository(database_path).set_consent(
        user_id=user_id,
        granted=True,
        policy_version="policy-v1",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_shadow_report",
            "--database",
            str(database_path),
            "--user-id",
            user_id,
        ],
    )

    main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["run_count"] == 0
    assert payload["reliable"] is False
    assert user_id not in json.dumps(payload)
    assert "runs" not in payload
