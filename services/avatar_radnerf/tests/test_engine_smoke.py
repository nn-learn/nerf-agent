import os
from pathlib import Path

import pytest

from avatar.engine.radnerf_engine import RadNerfEngine, RadNerfEngineConfig


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("PSYAVATAR_RUN_GPU_TESTS") != "1",
    reason="explicit Python 3.10 CUDA environment is required",
)
def test_engine_loads_existing_person_222_without_copying_assets() -> None:
    source_root = Path(os.environ["RADNERF_SOURCE_ROOT"])
    engine = RadNerfEngine(
        RadNerfEngineConfig(
            source_root=source_root,
            person_id="222",
        )
    )

    engine.load()

    assert engine.loaded
    assert engine.person_data_path == source_root / "data" / "222"
