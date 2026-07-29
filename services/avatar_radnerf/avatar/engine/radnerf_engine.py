import importlib
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RadNerfEngineConfig:
    source_root: Path
    person_id: str = "222"
    workspace_name: str = "trial_222_torso"
    wav2vec_model: str = "cpierse/wav2vec2-large-xlsr-53-esperanto"


class RadNerfEngine:
    """Read-only loader around the existing RAD-NeRF source and person assets."""

    def __init__(self, config: RadNerfEngineConfig) -> None:
        self.config = config
        self.loaded = False

    @property
    def person_data_path(self) -> Path:
        return self.config.source_root / "data" / self.config.person_id

    @property
    def checkpoint_path(self) -> Path:
        return (
            self.config.source_root
            / self.config.workspace_name
            / "checkpoints"
            / "ngp.pth"
        )

    def load(self) -> None:
        if not self.person_data_path.is_dir():
            raise FileNotFoundError(self.person_data_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        source = str(self.config.source_root.resolve())
        inserted = source not in sys.path
        if inserted:
            sys.path.insert(0, source)
        try:
            importlib.import_module("nerf.network")
            importlib.import_module("nerf.provider")
        finally:
            if inserted:
                sys.path.remove(source)
        self.loaded = True
