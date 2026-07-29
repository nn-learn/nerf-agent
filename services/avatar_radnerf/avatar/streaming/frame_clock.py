from dataclasses import dataclass, field
from fractions import Fraction


@dataclass(slots=True)
class FrameClock:
    fps: int
    start_pts_ms: int = 0
    _frame_index: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.start_pts_ms < 0:
            raise ValueError("start PTS must be non-negative")

    def next_pts_ms(self) -> int:
        offset = Fraction(self._frame_index * 1_000, self.fps)
        pts_ms = self.start_pts_ms + int(offset)
        self._frame_index += 1
        return pts_ms
