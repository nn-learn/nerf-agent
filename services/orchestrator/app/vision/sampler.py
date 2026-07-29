from typing import Literal, Protocol


class MillisecondClock(Protocol):
    def now_ms(self) -> int: ...


class AdaptiveSampler:
    MAX_BURST_MS = 5_000

    def __init__(
        self,
        *,
        clock: MillisecondClock,
        idle_fps: float = 0.5,
        burst_fps: float = 2.0,
    ) -> None:
        if idle_fps <= 0 or burst_fps <= 0:
            raise ValueError("sampling rates must be positive")
        self._clock = clock
        self._idle_interval_ms = round(1_000 / idle_fps)
        self._burst_interval_ms = round(1_000 / burst_fps)
        self._mode: Literal["idle", "burst"] = "idle"
        self._burst_until_ms: int | None = None
        self._next_sample_ms = clock.now_ms()

    @property
    def mode(self) -> Literal["idle", "burst"]:
        self._expire_burst(self._clock.now_ms())
        return self._mode

    def start_burst(self, *, duration_ms: int = MAX_BURST_MS) -> None:
        if duration_ms <= 0:
            raise ValueError("burst duration must be positive")
        now_ms = self._clock.now_ms()
        duration = min(duration_ms, self.MAX_BURST_MS)
        self._mode = "burst"
        self._burst_until_ms = now_ms + duration
        self._next_sample_ms = now_ms

    def should_sample(self) -> bool:
        now_ms = self._clock.now_ms()
        self._expire_burst(now_ms)
        if now_ms < self._next_sample_ms:
            return False
        interval = (
            self._burst_interval_ms
            if self._mode == "burst"
            else self._idle_interval_ms
        )
        self._next_sample_ms = now_ms + interval
        return True

    def _expire_burst(self, now_ms: int) -> None:
        if (
            self._mode == "burst"
            and self._burst_until_ms is not None
            and now_ms >= self._burst_until_ms
        ):
            self._mode = "idle"
            self._next_sample_ms = max(
                self._next_sample_ms,
                self._burst_until_ms + self._idle_interval_ms,
            )
            self._burst_until_ms = None
