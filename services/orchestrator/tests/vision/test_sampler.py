from app.vision.sampler import AdaptiveSampler


class FakeClock:
    def __init__(self) -> None:
        self.now = 0

    def now_ms(self) -> int:
        return self.now

    def advance_ms(self, duration_ms: int) -> None:
        self.now += duration_ms


def test_burst_never_exceeds_five_seconds_or_ten_frames() -> None:
    """Catches accidental always-on high-rate camera sampling."""
    clock = FakeClock()
    sampler = AdaptiveSampler(clock=clock, idle_fps=0.5, burst_fps=2.0)
    sampler.start_burst(duration_ms=5_000)

    accepted = []
    for _ in range(70):
        accepted.append(sampler.should_sample())
        clock.advance_ms(100)

    assert sum(accepted) <= 10
    assert sampler.mode == "idle"
