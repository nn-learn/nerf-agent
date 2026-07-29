from avatar.streaming.frame_clock import FrameClock


def test_frame_pts_uses_session_clock() -> None:
    """Catches video timestamps drifting from 25 FPS session time."""
    clock = FrameClock(fps=25, start_pts_ms=2_000)

    assert [clock.next_pts_ms() for _ in range(3)] == [
        2_000,
        2_040,
        2_080,
    ]
