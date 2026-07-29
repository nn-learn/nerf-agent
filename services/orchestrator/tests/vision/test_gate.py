import numpy as np

from app.vision.frame import VideoFrame
from app.vision.gate import FrameGate, FrameGateDecision


def _frame(rgb: np.ndarray, captured_at_ms: int = 1_000) -> VideoFrame:
    return VideoFrame.from_rgb(rgb.astype(np.uint8), captured_at_ms=captured_at_ms)


def test_gate_rejects_black_blurry_and_duplicate_frames() -> None:
    """Catches low-information camera data being sent to the vision provider."""
    gate = FrameGate()
    black = np.zeros((64, 64, 3), dtype=np.uint8)
    flat_gray = np.full((64, 64, 3), 110, dtype=np.uint8)
    checker = (
        (np.indices((64, 64)).sum(axis=0) % 2) * 255
    ).astype(np.uint8)
    sharp = np.repeat(checker[:, :, None], 3, axis=2)

    assert gate.evaluate(_frame(black)) is FrameGateDecision.BLACK
    assert gate.evaluate(_frame(flat_gray)) is FrameGateDecision.BLURRY
    assert gate.evaluate(_frame(sharp)) is FrameGateDecision.ACCEPT
    assert gate.evaluate(_frame(sharp, 1_100)) is FrameGateDecision.DUPLICATE


def test_accepted_frame_is_resized_and_encoded_in_memory() -> None:
    """Catches oversized or lossless camera payloads entering the provider queue."""
    rgb = np.full((1_600, 900, 3), 128, dtype=np.uint8)
    frame = _frame(rgb)

    encoded = frame.to_jpeg(max_long_edge=1_280, quality=80)

    assert max(encoded.width, encoded.height) == 1_280
    assert encoded.jpeg.startswith(b"\xff\xd8")
    assert encoded.captured_at_ms == frame.captured_at_ms
