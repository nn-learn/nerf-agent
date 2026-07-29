from enum import StrEnum
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from app.vision.frame import VideoFrame


class FrameGateDecision(StrEnum):
    ACCEPT = "ACCEPT"
    BLACK = "BLACK"
    BLURRY = "BLURRY"
    DUPLICATE = "DUPLICATE"


@lru_cache(maxsize=2)
def _dct_matrix(size: int) -> NDArray[np.float64]:
    positions = np.arange(size, dtype=np.float64)
    frequencies = positions[:, None]
    matrix = np.cos(np.pi * (2 * positions + 1) * frequencies / (2 * size))
    matrix[0, :] *= np.sqrt(1 / size)
    matrix[1:, :] *= np.sqrt(2 / size)
    return matrix


def _grayscale(rgb: NDArray[np.uint8]) -> NDArray[np.float64]:
    return (
        rgb[:, :, 0].astype(np.float64) * 0.299
        + rgb[:, :, 1].astype(np.float64) * 0.587
        + rgb[:, :, 2].astype(np.float64) * 0.114
    )


def _laplacian_variance(gray: NDArray[np.float64]) -> float:
    center = gray[1:-1, 1:-1]
    laplacian = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4 * center
    )
    return float(laplacian.var())


def _perceptual_hash(gray: NDArray[np.float64]) -> int:
    image = Image.fromarray(gray.astype(np.uint8), mode="L")
    resized = image.resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(resized, dtype=np.float64)
    matrix = _dct_matrix(32)
    low_frequency = (matrix @ pixels @ matrix.T)[:8, :8]
    coefficients = low_frequency.flatten()[1:]
    median = float(np.median(coefficients))
    result = 0
    for bit in coefficients > median:
        result = (result << 1) | int(bit)
    return result


class FrameGate:
    def __init__(
        self,
        *,
        black_mean_threshold: float = 8.0,
        blur_variance_threshold: float = 45.0,
        duplicate_hamming_threshold: int = 4,
    ) -> None:
        self._black_threshold = black_mean_threshold
        self._blur_threshold = blur_variance_threshold
        self._duplicate_threshold = duplicate_hamming_threshold
        self._previous_hash: int | None = None

    def evaluate(self, frame: VideoFrame) -> FrameGateDecision:
        gray = _grayscale(frame.rgb)
        if float(gray.mean()) < self._black_threshold:
            return FrameGateDecision.BLACK
        if _laplacian_variance(gray) < self._blur_threshold:
            return FrameGateDecision.BLURRY

        current_hash = _perceptual_hash(gray)
        if (
            self._previous_hash is not None
            and (current_hash ^ self._previous_hash).bit_count()
            <= self._duplicate_threshold
        ):
            return FrameGateDecision.DUPLICATE
        self._previous_hash = current_hash
        return FrameGateDecision.ACCEPT
