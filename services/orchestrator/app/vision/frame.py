import hashlib
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from numpy.typing import NDArray
from PIL import Image


@dataclass(frozen=True, slots=True)
class EncodedFrame:
    jpeg: bytes
    captured_at_ms: int
    sha256: str
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class VideoFrame:
    rgb: NDArray[np.uint8]
    captured_at_ms: int
    sha256: str

    @classmethod
    def from_rgb(
        cls,
        rgb: NDArray[np.uint8],
        *,
        captured_at_ms: int,
    ) -> "VideoFrame":
        if rgb.dtype != np.uint8:
            raise ValueError("video frame must use uint8 RGB values")
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("video frame must have H x W x 3 shape")
        if rgb.shape[0] < 2 or rgb.shape[1] < 2:
            raise ValueError("video frame dimensions are too small")
        immutable = np.ascontiguousarray(rgb.copy())
        immutable.setflags(write=False)
        digest = hashlib.sha256(immutable.tobytes()).hexdigest()
        return cls(
            rgb=immutable,
            captured_at_ms=captured_at_ms,
            sha256=digest,
        )

    def to_jpeg(
        self,
        *,
        max_long_edge: int = 1_280,
        quality: int = 80,
    ) -> EncodedFrame:
        if max_long_edge <= 0:
            raise ValueError("max_long_edge must be positive")
        if not 1 <= quality <= 95:
            raise ValueError("JPEG quality must be between 1 and 95")

        image = Image.fromarray(self.rgb, mode="RGB")
        width, height = image.size
        long_edge = max(width, height)
        if long_edge > max_long_edge:
            scale = max_long_edge / long_edge
            resized = (
                max(1, round(width * scale)),
                max(1, round(height * scale)),
            )
            image = image.resize(resized, Image.Resampling.LANCZOS)

        buffer = BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=quality,
            optimize=True,
        )
        encoded_width, encoded_height = image.size
        return EncodedFrame(
            jpeg=buffer.getvalue(),
            captured_at_ms=self.captured_at_ms,
            sha256=self.sha256,
            width=encoded_width,
            height=encoded_height,
        )
