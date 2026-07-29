import argparse
import asyncio
import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from app.contracts.vision import VisualObservation
from app.vision.frame import EncodedFrame, VideoFrame
from app.vision.gate import FrameGate, FrameGateDecision


class VisionProvider(Protocol):
    async def analyze(
        self,
        frames: list[EncodedFrame],
        *,
        trigger_turn_id: str,
        source_event_id: str,
    ) -> VisualObservation: ...


class VisionConsentRevoked(PermissionError):
    pass


class VisionBufferEmpty(LookupError):
    pass


class VisionService:
    def __init__(
        self,
        provider: VisionProvider,
        *,
        gate: FrameGate | None = None,
        max_frames: int = 4,
    ) -> None:
        if not 1 <= max_frames <= 4:
            raise ValueError("vision buffer must hold between one and four frames")
        self.provider = provider
        self._gate = gate or FrameGate()
        self._frames: deque[EncodedFrame] = deque(maxlen=max_frames)
        self._consent_granted = False
        self._consent_generation = 0
        self._lock = asyncio.Lock()

    @property
    def buffer_size(self) -> int:
        return len(self._frames)

    async def grant_camera_consent(self) -> None:
        async with self._lock:
            self._consent_granted = True
            self._consent_generation += 1

    async def revoke_camera_consent(self) -> None:
        async with self._lock:
            self._consent_granted = False
            self._consent_generation += 1
            self._frames.clear()

    async def enqueue(self, frame: VideoFrame) -> FrameGateDecision:
        async with self._lock:
            if not self._consent_granted:
                raise VisionConsentRevoked("camera consent is not active")
            decision = self._gate.evaluate(frame)
            if decision is FrameGateDecision.ACCEPT:
                self._frames.append(frame.to_jpeg())
            return decision

    async def analyze(
        self,
        *,
        trigger_turn_id: str,
        source_event_id: str,
    ) -> VisualObservation:
        async with self._lock:
            if not self._consent_granted:
                raise VisionConsentRevoked("camera consent is not active")
            if not self._frames:
                raise VisionBufferEmpty("no accepted camera frames")
            generation = self._consent_generation
            frames = list(self._frames)
            self._frames.clear()

        observation = await self.provider.analyze(
            frames,
            trigger_turn_id=trigger_turn_id,
            source_event_id=source_event_id,
        )

        async with self._lock:
            if (
                not self._consent_granted
                or generation != self._consent_generation
            ):
                raise VisionConsentRevoked(
                    "camera consent changed during vision analysis"
                )
        return observation


@dataclass(frozen=True, slots=True)
class VisionFixtureReport:
    total: int
    valid_observations: int
    rejected_appearance_inferences: int


def evaluate_fixture(path: Path) -> VisionFixtureReport:
    total = 0
    valid = 0
    rejected_appearance = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total += 1
        case = json.loads(line)
        name = str(case.get("name", f"case_{total}"))
        if case.get("case_type") == "frame_gate":
            if not isinstance(case.get("expected_decision"), str):
                raise ValueError(f"invalid frame-gate fixture contract: {name}")
            continue
        expected_valid = case.get("expected_valid")
        payload = case.get("payload")
        if not isinstance(expected_valid, bool) or not isinstance(payload, dict):
            raise ValueError(f"invalid vision fixture contract: {name}")
        try:
            VisualObservation.model_validate(payload)
        except ValidationError as error:
            if expected_valid:
                raise ValueError(
                    f"expected valid vision fixture was rejected: {name}"
                ) from error
            if "psychiatric inference" not in str(error):
                raise ValueError(
                    f"fixture failed for an unexpected reason: {name}"
                ) from error
            rejected_appearance += 1
        else:
            if not expected_valid:
                raise ValueError(
                    f"unsafe vision fixture was unexpectedly accepted: {name}"
                )
            valid += 1
    return VisionFixtureReport(
        total=total,
        valid_observations=valid,
        rejected_appearance_inferences=rejected_appearance,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the privacy-first mock vision contract."
    )
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--provider", choices=("mock",), default="mock")
    arguments = parser.parse_args()
    report = evaluate_fixture(arguments.fixture)
    print(json.dumps(asdict(report), ensure_ascii=False))


if __name__ == "__main__":
    main()
