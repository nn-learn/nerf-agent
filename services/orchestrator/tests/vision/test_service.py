import asyncio
from pathlib import Path

import numpy as np
import pytest

from app.contracts.vision import VisualObservation
from app.vision.frame import EncodedFrame, VideoFrame
from app.vision.service import (
    VisionConsentRevoked,
    VisionService,
    evaluate_fixture,
)


def sample_frame(captured_at_ms: int = 1_000) -> VideoFrame:
    checker = (
        (np.indices((64, 64)).sum(axis=0) % 2) * 255
    ).astype(np.uint8)
    rgb = np.repeat(checker[:, :, None], 3, axis=2)
    return VideoFrame.from_rgb(rgb, captured_at_ms=captured_at_ms)


class MockVisionProvider:
    def __init__(self) -> None:
        self.calls: list[list[EncodedFrame]] = []

    async def analyze(
        self,
        frames: list[EncodedFrame],
        *,
        trigger_turn_id: str,
        source_event_id: str,
    ) -> VisualObservation:
        self.calls.append(frames)
        captured_at_ms = frames[-1].captured_at_ms
        return VisualObservation(
            observation_id="vo_mock",
            source_event_id=source_event_id,
            trigger_turn_id=trigger_turn_id,
            captured_at_ms=captured_at_ms,
            valid_until_ms=captured_at_ms + 10_000,
            frame_count=len(frames),
            scene_summary="用户手持一张记录卡",
            objects=[],
            visible_text_summary="部分日期可见",
            action_summary="记录卡朝向摄像头",
            confidence=0.85,
            uncertainties=[],
            contains_sensitive_content=True,
            prohibited_inferences_removed=[],
        )


@pytest.mark.asyncio
async def test_revocation_clears_frames_before_provider_call() -> None:
    """Catches revoked frames surviving in a queue or reaching a provider."""
    provider = MockVisionProvider()
    service = VisionService(provider)
    await service.grant_camera_consent()
    await service.enqueue(sample_frame())

    await service.revoke_camera_consent()

    assert service.buffer_size == 0
    with pytest.raises(VisionConsentRevoked):
        await service.analyze(
            trigger_turn_id="turn_1",
            source_event_id="evt_1",
        )
    assert provider.calls == []


@pytest.mark.asyncio
async def test_revocation_during_provider_call_discards_observation() -> None:
    """Catches a slow provider returning camera context after consent was revoked."""
    started = asyncio.Event()
    release = asyncio.Event()

    class SlowProvider(MockVisionProvider):
        async def analyze(
            self,
            frames: list[EncodedFrame],
            *,
            trigger_turn_id: str,
            source_event_id: str,
        ) -> VisualObservation:
            started.set()
            await release.wait()
            return await super().analyze(
                frames,
                trigger_turn_id=trigger_turn_id,
                source_event_id=source_event_id,
            )

    provider = SlowProvider()
    service = VisionService(provider)
    await service.grant_camera_consent()
    await service.enqueue(sample_frame())
    pending = asyncio.create_task(
        service.analyze(trigger_turn_id="turn_1", source_event_id="evt_1")
    )
    await started.wait()

    await service.revoke_camera_consent()
    release.set()

    with pytest.raises(VisionConsentRevoked):
        await pending


def test_checked_in_vision_fixture_has_no_appearance_diagnosis() -> None:
    """Catches regression in the deterministic mock vision acceptance set."""
    fixture_path = Path(__file__).parents[4] / "evals" / "vision_cases.jsonl"

    report = evaluate_fixture(fixture_path)

    assert report.total == 3
    assert report.valid_observations == 2
    assert report.rejected_appearance_inferences == 1
