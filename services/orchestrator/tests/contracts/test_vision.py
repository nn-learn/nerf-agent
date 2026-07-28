import pytest
from pydantic import ValidationError

from app.contracts.vision import VisualObservation


def observation_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "observation_id": "vo_1",
        "source_event_id": "evt_1",
        "trigger_turn_id": "turn_1",
        "captured_at_ms": 1_000,
        "valid_until_ms": 11_000,
        "frame_count": 1,
        "scene_summary": "用户手持睡眠记录卡",
        "objects": [
            {
                "label": "sleep_log_card",
                "attributes": ["paper", "handheld"],
                "confidence": 0.91,
            }
        ],
        "visible_text_summary": "部分日期可见，个别数字不清晰",
        "action_summary": "用户将记录卡朝向摄像头",
        "confidence": 0.86,
        "uncertainties": ["右下角数字模糊"],
        "contains_sensitive_content": True,
        "prohibited_inferences_removed": [],
    }
    payload.update(overrides)
    return payload


def test_visual_observation_expires_at_the_ten_second_boundary() -> None:
    """Catches stale camera context being reused in a later conversational turn."""
    item = VisualObservation.model_validate(observation_payload())

    assert item.is_fresh(10_999)
    assert not item.is_fresh(11_000)


def test_visual_observation_requires_exact_ten_second_lifetime() -> None:
    """Catches providers or callers that extend visual retention beyond the contract."""
    with pytest.raises(ValidationError, match="10 seconds"):
        VisualObservation.model_validate(observation_payload(valid_until_ms=11_001))


def test_visual_observation_rejects_psychiatric_inference_from_appearance() -> None:
    """Catches visual output that turns appearance into a psychiatric diagnosis."""
    with pytest.raises(ValidationError, match="psychiatric inference"):
        VisualObservation.model_validate(
            observation_payload(scene_summary="用户看起来患有抑郁症")
        )


def test_visible_document_text_is_not_misclassified_as_appearance_inference() -> None:
    """Catches an overbroad filter that hides literal text visible on a document."""
    item = VisualObservation.model_validate(
        observation_payload(visible_text_summary="标题包含“抑郁症筛查量表”")
    )

    assert item.visible_text_summary == "标题包含“抑郁症筛查量表”"
