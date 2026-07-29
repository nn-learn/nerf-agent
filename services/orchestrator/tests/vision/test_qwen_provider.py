import json

import httpx
import pytest
from pydantic import SecretStr

from app.providers.qwen_vision import QwenVisionProvider
from app.vision.frame import EncodedFrame


@pytest.mark.asyncio
async def test_qwen_provider_uses_non_thinking_structured_multiframe_request() -> None:
    """Catches model drift, unstructured output and accidental single-frame upload."""
    captured_request: dict[str, object] = {}
    analysis = {
        "scene_summary": "用户手持一张睡眠记录卡",
        "objects": [
            {
                "label": "sleep_log_card",
                "attributes": ["paper", "handheld"],
                "confidence": 0.91,
            }
        ],
        "visible_text_summary": "部分日期可见",
        "action_summary": "记录卡朝向摄像头",
        "confidence": 0.86,
        "uncertainties": ["右下角数字模糊"],
        "contains_sensitive_content": True,
        "prohibited_inferences_removed": [],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(analysis, ensure_ascii=False),
                            }
                        ],
                    }
                ]
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://model.example/compatible-mode/v1",
    ) as client:
        provider = QwenVisionProvider(
            api_key=SecretStr("not-a-real-key"),
            model="qwen3.6-flash",
            client=client,
        )
        frames = [
            EncodedFrame(
                jpeg=b"\xff\xd8frame-one\xff\xd9",
                captured_at_ms=1_000,
                sha256="a" * 64,
                width=640,
                height=360,
            ),
            EncodedFrame(
                jpeg=b"\xff\xd8frame-two\xff\xd9",
                captured_at_ms=1_500,
                sha256="b" * 64,
                width=640,
                height=360,
            ),
        ]

        observation = await provider.analyze(
            frames,
            trigger_turn_id="turn_1",
            source_event_id="evt_1",
        )

    assert captured_request["model"] == "qwen3.6-flash"
    assert captured_request["enable_thinking"] is False
    text_config = captured_request["text"]
    assert isinstance(text_config, dict)
    assert text_config["format"]["type"] == "json_schema"
    request_input = captured_request["input"]
    assert isinstance(request_input, list)
    content = request_input[0]["content"]
    assert sum(item["type"] == "input_image" for item in content) == 2
    assert observation.frame_count == 2
    assert observation.captured_at_ms == 1_500
    assert observation.valid_until_ms == 11_500
