import json

import httpx
import pytest
from pydantic import SecretStr

from app.contracts.session import CancellationRegistry
from app.providers.qwen_text import QwenTextProvider
from app.safety.models import RiskLevel


@pytest.mark.asyncio
async def test_qwen_text_keeps_model_and_validates_full_response_before_streaming() -> None:
    """Catches qwen-max replacement and unvalidated token-by-token publication."""
    captured: dict[str, object] = {}
    agent_response = {
        "spoken_text": "听起来你最近承受了不少压力。我们可以先梳理最困扰的一件事。",
        "display_text": "我们可以先梳理最困扰的一件事。",
        "support_mode": "listen",
        "risk_level": "GREEN",
        "evidence_ids": [],
        "visual_observation_ids": [],
        "action_proposals": [],
        "memory_candidates": [],
        "avatar_style": "warm",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                agent_response,
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            },
        )

    registry = CancellationRegistry()
    token = await registry.issue("turn_1")
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
    ) as client:
        provider = QwenTextProvider(
            api_key=SecretStr("not-a-real-key"),
            registry=registry,
            model="qwen-max",
            client=client,
        )
        response = await provider.respond(
            {
                "transcript": "最近压力很大",
                "reviewed_evidence": [],
            },
            turn_id="turn_1",
            cancel_token=token,
            risk_level=RiskLevel.GREEN,
        )
        spoken = [
            sentence
            async for sentence in provider.stream_spoken(
                response,
                turn_id="turn_1",
                cancel_token=token,
            )
        ]

    assert captured["model"] == "qwen-max"
    assert captured["response_format"] == {"type": "json_object"}
    assert response.risk_level is RiskLevel.GREEN
    assert "".join(spoken) == response.spoken_text
