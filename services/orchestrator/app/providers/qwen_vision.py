import base64
from uuid import uuid4

import httpx
from pydantic import BaseModel, Field, SecretStr

from app.contracts.vision import VisualObject, VisualObservation
from app.vision.frame import EncodedFrame

VISION_POLICY = """\
只描述画面中直接可见的物体、环境、文字概况和动作。
不得根据脸部、眼神、姿态或外观推断心理疾病、人格、自伤或危险倾向。
无法确认时写入 uncertainties，不要猜测身份或敏感属性。
不要输出诊断、治疗建议、药物建议或风险等级。
"""


class VisualAnalysis(BaseModel):
    scene_summary: str
    objects: list[VisualObject] = Field(default_factory=list)
    visible_text_summary: str
    action_summary: str
    confidence: float = Field(ge=0, le=1)
    uncertainties: list[str] = Field(default_factory=list)
    contains_sensitive_content: bool
    prohibited_inferences_removed: list[str] = Field(default_factory=list)


class VisionProviderResponseError(ValueError):
    pass


class QwenVisionProvider:
    """Qwen3.6-Flash adapter that never persists or logs image payloads."""

    def __init__(
        self,
        *,
        api_key: SecretStr,
        model: str = "qwen3.6-flash",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=30)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def analyze(
        self,
        frames: list[EncodedFrame],
        *,
        trigger_turn_id: str,
        source_event_id: str,
    ) -> VisualObservation:
        if not 1 <= len(frames) <= 4:
            raise ValueError("vision requests require between one and four frames")

        content: list[dict[str, object]] = [
            {
                "type": "input_image",
                "image_url": (
                    "data:image/jpeg;base64,"
                    + base64.b64encode(frame.jpeg).decode("ascii")
                ),
            }
            for frame in frames
        ]
        content.append(
            {
                "type": "input_text",
                "text": (
                    "按给定 JSON Schema 返回本轮视觉观察。"
                    "只概括可见文字，不转录身份证号、住址或联系方式。"
                ),
            }
        )
        request_body: dict[str, object] = {
            "model": self._model,
            "instructions": VISION_POLICY,
            "input": [{"role": "user", "content": content}],
            "enable_thinking": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "visual_observation",
                    "strict": True,
                    "schema": VisualAnalysis.model_json_schema(),
                }
            },
        }
        try:
            response = await self._client.post(
                f"{self._base_url}/responses",
                headers={
                    "Authorization": (
                        f"Bearer {self._api_key.get_secret_value()}"
                    ),
                    "Content-Type": "application/json",
                },
                json=request_body,
            )
            response.raise_for_status()
            output_text = self._extract_output_text(response.json())
            analysis = VisualAnalysis.model_validate_json(output_text)
        except (httpx.HTTPError, ValueError) as error:
            raise VisionProviderResponseError(
                "vision provider returned an invalid or unavailable response"
            ) from error

        captured_at_ms = max(frame.captured_at_ms for frame in frames)
        return VisualObservation(
            observation_id=f"vo_{uuid4().hex}",
            source_event_id=source_event_id,
            trigger_turn_id=trigger_turn_id,
            captured_at_ms=captured_at_ms,
            valid_until_ms=captured_at_ms + 10_000,
            frame_count=len(frames),
            **analysis.model_dump(),
        )

    @staticmethod
    def _extract_output_text(payload: object) -> str:
        if not isinstance(payload, dict):
            raise VisionProviderResponseError("invalid response envelope")
        direct = payload.get("output_text")
        if isinstance(direct, str) and direct:
            return direct
        output = payload.get("output")
        if not isinstance(output, list):
            raise VisionProviderResponseError("missing structured output")
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "output_text"
                    and isinstance(part.get("text"), str)
                ):
                    return str(part["text"])
        raise VisionProviderResponseError("missing output text")
