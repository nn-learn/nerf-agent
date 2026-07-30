import asyncio
import json
from dataclasses import dataclass

import httpx
from pydantic import BaseModel, Field, ValidationError

from app.contracts.session import CancellationRegistry
from app.providers.qwen_text import (
    SUPPORT_POLICY,
    TextProviderCancelled,
    TextProviderResponseError,
)
from app.safety.models import AgentResponse, RiskLevel
from app.safety.output_guard import OutputGuard


@dataclass(frozen=True, slots=True)
class OllamaReadiness:
    ready: bool
    version: str | None
    model: str
    reason: str | None = None


class OllamaMetrics(BaseModel):
    total_duration_ns: int = Field(ge=0)
    load_duration_ns: int = Field(ge=0)
    prompt_eval_count: int = Field(ge=0)
    eval_count: int = Field(ge=0)


@dataclass(frozen=True, slots=True)
class OllamaTextResult:
    response: AgentResponse
    metrics: OllamaMetrics


class OllamaTextProvider:
    def __init__(
        self,
        *,
        registry: CancellationRegistry,
        model: str = "qwen3.6:latest",
        base_url: str = "http://127.0.0.1:11434",
        keep_alive: str = "30m",
        timeout_seconds: float = 180,
        num_ctx: int = 4096,
        num_predict: int = 320,
        client: httpx.AsyncClient | None = None,
        output_guard: OutputGuard | None = None,
        generation_lock: asyncio.Lock | None = None,
    ) -> None:
        self._registry = registry
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._keep_alive = keep_alive
        self._num_ctx = num_ctx
        self._num_predict = num_predict
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)
        self._owns_client = client is None
        self._guard = output_guard or OutputGuard()
        self._generation_lock = generation_lock or asyncio.Lock()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def respond(
        self,
        context: dict[str, object],
        *,
        turn_id: str,
        cancel_token: str,
        risk_level: RiskLevel,
    ) -> OllamaTextResult:
        if not await self._registry.is_current(turn_id, cancel_token):
            raise TextProviderCancelled(cancel_token)

        request_body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SUPPORT_POLICY},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "risk_level": risk_level.value,
                            "context": context,
                            "response_schema": AgentResponse.model_json_schema(),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "stream": False,
            "think": False,
            "format": AgentResponse.model_json_schema(),
            "keep_alive": self._keep_alive,
            "options": {
                "temperature": 0.2,
                "num_ctx": self._num_ctx,
                "num_predict": self._num_predict,
            },
        }

        async with self._generation_lock:
            if not await self._registry.is_current(turn_id, cancel_token):
                raise TextProviderCancelled(cancel_token)
            try:
                response = await self._client.post(
                    f"{self._base_url}/api/chat",
                    json=request_body,
                )
                response.raise_for_status()
                payload = response.json()
                if not await self._registry.is_current(turn_id, cancel_token):
                    raise TextProviderCancelled(cancel_token)
                content = self._extract_content(payload)
                agent_response = AgentResponse.model_validate_json(content)
            except TextProviderCancelled:
                raise
            except (httpx.HTTPError, ValidationError, ValueError, TypeError) as error:
                raise TextProviderResponseError(
                    "text provider returned an invalid or unavailable response"
                ) from error

            if agent_response.risk_level is not risk_level:
                raise TextProviderResponseError(
                    "text provider changed the deterministic risk level"
                )
            reviewed_evidence = context.get("reviewed_evidence", [])
            if not isinstance(reviewed_evidence, list):
                reviewed_evidence = []
            allowed_ids = {
                str(item["chunk_id"])
                for item in reviewed_evidence
                if isinstance(item, dict) and "chunk_id" in item
            }
            unknown_ids = set(agent_response.evidence_ids) - allowed_ids
            if unknown_ids:
                raise TextProviderResponseError(
                    "text provider cited evidence outside the reviewed turn bundle"
                )
            if agent_response.evidence_ids and not context.get(
                "has_sufficient_evidence", False
            ):
                raise TextProviderResponseError(
                    "text provider cited an insufficient evidence bundle"
                )
            try:
                guarded = self._guard.validate(agent_response)
                return OllamaTextResult(
                    response=guarded,
                    metrics=OllamaMetrics(
                        total_duration_ns=payload.get("total_duration", 0),
                        load_duration_ns=payload.get("load_duration", 0),
                        prompt_eval_count=payload.get("prompt_eval_count", 0),
                        eval_count=payload.get("eval_count", 0),
                    ),
                )
            except (ValidationError, ValueError, TypeError) as error:
                raise TextProviderResponseError(
                    "text provider returned an invalid or unavailable response"
                ) from error

    async def readiness(self) -> OllamaReadiness:
        try:
            version_response = await self._client.get(
                f"{self._base_url}/api/version"
            )
            version_response.raise_for_status()
            version = self._extract_version(version_response.json())

            tags_response = await self._client.get(f"{self._base_url}/api/tags")
            tags_response.raise_for_status()
            model_names = self._extract_model_names(tags_response.json())
        except (httpx.HTTPError, ValueError, TypeError) as error:
            raise TextProviderResponseError(
                "text provider readiness check failed"
            ) from error

        if self._model not in model_names:
            return OllamaReadiness(
                ready=False,
                version=version,
                model=self._model,
                reason="model_not_found",
            )
        return OllamaReadiness(
            ready=True,
            version=version,
            model=self._model,
        )

    async def prewarm(self) -> None:
        try:
            response = await self._client.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": "",
                    "stream": False,
                    "keep_alive": self._keep_alive,
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise TextProviderResponseError(
                "text provider prewarm failed"
            ) from error

    @staticmethod
    def _extract_content(payload: object) -> str:
        if not isinstance(payload, dict):
            raise TextProviderResponseError("invalid response envelope")
        message = payload.get("message")
        if not isinstance(message, dict):
            raise TextProviderResponseError("missing response message")
        content = message.get("content")
        if not isinstance(content, str) or not content:
            raise TextProviderResponseError("missing response content")
        return content

    @staticmethod
    def _extract_version(payload: object) -> str:
        if not isinstance(payload, dict):
            raise TextProviderResponseError("invalid version envelope")
        version = payload.get("version")
        if not isinstance(version, str) or not version:
            raise TextProviderResponseError("missing Ollama version")
        return version

    @staticmethod
    def _extract_model_names(payload: object) -> set[str]:
        if not isinstance(payload, dict):
            raise TextProviderResponseError("invalid tags envelope")
        models = payload.get("models")
        if not isinstance(models, list):
            raise TextProviderResponseError("missing model tags")
        names: set[str] = set()
        for model in models:
            if not isinstance(model, dict):
                raise TextProviderResponseError("invalid model tag")
            name = model.get("name")
            if not isinstance(name, str) or not name:
                raise TextProviderResponseError("missing model name")
            names.add(name)
        return names
