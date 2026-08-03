from typing import Protocol

from app.providers.ollama_text import OllamaTextProvider
from app.providers.protocols import AgentPlan
from app.rag.retriever import EvidenceBundle
from app.safety.models import AgentResponse, RiskAssessment, RiskLevel


class ReviewedRetriever(Protocol):
    async def retrieve(
        self,
        query: str,
        *,
        risk_level: RiskLevel,
        k: int,
    ) -> EvidenceBundle:
        raise NotImplementedError


class LocalAgentProvider:
    def __init__(
        self,
        *,
        text_provider: OllamaTextProvider,
        retriever: ReviewedRetriever,
        top_k: int = 3,
    ) -> None:
        self._text = text_provider
        self._retriever = retriever
        self._top_k = top_k

    async def load_context(
        self,
        transcript: str,
        visual_summary: str,
        risk: RiskAssessment,
    ) -> dict[str, object]:
        bundle = await self._retriever.retrieve(
            transcript,
            risk_level=risk.level,
            k=self._top_k,
        )
        return {
            "transcript": transcript,
            "visual_summary": visual_summary,
            "reviewed_evidence": [
                item.model_dump(mode="json") for item in bundle.items
            ]
            if bundle.has_sufficient_evidence
            else [],
            "has_sufficient_evidence": bundle.has_sufficient_evidence,
            "working_memory": [],
            "long_term_memory": [],
        }

    async def plan_reply(
        self,
        transcript: str,
        risk: RiskAssessment,
        context: dict[str, object],
        *,
        turn_id: str,
        cancel_token: str,
    ) -> AgentPlan:
        result = await self._text.respond(
            context,
            turn_id=turn_id,
            cancel_token=cancel_token,
            risk_level=risk.level,
        )
        return AgentPlan(
            response=result.response,
            provider_metrics=result.metrics.model_dump(),
        )

    async def plan_crisis(
        self,
        transcript: str,
        risk: RiskAssessment,
    ) -> AgentPlan:
        return AgentPlan(
            response=AgentResponse(
                spoken_text=(
                    "我很重视你刚才说的情况。请先远离可能伤害你的物品，"
                    "并尽快联系身边可信赖的人或当地紧急服务。"
                ),
                display_text=(
                    "我很重视你刚才说的情况。请先远离可能伤害你的物品，"
                    "并尽快联系身边可信赖的人或当地紧急服务。"
                ),
                support_mode="handoff",
                risk_level=risk.level,
                evidence_ids=[],
                visual_observation_ids=[],
                action_proposals=[
                    {
                        "tool": "request_clinician_handoff",
                        "status": "PROPOSED",
                    }
                ],
                memory_candidates=[],
                avatar_style="handoff_calm",
            ),
            provider_metrics={"provider": "deterministic_crisis"},
        )
