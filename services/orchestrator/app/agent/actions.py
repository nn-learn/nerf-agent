from pydantic import BaseModel, Field

from app.agent.models import AgentDirective
from app.safety.models import AgentResponse
from app.tools.models import ApprovalRole
from app.tools.policy import CapabilityRegistry, ToolPolicy


class CapabilityProposalAudit(BaseModel):
    proposed_count: int = Field(ge=0)
    accepted_count: int = Field(ge=0)
    blocked_count: int = Field(ge=0)
    accepted_capabilities: list[str] = Field(default_factory=list)
    blocked_reason_codes: list[str] = Field(default_factory=list)


class CapabilityProposalGate:
    """Treat model action proposals as untrusted data and project safe metadata."""

    def __init__(self, registry: CapabilityRegistry | None = None) -> None:
        self._policy = ToolPolicy(registry)

    def apply(
        self,
        response: AgentResponse,
        directive: AgentDirective,
    ) -> tuple[AgentResponse, CapabilityProposalAudit]:
        accepted: list[dict[str, object]] = []
        blocked: list[str] = []
        for proposal in response.action_proposals:
            tool = proposal.get("tool") if isinstance(proposal, dict) else None
            arguments = (
                proposal.get("arguments", {})
                if isinstance(proposal, dict)
                else {}
            )
            if not isinstance(tool, str) or not tool:
                blocked.append("MALFORMED_CAPABILITY_PROPOSAL")
                continue
            if tool not in directive.allowed_capabilities:
                blocked.append("CAPABILITY_NOT_IN_DIRECTIVE")
                continue
            if len(accepted) >= directive.max_tool_proposals:
                blocked.append("CAPABILITY_BUDGET_EXCEEDED")
                continue
            if not self._policy.is_allowed(
                tool,
                directive.risk_level,
                directive_allowlist=set(directive.allowed_capabilities),
            ):
                blocked.append("CAPABILITY_POLICY_DENIED")
                continue
            if not isinstance(arguments, dict):
                blocked.append("CAPABILITY_ARGUMENTS_INVALID")
                continue
            definition = self._policy.registry.get(tool)
            accepted.append(
                {
                    "tool": tool,
                    "arguments": arguments,
                    "status": (
                        "AWAITING_APPROVAL"
                        if definition.approval_role is not ApprovalRole.NONE
                        else "PROPOSED"
                    ),
                    "origin": definition.origin.value,
                    "impact": definition.impact.value,
                    "requires_approval": (
                        definition.approval_role is not ApprovalRole.NONE
                    ),
                    "approval_role": definition.approval_role.value,
                }
            )
        controlled = response.model_copy(update={"action_proposals": accepted})
        return controlled, CapabilityProposalAudit(
            proposed_count=len(response.action_proposals),
            accepted_count=len(accepted),
            blocked_count=len(blocked),
            accepted_capabilities=[str(item["tool"]) for item in accepted],
            blocked_reason_codes=blocked,
        )
