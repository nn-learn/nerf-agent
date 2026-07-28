from collections.abc import Callable
from uuid import uuid4

from app.safety.models import RiskLevel
from app.tools.models import ToolProposal, ToolResult, ToolState
from app.tools.policy import ToolPolicy

ToolHandler = Callable[[dict[str, object]], dict[str, object]]


class ToolPolicyError(ValueError):
    pass


class ToolGateway:
    def __init__(
        self,
        *,
        handlers: dict[str, ToolHandler] | None = None,
        policy: ToolPolicy | None = None,
    ) -> None:
        self._handlers = handlers or {}
        self._policy = policy or ToolPolicy()
        self._proposals: dict[str, ToolProposal] = {}
        self._by_idempotency_key: dict[str, str] = {}
        self._results: dict[str, ToolResult] = {}

    @classmethod
    def with_builtin_demo_tools(cls) -> "ToolGateway":
        def breathing(arguments: dict[str, object]) -> dict[str, object]:
            duration = arguments.get("duration_seconds", 60)
            if not isinstance(duration, int) or duration <= 0:
                raise ValueError("duration_seconds must be a positive integer")
            return {
                "exercise": "paced_breathing",
                "duration_seconds": duration,
            }

        def draft_contact(arguments: dict[str, object]) -> dict[str, object]:
            message = arguments.get("message")
            if not isinstance(message, str) or not message.strip():
                raise ValueError("message must be non-empty text")
            return {"draft": message, "sent": False}

        def local_action(arguments: dict[str, object]) -> dict[str, object]:
            return {"accepted": True, "arguments": arguments}

        return cls(
            handlers={
                "start_breathing_exercise": breathing,
                "show_reviewed_resource": local_action,
                "request_clinician_handoff": local_action,
                "draft_emergency_contact_message": draft_contact,
                "export_user_memories": local_action,
                "delete_user_memory": local_action,
            }
        )

    def propose(
        self,
        *,
        name: str,
        arguments: dict[str, object],
        risk_level: RiskLevel,
        idempotency_key: str | None = None,
    ) -> ToolProposal:
        key = idempotency_key or f"idem_{uuid4().hex}"
        existing_id = self._by_idempotency_key.get(key)
        if existing_id is not None:
            return self._proposals[existing_id]
        if not self._policy.is_known(name):
            raise ToolPolicyError(f"unknown tool: {name}")
        if not self._policy.is_allowed(name, risk_level):
            raise ToolPolicyError(
                f"tool {name} is not allowed at risk level {risk_level.value}"
            )
        state = (
            ToolState.AWAITING_APPROVAL
            if self._policy.requires_approval(name)
            else ToolState.PROPOSED
        )
        proposal = ToolProposal(
            tool_call_id=f"tool_{uuid4().hex}",
            name=name,
            arguments=arguments,
            risk_level=risk_level,
            idempotency_key=key,
            state=state,
        )
        self._proposals[proposal.tool_call_id] = proposal
        self._by_idempotency_key[key] = proposal.tool_call_id
        return proposal

    def approve(self, tool_call_id: str) -> None:
        proposal = self._proposals[tool_call_id]
        if proposal.state is not ToolState.AWAITING_APPROVAL:
            raise ToolPolicyError("tool call is not awaiting approval")
        self._proposals[tool_call_id] = proposal.model_copy(
            update={"state": ToolState.APPROVED}
        )

    def execute(self, tool_call_id: str) -> ToolResult:
        cached = self._results.get(tool_call_id)
        if cached is not None:
            return cached
        proposal = self._proposals[tool_call_id]
        if proposal.state is ToolState.AWAITING_APPROVAL:
            return ToolResult(
                tool_call_id=tool_call_id,
                state=ToolState.AWAITING_APPROVAL,
            )
        if proposal.state not in {ToolState.PROPOSED, ToolState.APPROVED}:
            raise ToolPolicyError(
                f"tool call cannot execute from state {proposal.state.value}"
            )

        running = proposal.model_copy(update={"state": ToolState.RUNNING})
        self._proposals[tool_call_id] = running
        try:
            output = self._handlers[running.name](running.arguments)
            result = ToolResult(
                tool_call_id=tool_call_id,
                state=ToolState.SUCCEEDED,
                output=output,
            )
        except Exception as error:
            result = ToolResult(
                tool_call_id=tool_call_id,
                state=ToolState.FAILED,
                error=str(error),
            )
        self._results[tool_call_id] = result
        self._proposals[tool_call_id] = running.model_copy(
            update={"state": result.state}
        )
        return result
