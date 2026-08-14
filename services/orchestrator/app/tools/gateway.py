import hashlib
import json
from collections.abc import Callable
from uuid import uuid4

from app.safety.models import RiskLevel
from app.tools.models import ApprovalRole, ToolProposal, ToolResult, ToolState
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
        self._turn_call_ids: dict[str, list[str]] = {}

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
        turn_id: str = "legacy_turn",
        directive_allowlist: set[str] | None = None,
        max_tool_calls: int = 3,
        idempotency_key: str | None = None,
    ) -> ToolProposal:
        if max_tool_calls < 0 or max_tool_calls > 3:
            raise ToolPolicyError("max_tool_calls must be between zero and three")
        argument_digest = self._argument_digest(arguments)
        key = idempotency_key or "idem_" + hashlib.sha256(
            f"{turn_id}:{name}:{argument_digest}".encode()
        ).hexdigest()[:32]
        existing_id = self._by_idempotency_key.get(key)
        if existing_id is not None:
            existing = self._proposals[existing_id]
            if (
                existing.name != name
                or existing.turn_id != turn_id
                or existing.argument_digest != argument_digest
            ):
                raise ToolPolicyError("idempotency key is bound to another proposal")
            return existing
        if not self._policy.is_known(name):
            raise ToolPolicyError(f"unknown tool: {name}")
        if not self._policy.is_allowed(
            name,
            risk_level,
            directive_allowlist=directive_allowlist,
        ):
            raise ToolPolicyError(
                f"tool {name} is not allowed at risk level {risk_level.value}"
            )
        definition = self._policy.registry.get(name)
        turn_calls = self._turn_call_ids.setdefault(turn_id, [])
        if len(turn_calls) >= max_tool_calls:
            raise ToolPolicyError("per-turn tool budget exhausted")
        same_capability_count = sum(
            self._proposals[call_id].name == name for call_id in turn_calls
        )
        if same_capability_count >= definition.max_per_turn:
            raise ToolPolicyError("per-capability turn budget exhausted")
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
            turn_id=turn_id,
            capability_origin=definition.origin,
            capability_impact=definition.impact,
            approval_role=definition.approval_role,
            argument_digest=argument_digest,
            idempotency_key=key,
            state=state,
        )
        self._proposals[proposal.tool_call_id] = proposal
        self._by_idempotency_key[key] = proposal.tool_call_id
        turn_calls.append(proposal.tool_call_id)
        return proposal

    def approve(
        self,
        tool_call_id: str,
        *,
        actor_role: ApprovalRole = ApprovalRole.USER,
    ) -> None:
        proposal = self._proposals[tool_call_id]
        if proposal.state is not ToolState.AWAITING_APPROVAL:
            raise ToolPolicyError("tool call is not awaiting approval")
        if actor_role is not proposal.approval_role:
            raise ToolPolicyError(
                f"tool call requires {proposal.approval_role.value} approval"
            )
        self._proposals[tool_call_id] = proposal.model_copy(
            update={
                "state": ToolState.APPROVED,
                "approved_by_role": actor_role,
            }
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

    @staticmethod
    def _argument_digest(arguments: dict[str, object]) -> str:
        try:
            encoded = json.dumps(
                arguments,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise ToolPolicyError("tool arguments must be JSON-serializable") from error
        return hashlib.sha256(encoded).hexdigest()
