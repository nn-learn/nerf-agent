from collections.abc import Iterable

from app.safety.models import RiskLevel
from app.tools.models import (
    ApprovalRole,
    CapabilityDefinition,
    CapabilityImpact,
    CapabilityOrigin,
)


class CapabilityRegistrationError(ValueError):
    pass


class CapabilityRegistry:
    """Host-owned capability metadata; model/MCP descriptions grant no authority."""

    _high_impact = {
        CapabilityImpact.SENSITIVE_DATA_EXPORT,
        CapabilityImpact.DATA_DELETION,
        CapabilityImpact.EXTERNAL_COMMUNICATION,
    }

    def __init__(
        self,
        definitions: Iterable[CapabilityDefinition] | None = None,
        *,
        trusted_mcp_servers: Iterable[str] = (),
    ) -> None:
        self._definitions: dict[str, CapabilityDefinition] = {}
        self._trusted_mcp_servers = set(trusted_mcp_servers)
        initial = self.builtin_definitions() if definitions is None else definitions
        for definition in initial:
            self.register(definition)

    def register(self, definition: CapabilityDefinition) -> None:
        if definition.name in self._definitions:
            raise CapabilityRegistrationError(
                f"duplicate capability: {definition.name}"
            )
        if not definition.trusted:
            raise CapabilityRegistrationError("untrusted capability cannot be registered")
        if (
            definition.origin is CapabilityOrigin.MCP
            and definition.source_id not in self._trusted_mcp_servers
        ):
            raise CapabilityRegistrationError(
                f"MCP server is not host-trusted: {definition.source_id}"
            )
        if (
            definition.impact in self._high_impact
            and definition.approval_role is ApprovalRole.NONE
        ):
            raise CapabilityRegistrationError(
                "high-impact capability requires an approval role"
            )
        self._definitions[definition.name] = definition.model_copy(deep=True)

    def get(self, name: str) -> CapabilityDefinition:
        try:
            return self._definitions[name].model_copy(deep=True)
        except KeyError as error:
            raise CapabilityRegistrationError(f"unknown capability: {name}") from error

    def contains(self, name: str) -> bool:
        return name in self._definitions

    @staticmethod
    def builtin_definitions() -> list[CapabilityDefinition]:
        normal = [RiskLevel.GREEN, RiskLevel.AMBER]
        crisis = [RiskLevel.RED, RiskLevel.EMERGENCY]
        return [
            CapabilityDefinition(
                name="start_breathing_exercise",
                origin=CapabilityOrigin.LOCAL,
                source_id="psyavatar.local",
                trusted=True,
                impact=CapabilityImpact.LOCAL_REVERSIBLE,
                approval_role=ApprovalRole.NONE,
                allowed_risk_levels=normal,
            ),
            CapabilityDefinition(
                name="show_reviewed_resource",
                origin=CapabilityOrigin.LOCAL,
                source_id="psyavatar.local",
                trusted=True,
                impact=CapabilityImpact.READ_ONLY,
                approval_role=ApprovalRole.NONE,
                allowed_risk_levels=[*normal, *crisis],
                max_per_turn=2,
            ),
            CapabilityDefinition(
                name="request_clinician_handoff",
                origin=CapabilityOrigin.LOCAL,
                source_id="psyavatar.local",
                trusted=True,
                impact=CapabilityImpact.CLINICAL_ESCALATION,
                approval_role=ApprovalRole.NONE,
                allowed_risk_levels=[*normal, *crisis],
            ),
            CapabilityDefinition(
                name="draft_emergency_contact_message",
                origin=CapabilityOrigin.LOCAL,
                source_id="psyavatar.local",
                trusted=True,
                impact=CapabilityImpact.EXTERNAL_COMMUNICATION,
                approval_role=ApprovalRole.USER,
                allowed_risk_levels=crisis,
            ),
            CapabilityDefinition(
                name="export_user_memories",
                origin=CapabilityOrigin.LOCAL,
                source_id="psyavatar.local",
                trusted=True,
                impact=CapabilityImpact.SENSITIVE_DATA_EXPORT,
                approval_role=ApprovalRole.USER,
                allowed_risk_levels=normal,
            ),
            CapabilityDefinition(
                name="delete_user_memory",
                origin=CapabilityOrigin.LOCAL,
                source_id="psyavatar.local",
                trusted=True,
                impact=CapabilityImpact.DATA_DELETION,
                approval_role=ApprovalRole.USER,
                allowed_risk_levels=normal,
            ),
        ]


class ToolPolicy:
    def __init__(self, registry: CapabilityRegistry | None = None) -> None:
        self.registry = registry or CapabilityRegistry()

    def is_known(self, name: str) -> bool:
        return self.registry.contains(name)

    def requires_approval(self, name: str) -> bool:
        return self.registry.get(name).approval_role is not ApprovalRole.NONE

    def is_allowed(
        self,
        name: str,
        risk_level: RiskLevel,
        *,
        directive_allowlist: set[str] | None = None,
    ) -> bool:
        if not self.registry.contains(name):
            return False
        definition = self.registry.get(name)
        return (
            definition.trusted
            and risk_level in definition.allowed_risk_levels
            and (
                directive_allowlist is None
                or definition.name in directive_allowlist
            )
        )
