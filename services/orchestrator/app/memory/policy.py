from app.memory.models import MemoryCandidate, MemoryDecision, MemoryKind


class MemoryPolicy:
    def evaluate(
        self,
        candidate: MemoryCandidate,
        *,
        consent_granted: bool,
    ) -> MemoryDecision:
        if candidate.kind is MemoryKind.SAFETY:
            return MemoryDecision.REJECT
        if candidate.source == "visual" and candidate.contains_sensitive_content:
            return MemoryDecision.REJECT
        if not consent_granted:
            return MemoryDecision.REQUEST_CONSENT
        return MemoryDecision.ACCEPT
