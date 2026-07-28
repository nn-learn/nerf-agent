from app.rag.retriever import EvidenceBundle, EvidenceItem


class MissingEvidenceError(ValueError):
    pass


class EvidencePolicy:
    def validate_claim_ids(
        self,
        bundle: EvidenceBundle,
        claim_ids: list[str],
    ) -> list[EvidenceItem]:
        if not bundle.has_sufficient_evidence:
            raise MissingEvidenceError("evidence bundle is below the confidence threshold")

        by_id = {item.chunk_id: item for item in bundle.items}
        missing = [claim_id for claim_id in claim_ids if claim_id not in by_id]
        if missing:
            raise MissingEvidenceError(
                f"claim references evidence absent from this turn: {missing[0]}"
            )
        return [by_id[claim_id] for claim_id in claim_ids]

