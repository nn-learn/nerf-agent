import re
from datetime import date

from pydantic import BaseModel, Field, field_validator

from app.safety.models import RiskLevel


class KnowledgeDocument(BaseModel):
    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    version: str
    reviewer: str = Field(min_length=1)
    reviewed_at: date
    expires_at: date
    audience: list[str] = Field(min_length=1)
    allowed_risk_levels: list[RiskLevel] = Field(min_length=1)
    body: str = Field(min_length=1)

    @field_validator("reviewer")
    @classmethod
    def reviewer_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reviewer must not be blank")
        return value.strip()

    @field_validator("version")
    @classmethod
    def version_must_be_semantic(cls, value: str) -> str:
        if re.fullmatch(r"\d+\.\d+\.\d+", value) is None:
            raise ValueError("version must be a semantic version")
        return value


class KnowledgeChunk(BaseModel):
    chunk_id: str
    document_id: str
    document_version: str
    title: str
    text: str
    allowed_risk_levels: list[RiskLevel]


def load_reviewed_documents(
    documents: list[KnowledgeDocument],
    *,
    as_of: date,
) -> list[KnowledgeDocument]:
    return [
        document
        for document in documents
        if document.reviewed_at <= as_of < document.expires_at
    ]


def chunk_document(document: KnowledgeDocument) -> list[KnowledgeChunk]:
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", document.body)
        if paragraph.strip()
    ]
    return [
        KnowledgeChunk(
            chunk_id=f"{document.document_id}:{index}",
            document_id=document.document_id,
            document_version=document.version,
            title=document.title,
            text=paragraph,
            allowed_risk_levels=document.allowed_risk_levels,
        )
        for index, paragraph in enumerate(paragraphs)
    ]
