from importlib import import_module
from typing import Any, Protocol

from app.rag.retriever import EmbeddingDimensionError


class SentenceEncoder(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool,
    ) -> Any: ...


class BgeM3EmbeddingProvider:
    dimensions = 1024

    def __init__(self, encoder: SentenceEncoder | None = None) -> None:
        if encoder is None:
            sentence_transformers = import_module("sentence_transformers")
            encoder = sentence_transformers.SentenceTransformer(
                "BAAI/bge-m3",
                device="cpu",
            )
        self._encoder = encoder

    def embed(self, text: str) -> list[float]:
        encoded = self._encoder.encode(
            [text],
            normalize_embeddings=True,
        )
        first = encoded[0]
        vector = first.tolist() if hasattr(first, "tolist") else list(first)
        if len(vector) != self.dimensions:
            raise EmbeddingDimensionError(
                f"expected {self.dimensions} embedding dimensions, got {len(vector)}"
            )
        return [float(value) for value in vector]
