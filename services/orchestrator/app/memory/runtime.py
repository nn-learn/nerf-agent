import httpx

from app.memory.consolidation import MemoryClaimNormalizer, MemoryClaimNormalizerPort
from app.memory.extraction import MemoryExtractor, RuleBasedMemoryExtractor
from app.memory.ollama_extractor import OllamaMemoryExtractor
from app.memory.ollama_normalizer import OllamaMemoryClaimNormalizer
from app.settings import Settings


def build_memory_extractor(
    settings: Settings,
    *,
    client: httpx.Client | None = None,
) -> MemoryExtractor:
    if settings.memory_extractor_mode == "rule":
        return RuleBasedMemoryExtractor()
    return OllamaMemoryExtractor(
        model=settings.text_model,
        base_url=settings.ollama_base_url,
        keep_alive=settings.ollama_keep_alive,
        timeout_seconds=settings.ollama_timeout_seconds,
        num_ctx=settings.memory_ollama_num_ctx,
        num_predict=settings.memory_ollama_num_predict,
        cache_entries=settings.memory_extraction_cache_entries,
        use_rule_fallback=settings.memory_rule_fallback,
        client=client,
    )


def build_memory_claim_normalizer(
    settings: Settings,
    *,
    client: httpx.Client | None = None,
) -> MemoryClaimNormalizerPort:
    if settings.memory_claim_normalizer_mode == "rule":
        return MemoryClaimNormalizer()
    return OllamaMemoryClaimNormalizer(
        model=settings.text_model,
        base_url=settings.ollama_base_url,
        keep_alive=settings.ollama_keep_alive,
        timeout_seconds=settings.memory_claim_ollama_timeout_seconds,
        num_ctx=settings.memory_claim_ollama_num_ctx,
        num_predict=settings.memory_claim_ollama_num_predict,
        cache_entries=settings.memory_claim_ollama_cache_entries,
        max_claims_per_batch=settings.memory_claim_ollama_batch_claims,
        client=client,
    )
