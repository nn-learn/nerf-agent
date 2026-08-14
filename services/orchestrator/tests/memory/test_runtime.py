import httpx

from app.memory.consolidation import MemoryClaimNormalizer
from app.memory.extraction import RuleBasedMemoryExtractor
from app.memory.ollama_extractor import OllamaMemoryExtractor
from app.memory.ollama_normalizer import OllamaMemoryClaimNormalizer
from app.memory.runtime import build_memory_claim_normalizer, build_memory_extractor
from app.settings import Settings


def test_memory_runtime_keeps_rule_extractor_as_safe_default() -> None:
    extractor = build_memory_extractor(Settings(memory_extractor_mode="rule"))

    assert isinstance(extractor, RuleBasedMemoryExtractor)


def test_memory_runtime_builds_configured_ollama_extractor() -> None:
    extractor = build_memory_extractor(
        Settings(
            memory_extractor_mode="ollama",
            text_model="qwen3.6:latest",
            memory_rule_fallback=False,
        ),
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={"message": {"content": '{"candidates": []}'}},
                )
            )
        ),
    )

    assert isinstance(extractor, OllamaMemoryExtractor)


def test_memory_claim_normalizer_keeps_rule_mode_as_safe_default() -> None:
    normalizer = build_memory_claim_normalizer(Settings())

    assert isinstance(normalizer, MemoryClaimNormalizer)


def test_memory_runtime_builds_optional_ollama_claim_normalizer() -> None:
    normalizer = build_memory_claim_normalizer(
        Settings(memory_claim_normalizer_mode="ollama"),
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={
                        "message": {
                            "content": '{"clusters":[{"member_memory_ids":["memory_1"]}]}'
                        }
                    },
                )
            )
        ),
    )

    assert isinstance(normalizer, OllamaMemoryClaimNormalizer)
