import argparse
import json
from pathlib import Path

from app.memory.multisession_evaluation import (
    MemoryABReport,
    MemoryQualityGate,
    MultiSessionMemoryEvaluator,
    OllamaMemoryAnswerGenerator,
    load_multisession_cases,
)
from app.memory.repository import MemoryRepository
from app.memory.retrieval import (
    GovernedEmbeddingMemoryRetriever,
    GovernedMemoryRetriever,
)
from app.rag.bge import BgeM3EmbeddingProvider


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Memory V1.4 multi-session evaluation")
    parser.add_argument(
        "--with-bge",
        action="store_true",
        help="load local BAAI/bge-m3 and run the semantic A/B branch",
    )
    parser.add_argument(
        "--with-ollama-answers",
        action="store_true",
        help="ask local Ollama to generate answers for adherence evaluation",
    )
    parser.add_argument("--ollama-model", default="qwen3.6:latest")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument(
        "--bge-min-relevance",
        type=float,
        default=0.40,
        help="cosine threshold for semantic memory candidates (default: 0.40)",
    )
    args = parser.parse_args()

    cases_path = Path(__file__).parents[4] / "evals" / "memory_multisession_cases.jsonl"
    cases = load_multisession_cases(cases_path)
    answer_generator = (
        OllamaMemoryAnswerGenerator(
            model=args.ollama_model,
            base_url=args.ollama_base_url,
        )
        if args.with_ollama_answers
        else None
    )
    try:
        lexical = MultiSessionMemoryEvaluator(
            strategy="lexical",
            retriever_factory=lambda repository: GovernedMemoryRetriever(repository),
            answer_generator=answer_generator if not args.with_bge else None,
        ).evaluate(cases)
        payload: dict[str, object] = {"lexical": lexical.model_dump()}
        gate = MemoryQualityGate()
        payload["lexical_gate_violations"] = gate.evaluate(
            lexical,
            require_answers=args.with_ollama_answers and not args.with_bge,
        )
        if args.with_bge:
            provider = BgeM3EmbeddingProvider()

            def semantic_factory(
                repository: MemoryRepository,
            ) -> GovernedEmbeddingMemoryRetriever:
                return GovernedEmbeddingMemoryRetriever(
                    repository,
                    embedding_provider=provider,
                    min_relevance=args.bge_min_relevance,
                )

            semantic = MultiSessionMemoryEvaluator(
                strategy=f"bge-m3@{args.bge_min_relevance:.2f}",
                retriever_factory=semantic_factory,
                answer_generator=answer_generator,
            ).evaluate(cases)
            payload["ab"] = MemoryABReport(
                lexical=lexical,
                semantic=semantic,
                recall_at_5_delta=round(
                    semantic.retrieval_recall_at_5 - lexical.retrieval_recall_at_5,
                    6,
                ),
                ndcg_at_5_delta=round(semantic.ndcg_at_5 - lexical.ndcg_at_5, 6),
                forbidden_retrieval_rate_delta=round(
                    semantic.forbidden_retrieval_rate
                    - lexical.forbidden_retrieval_rate,
                    6,
                ),
            ).model_dump()
            payload["semantic_gate_violations"] = gate.evaluate(
                semantic,
                require_answers=args.with_ollama_answers,
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        if answer_generator is not None:
            answer_generator.close()


if __name__ == "__main__":
    main()
