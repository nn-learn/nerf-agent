import argparse
import json
import tempfile
from pathlib import Path

from app.evals.privacy import audit_memory_privacy
from app.memory.multisession_evaluation import (
    OllamaMemoryAnswerGenerator,
    load_multisession_cases,
)
from app.memory.release_evaluation import MemoryV2ReleaseEvaluator
from app.memory.repository import MemoryRepository
from app.memory.retrieval import CachedMemoryEmbeddingProvider
from app.memory.strategy_evaluation import MemoryStrategyEvaluator, load_v24_manifest
from app.memory.stress_evaluation import MemoryStressEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Memory V2 release gate")
    parser.add_argument("--with-bge", action="store_true")
    parser.add_argument("--with-ollama-answers", action="store_true")
    parser.add_argument("--answer-arm")
    parser.add_argument("--ollama-model", default="qwen3.6:latest")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--database", type=Path)
    args = parser.parse_args()
    root = Path(__file__).parents[4]
    cases = load_multisession_cases(root / "evals" / "memory_v24_cases.jsonl")
    manifest = load_v24_manifest(root / "evals" / "memory_v24_manifest.json")
    embedding_provider = None
    if args.with_bge:
        from app.rag.bge import BgeM3EmbeddingProvider

        embedding_provider = CachedMemoryEmbeddingProvider(BgeM3EmbeddingProvider())
    answer_arm = args.answer_arm or (
        "bge-m3-hybrid" if args.with_bge else "episode-freshness-v2.3"
    )
    answer_generator = (
        OllamaMemoryAnswerGenerator(
            model=args.ollama_model,
            base_url=args.ollama_base_url,
        )
        if args.with_ollama_answers
        else None
    )
    try:
        strategy_report = MemoryStrategyEvaluator(
            embedding_provider=embedding_provider,
            answer_generator=answer_generator,
            answer_arm=answer_arm,
        ).evaluate(cases, manifest=manifest)
    finally:
        if answer_generator is not None:
            answer_generator.close()
    stress_report = MemoryStressEvaluator().evaluate()
    if args.database is not None:
        privacy_report = audit_memory_privacy(args.database)
        privacy_scope = "DEPLOYMENT_DATABASE"
    else:
        with tempfile.TemporaryDirectory(
            prefix="psyavatar-v2-release-",
        ) as directory:
            database_path = Path(directory) / "memory.sqlite3"
            MemoryRepository(database_path).initialize()
            privacy_report = audit_memory_privacy(database_path)
        privacy_scope = "SYNTHETIC_SCHEMA"
    report = MemoryV2ReleaseEvaluator().evaluate(
        strategy_report=strategy_report,
        stress_report=stress_report,
        privacy_report=privacy_report,
        privacy_audit_scope=privacy_scope,
        selected_strategy=answer_arm,
    )
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
