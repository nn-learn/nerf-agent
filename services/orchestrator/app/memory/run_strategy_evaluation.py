import argparse
import json
from pathlib import Path

from app.memory.evaluation_protocol import (
    DatasetStatus,
    apply_adjudicated_labels,
    audit_annotations,
    load_adjudicated_labels,
    load_annotations,
    validate_manifest_cases,
)
from app.memory.multisession_evaluation import (
    OllamaMemoryAnswerGenerator,
    load_multisession_cases,
)
from app.memory.retrieval import CachedMemoryEmbeddingProvider
from app.memory.strategy_evaluation import MemoryStrategyEvaluator, load_v24_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Memory V2.4 strategy matrix")
    parser.add_argument("--with-bge", action="store_true")
    parser.add_argument("--with-ollama-answers", action="store_true")
    parser.add_argument("--answer-arm", default="episode-freshness-v2.3")
    parser.add_argument("--ollama-model", default="qwen3.6:latest")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--adjudicated-labels", type=Path)
    args = parser.parse_args()
    root = Path(__file__).parents[4]
    cases = load_multisession_cases(root / "evals" / "memory_v24_cases.jsonl")
    manifest = load_v24_manifest(root / "evals" / "memory_v24_manifest.json")
    validate_manifest_cases(manifest, cases)
    annotation_audit = None
    if args.annotations is not None:
        annotation_audit = audit_annotations(
            load_annotations(args.annotations),
            minimum_annotators=manifest.minimum_annotators,
            expected_dataset_version=manifest.dataset_version,
        )
    if args.adjudicated_labels is not None:
        cases = apply_adjudicated_labels(
            cases,
            load_adjudicated_labels(args.adjudicated_labels),
        )
    if (
        manifest.dataset_status is DatasetStatus.INDEPENDENTLY_ANNOTATED
        and args.adjudicated_labels is None
    ):
        raise ValueError(
            "INDEPENDENTLY_ANNOTATED V2.4 data requires adjudicated labels"
        )
    provider = None
    if args.with_bge:
        from app.rag.bge import BgeM3EmbeddingProvider

        provider = CachedMemoryEmbeddingProvider(BgeM3EmbeddingProvider())
    answer_generator = (
        OllamaMemoryAnswerGenerator(
            model=args.ollama_model,
            base_url=args.ollama_base_url,
        )
        if args.with_ollama_answers
        else None
    )
    try:
        report = MemoryStrategyEvaluator(
            embedding_provider=provider,
            answer_generator=answer_generator,
            answer_arm=args.answer_arm,
        ).evaluate(
            cases,
            manifest=manifest,
            annotation_audit=annotation_audit,
        )
    finally:
        if answer_generator is not None:
            answer_generator.close()
    print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
