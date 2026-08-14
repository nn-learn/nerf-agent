import json
from pathlib import Path

from app.memory.evaluation import MemoryEvaluator, load_cases
from app.memory.extraction import RuleBasedMemoryExtractor
from app.memory.longitudinal_evaluation import run_longitudinal_evaluation
from app.memory.multisession_evaluation import (
    MultiSessionMemoryEvaluator,
    load_multisession_cases,
)
from app.memory.retrieval import GovernedMemoryRetriever

if __name__ == "__main__":
    cases_path = Path(__file__).parents[4] / "evals" / "memory_cases.jsonl"
    extraction_report = MemoryEvaluator(
        extractor=RuleBasedMemoryExtractor(),
    ).evaluate(load_cases(cases_path))
    longitudinal_report = run_longitudinal_evaluation()
    multisession_cases_path = (
        Path(__file__).parents[4] / "evals" / "memory_multisession_cases.jsonl"
    )
    multisession_report = MultiSessionMemoryEvaluator(
        strategy="lexical",
        retriever_factory=lambda repository: GovernedMemoryRetriever(repository),
    ).evaluate(load_multisession_cases(multisession_cases_path))
    print(
        json.dumps(
            {
                "extraction_policy": extraction_report.model_dump(),
                "longitudinal": longitudinal_report.model_dump(),
                "multisession_retrieval": multisession_report.model_dump(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
