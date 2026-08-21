import argparse
import json
from pathlib import Path

from app.memory.v5_evaluation import MemoryV5Evaluator, load_memory_v5_cases
from app.memory.v5_release_evaluation import MemoryV5ReleaseEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Memory V5 release gate")
    parser.add_argument("--cases", type=Path)
    arguments = parser.parse_args()
    root = Path(__file__).parents[4]
    cases_path = arguments.cases or root / "evals" / "memory_v5_cases.jsonl"
    evaluation = MemoryV5Evaluator().evaluate(load_memory_v5_cases(cases_path))
    release = MemoryV5ReleaseEvaluator().evaluate(evaluation)
    print(
        json.dumps(
            {
                "evaluation": evaluation.model_dump(mode="json"),
                "release": release.model_dump(mode="json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
