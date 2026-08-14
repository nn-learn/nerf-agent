import argparse
import json
from pathlib import Path

from app.memory.consolidation_evaluation import (
    MemoryConsolidationEvaluator,
    load_consolidation_cases,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate governed Memory V2 profiles")
    parser.add_argument("fixture", type=Path)
    args = parser.parse_args()
    report = MemoryConsolidationEvaluator().evaluate(
        load_consolidation_cases(args.fixture)
    )
    print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))
    if report.failed_case_ids:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
