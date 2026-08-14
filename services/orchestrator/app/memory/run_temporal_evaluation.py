import argparse
import json
from pathlib import Path

from app.memory.temporal_evaluation import MemoryTemporalEvaluator, load_temporal_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Memory V2.2 temporal governance")
    parser.add_argument("fixture", type=Path)
    args = parser.parse_args()
    report = MemoryTemporalEvaluator().evaluate(load_temporal_cases(args.fixture))
    print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))
    if report.failed_case_ids:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
