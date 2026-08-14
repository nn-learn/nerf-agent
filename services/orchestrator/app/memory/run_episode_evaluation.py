import argparse
import json
from pathlib import Path

from app.memory.episode_evaluation import MemoryEpisodeEvaluator, load_episode_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Memory V2.3 episode indexes")
    parser.add_argument("fixture", type=Path)
    args = parser.parse_args()
    report = MemoryEpisodeEvaluator().evaluate(load_episode_cases(args.fixture))
    print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))
    if report.failed_case_ids:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
