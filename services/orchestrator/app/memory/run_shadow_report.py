import argparse
import json
from pathlib import Path

from app.memory.shadow import MemoryShadowRepository


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export aggregate-only V1.6 memory shadow metrics.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("runtime/events.sqlite3"),
        help="Orchestrator SQLite database.",
    )
    parser.add_argument(
        "--user-id",
        required=True,
        help="Pseudonymous memory subject ID; never printed in the report.",
    )
    parser.add_argument(
        "--minimum-reliable-runs",
        type=int,
        default=100,
    )
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    repository = MemoryShadowRepository(
        arguments.database,
        minimum_reliable_runs=arguments.minimum_reliable_runs,
    )
    report = repository.aggregate(arguments.user_id)
    print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
