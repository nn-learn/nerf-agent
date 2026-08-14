import argparse
import json
from dataclasses import asdict
from pathlib import Path

from app.evals.privacy import audit_memory_privacy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit PsyAvatar Memory V2 privacy invariants",
    )
    parser.add_argument("database", type=Path)
    parser.add_argument("--as-of-ms", type=int)
    args = parser.parse_args()
    report = audit_memory_privacy(args.database, as_of_ms=args.as_of_ms)
    payload = asdict(report)
    payload.update(
        {
            "owner_mismatch_count": report.owner_mismatch_count,
            "orphan_reference_count": report.orphan_reference_count,
            "passed": report.passed,
        }
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
