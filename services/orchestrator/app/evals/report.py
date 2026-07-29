import argparse
import json
from dataclasses import asdict
from pathlib import Path

from app.evals.risk import evaluate_risk_file
from app.evals.vision import evaluate_vision_safety_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print deterministic PsyAvatar Care acceptance metrics."
    )
    parser.add_argument("--risk", type=Path, required=True)
    parser.add_argument("--vision", type=Path, required=True)
    arguments = parser.parse_args()
    risk = evaluate_risk_file(arguments.risk)
    vision = evaluate_vision_safety_file(arguments.vision)
    print(
        json.dumps(
            {
                "risk": {
                    **asdict(risk),
                    "accuracy": risk.accuracy,
                    "high_risk_recall": risk.high_risk_recall,
                },
                "vision": asdict(vision),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
