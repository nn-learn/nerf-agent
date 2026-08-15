import argparse
import json
from pathlib import Path

from app.evals.agent_v3 import AgentV3Evaluator, load_agent_v3_cases
from app.evals.agent_v3_release import AgentV3ReleaseEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Agent V3 release gate")
    parser.add_argument("--cases", type=Path)
    arguments = parser.parse_args()
    root = Path(__file__).parents[4]
    cases_path = arguments.cases or root / "evals" / "agent_v3_cases.jsonl"
    evaluation = AgentV3Evaluator().evaluate(load_agent_v3_cases(cases_path))
    release = AgentV3ReleaseEvaluator().evaluate(evaluation)
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
