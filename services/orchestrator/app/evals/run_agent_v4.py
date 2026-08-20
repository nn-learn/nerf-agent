import argparse
import json
from pathlib import Path

from app.evals.agent_v4 import AgentV4Evaluator, load_agent_v4_scenarios
from app.evals.agent_v4_release import AgentV4ReleaseEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Agent V4 release gate")
    parser.add_argument("--scenarios", type=Path)
    arguments = parser.parse_args()
    root = Path(__file__).parents[4]
    scenarios_path = (
        arguments.scenarios or root / "evals" / "agent_v4_scenarios.jsonl"
    )
    evaluation = AgentV4Evaluator().evaluate(
        load_agent_v4_scenarios(scenarios_path)
    )
    release = AgentV4ReleaseEvaluator().evaluate(evaluation)
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
