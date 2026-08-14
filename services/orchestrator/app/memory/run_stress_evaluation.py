import argparse
import json

from app.memory.stress_evaluation import MemoryStressEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Memory V2.4.1 stress probes")
    parser.add_argument("--users", type=int, default=10)
    parser.add_argument("--latency-gate-ms", type=float, default=100.0)
    args = parser.parse_args()
    report = MemoryStressEvaluator(
        user_count=args.users,
        latency_gate_ms=args.latency_gate_ms,
    ).evaluate()
    print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
