import argparse
import json
from pathlib import Path

from app.memory.evaluation_protocol import DatasetStatus
from app.memory.human_review import (
    audit_human_reviews,
    build_human_review_pack,
    load_human_review_candidates,
    load_human_review_submissions,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or audit a deidentified Memory V2.5 human-review pack",
    )
    parser.add_argument("candidates", type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument(
        "--dataset-status",
        choices=[item.value for item in DatasetStatus],
        default=DatasetStatus.ENGINEERING_FIXTURE.value,
    )
    parser.add_argument("--data-classification", required=True)
    parser.add_argument("--id-salt", required=True)
    parser.add_argument("--deidentified", action="store_true")
    parser.add_argument("--submissions", type=Path)
    args = parser.parse_args()
    pack = build_human_review_pack(
        load_human_review_candidates(args.candidates),
        dataset_id=args.dataset_id,
        dataset_status=DatasetStatus(args.dataset_status),
        data_classification=args.data_classification,
        source_deidentified=args.deidentified,
        id_salt=args.id_salt,
    )
    payload: dict[str, object] = {"pack": pack.model_dump(mode="json")}
    if args.submissions is not None:
        payload["audit"] = audit_human_reviews(
            pack,
            load_human_review_submissions(args.submissions),
        ).model_dump(mode="json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
