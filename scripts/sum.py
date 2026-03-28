import argparse
import json
from collections import defaultdict


DEFAULT_FILE_PATH = (
    "/data1/yfl_data/DynaHOI/scripts/evaluation_results/"
    "gr00t_finetune.jsonl"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate count and score sum by task_type from a JSONL file."
    )
    parser.add_argument(
        "file_path",
        nargs="?",
        default=DEFAULT_FILE_PATH,
        help="Path to the input JSONL file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    stats = defaultdict(lambda: {"count": 0, "score_sum": 0.0})

    total_count = 0
    total_score_sum = 0.0

    with open(args.file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            task_type = data.get("task_type", "<missing>")
            score = float(data.get("score", 0.0))

            stats[task_type]["count"] += 1
            stats[task_type]["score_sum"] += score

            total_count += 1
            total_score_sum += score

    print(f"file_path = {args.file_path}")
    print(f"total_count = {total_count}")
    print(f"total_score_sum = {total_score_sum}")
    print()
    print("Per task_type statistics:")

    for task_type in sorted(stats):
        print(
            f"{task_type}: count = {stats[task_type]['count']}, "
            f"score_sum = {stats[task_type]['score_sum']}"
        )


if __name__ == "__main__":
    main()
