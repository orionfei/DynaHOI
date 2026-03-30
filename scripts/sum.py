import argparse
import json
from typing import Tuple


DEFAULT_FILE_PATH = (
    "/data1/yfl_data/DynaHOI/scripts/evaluation_results/"
    "gr00t.jsonl"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate overall evaluation metrics from a JSONL file."
    )
    parser.add_argument(
        "file_path",
        nargs="?",
        default=DEFAULT_FILE_PATH,
        help="Path to the input JSONL file.",
    )
    return parser.parse_args()


def parse_completion_ratio(raw_value: str, line_no: int) -> Tuple[int, int]:
    parts = [part.strip() for part in raw_value.split("/")]
    if len(parts) != 2:
        raise ValueError(
            f"Line {line_no}: invalid 'successIndex / total_frames' format: {raw_value!r}"
        )

    try:
        success_index = int(parts[0])
        total_frames = int(parts[1])
    except ValueError as exc:
        raise ValueError(
            f"Line {line_no}: invalid integer values in 'successIndex / total_frames': {raw_value!r}"
        ) from exc

    if total_frames <= 0:
        raise ValueError(
            f"Line {line_no}: total_frames must be positive, got {total_frames}."
        )
    if success_index < 0:
        raise ValueError(
            f"Line {line_no}: success_index must be non-negative, got {success_index}."
        )

    return success_index, total_frames


def require_field(data: dict, field_name: str, line_no: int):
    if field_name not in data:
        raise ValueError(f"Line {line_no}: missing required field {field_name!r}.")
    return data[field_name]


def main():
    args = parse_args()

    total_count = 0
    success_count = 0
    score_sum = 0.0
    q_smooth_sum = 0.0
    q_line_sum = 0.0
    r_time_sum = 0.0

    with open(args.file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)

            success = bool(require_field(data, "success", line_no))
            score = float(require_field(data, "score", line_no))
            q_smooth = float(require_field(data, "smoothness_var", line_no))
            q_line = float(require_field(data, "linearity", line_no))
            completion_raw = require_field(data, "successIndex / total_frames", line_no)

            success_index, total_frames = parse_completion_ratio(completion_raw, line_no)
            r_time = 1.0 - (success_index / total_frames)

            total_count += 1
            success_count += int(success)
            score_sum += score
            q_smooth_sum += q_smooth
            q_line_sum += q_line
            r_time_sum += r_time

    if total_count == 0:
        raise ValueError(f"No valid JSON lines found in {args.file_path}.")

    success_rate = success_count / total_count
    avg_score = score_sum / total_count
    avg_q_smooth = q_smooth_sum / total_count
    avg_q_line = q_line_sum / total_count
    avg_r_time = r_time_sum / total_count

    print(f"file_path = {args.file_path}")
    print(f"total_count = {total_count}")
    print(f"success_count = {success_count}")
    print(f"success_rate = {success_rate:.6f}")
    print(f"avg_score = {avg_score:.6f}")
    print(f"avg_q_smooth = {avg_q_smooth:.6f}")
    print(f"avg_q_line = {avg_q_line:.6f}")
    print(f"avg_r_time = {avg_r_time:.6f}")


if __name__ == "__main__":
    main()
