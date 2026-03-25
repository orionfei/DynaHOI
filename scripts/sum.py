import json

file_path = "/data1/yfl_data/DynaHOI/scripts/evaluation_results/results_v1-checkpoint-8750:uniform5_baseline:motionhint_diff_map_and_crop.jsonl"

score_sum = 0.0
count = 0

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        if "score" in data:
            score_sum += float(data["score"])
            count += 1

print("score_sum =", score_sum)
print("count =", count)
