from pathlib import Path

DATA_PATH = Path("../data/raw/laww_dataset.jsonl")

count = 0

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for _ in f:
        count += 1

print("Total records:", count)