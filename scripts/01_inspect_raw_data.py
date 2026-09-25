import json
from pathlib import Path

DATA_PATH = Path("../data/raw/laww_dataset.jsonl")

print("Reading:", DATA_PATH)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    first_line = f.readline()

record = json.loads(first_line)

print("\nFirst record:")
print(record)

print("\nColumns:")
for key in record.keys():
    print("-", key)

