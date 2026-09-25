import json
from pathlib import Path

DATA_PATH = Path("../data/processed/judgments_processed.jsonl")

count = 0
missing_clean_text = 0
empty_clean_text = 0

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)

        count += 1

        if "clean_text" not in record:
            missing_clean_text += 1
        elif not record["clean_text"].strip():
            empty_clean_text += 1

print("Processed records:", count)
print("Missing clean_text:", missing_clean_text)
print("Empty clean_text:", empty_clean_text)