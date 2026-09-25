import json
from pathlib import Path
from collections import Counter
from datetime import datetime

DATA_PATH = Path("../data/raw/laww_dataset.jsonl")

decade_counts = Counter()
invalid_dates = 0

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)

        date = record.get("judgment_dates")

        if date:
            try:
                year = datetime.strptime(
                    date,
                    "%d-%m-%Y"
                ).year

                decade = (year // 10) * 10
                decade_counts[decade] += 1

            except ValueError:
                invalid_dates += 1

print("Decade distribution:\n")

for decade, count in sorted(decade_counts.items()):
    print(f"{decade}s: {count}")

print("\nTotal valid dates:", sum(decade_counts.values()))
print("Invalid dates:", invalid_dates)