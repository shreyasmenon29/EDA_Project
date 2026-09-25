import json
import random
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

DATA_PATH = Path("../data/raw/laww_dataset.jsonl")
OUTPUT_PATH = Path("../data/raw/sampled_judgments.jsonl")

TARGET_SIZE = 12000
SEED = 42

random.seed(SEED)

# --------------------------------------------------
# PASS 1: Count judgments in each decade
# --------------------------------------------------

decade_counts = Counter()

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        try:
            record = json.loads(line)
            date = record.get("judgment_dates")

            if date:
                year = datetime.strptime(
                    date,
                    "%d-%m-%Y"
                ).year

                decade = (year // 10) * 10
                decade_counts[decade] += 1

        except (ValueError, TypeError):
            continue

total_valid = sum(decade_counts.values())

print("Original distribution:")
for decade, count in sorted(decade_counts.items()):
    print(f"{decade}s: {count}")

print("\nTotal valid records:", total_valid)

# --------------------------------------------------
# Calculate proportional sample sizes
# --------------------------------------------------

sample_sizes = {}

for decade, count in decade_counts.items():
    sample_sizes[decade] = round(
        (count / total_valid) * TARGET_SIZE
    )

# Adjust rounding so total is exactly TARGET_SIZE
difference = TARGET_SIZE - sum(sample_sizes.values())

largest_decade = max(
    sample_sizes,
    key=sample_sizes.get
)

sample_sizes[largest_decade] += difference

print("\nSample sizes:")
for decade, size in sorted(sample_sizes.items()):
    print(f"{decade}s: {size}")

print(
    "\nTotal sample:",
    sum(sample_sizes.values())
)

# --------------------------------------------------
# PASS 2: Reservoir sampling within each decade
# --------------------------------------------------

reservoirs = defaultdict(list)
seen = Counter()

with open(DATA_PATH, "r", encoding="utf-8") as f:

    for line in f:

        try:
            record = json.loads(line)

            date = record.get("judgment_dates")

            if not date:
                continue

            year = datetime.strptime(
                date,
                "%d-%m-%Y"
            ).year

            decade = (year // 10) * 10

            if decade not in sample_sizes:
                continue

            seen[decade] += 1

            k = sample_sizes[decade]

            if len(reservoirs[decade]) < k:
                reservoirs[decade].append(record)

            else:
                replacement_index = random.randint(
                    0,
                    seen[decade] - 1
                )

                if replacement_index < k:
                    reservoirs[decade][replacement_index] = record

        except (ValueError, TypeError):
            continue

# --------------------------------------------------
# Write sample
# --------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    for decade in sorted(reservoirs):

        for record in reservoirs[decade]:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

print("\nSample created successfully!")
print("Output:", OUTPUT_PATH)

print("\nFinal sample distribution:")

total_sampled = 0

for decade in sorted(reservoirs):

    count = len(reservoirs[decade])
    total_sampled += count

    print(
        f"{decade}s: {count}"
    )

print("\nTotal sampled:", total_sampled)