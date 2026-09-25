import json
import os
import re
import csv

INPUT_FILE = "data/processed/judgments_processed.jsonl"
OUTPUT_DIR = "outputs/tables"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    records = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


def get_word_count(text):
    return len(
        re.findall(
            r"\b[\w'-]+\b",
            str(text)
        )
    )


def main():

    records = load_data()

    analyzed = []

    for record in records:

        text = record.get(
            "clean_text",
            ""
        )

        word_count = get_word_count(text)

        analyzed.append({
            "diary_no": record.get("diary_no", ""),
            "judgment_date": record.get(
                "judgment_dates",
                ""
            ),
            "bench": record.get(
                "bench",
                ""
            ),
            "judgement_by": record.get(
                "judgement_by",
                ""
            ),
            "case_no": record.get(
                "case_no",
                ""
            ),
            "pet": record.get(
                "pet",
                ""
            ),
            "res": record.get(
                "res",
                ""
            ),
            "word_count": word_count,
            "character_count": len(
                str(text)
            ),
            "paragraph_count": len([
                p for p in re.split(
                    r"\n\s*\n",
                    str(text)
                )
                if p.strip()
            ])
        })

    # --------------------------------------------------
    # Longest judgments
    # --------------------------------------------------

    longest = sorted(
        analyzed,
        key=lambda x: x["word_count"],
        reverse=True
    )[:25]

    # --------------------------------------------------
    # Shortest judgments
    # --------------------------------------------------

    shortest = sorted(
        analyzed,
        key=lambda x: x["word_count"]
    )[:25]

    # --------------------------------------------------
    # Save longest
    # --------------------------------------------------

    longest_file = os.path.join(
        OUTPUT_DIR,
        "longest_judgments.csv"
    )

    fieldnames = [
        "diary_no",
        "judgment_date",
        "case_no",
        "bench",
        "judgement_by",
        "pet",
        "res",
        "word_count",
        "character_count",
        "paragraph_count"
    ]

    with open(
        longest_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(longest)

    # --------------------------------------------------
    # Save shortest
    # --------------------------------------------------

    shortest_file = os.path.join(
        OUTPUT_DIR,
        "shortest_judgments.csv"
    )

    with open(
        shortest_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(shortest)

    # --------------------------------------------------
    # Print longest
    # --------------------------------------------------

    print("=" * 80)
    print("TOP 25 LONGEST JUDGMENTS")
    print("=" * 80)

    for i, item in enumerate(longest, start=1):

        print(
            f"{i:2}. "
            f"{item['word_count']:>8} words | "
            f"{item['judgment_date']} | "
            f"Diary: {item['diary_no']}"
        )

    # --------------------------------------------------
    # Print shortest
    # --------------------------------------------------

    print("\n")
    print("=" * 80)
    print("TOP 25 SHORTEST JUDGMENTS")
    print("=" * 80)

    for i, item in enumerate(shortest, start=1):

        print(
            f"{i:2}. "
            f"{item['word_count']:>8} words | "
            f"{item['judgment_date']} | "
            f"Diary: {item['diary_no']}"
        )

    print("\n")
    print("=" * 80)
    print("OUTLIER ANALYSIS COMPLETE")
    print("=" * 80)

    print("\nGenerated:")
    print(longest_file)
    print(shortest_file)


if __name__ == "__main__":
    main()