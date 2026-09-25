import json
import os
import csv
from collections import Counter, defaultdict
from datetime import datetime

INPUT_FILE = "data/processed/judgments_processed.jsonl"
OUTPUT_DIR = "outputs/tables"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def load_data():
    records = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


def is_missing(value):
    if value is None:
        return True

    if isinstance(value, str):
        normalized = value.strip().lower()

        return normalized in {
            "",
            "nan",
            "none",
            "null"
        }

    if isinstance(value, list):
        if len(value) == 0:
            return True

        # Treat lists containing only missing-like values as missing
        return all(
            isinstance(item, str)
            and item.strip().lower() in {
                "",
                "nan",
                "none",
                "null"
            }
            for item in value
        )

    return False


def parse_date(date_value):
    if is_missing(date_value):
        return None

    try:
        return datetime.strptime(
            str(date_value).strip(),
            "%d-%m-%Y"
        )
    except ValueError:
        return None


def get_decade(date_value):
    date = parse_date(date_value)

    if date is None:
        return "Unknown"

    return f"{(date.year // 10) * 10}s"


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    records = load_data()

    total_records = len(records)

    print("=" * 70)
    print("PHASE 2 - STEP 1: DATASET HEALTH & QUALITY")
    print("=" * 70)

    print(f"\nTotal records: {total_records}")

    # --------------------------------------------------
    # 1. Duplicate diary numbers
    # --------------------------------------------------

    diary_numbers = [
        str(r.get("diary_no")).strip()
        for r in records
        if not is_missing(r.get("diary_no"))
    ]

    diary_counter = Counter(diary_numbers)

    duplicate_diary_groups = {
        value: count
        for value, count in diary_counter.items()
        if count > 1
    }

    duplicate_diary_occurrences = sum(
        count - 1
        for count in diary_counter.values()
        if count > 1
    )

    print("\n--- Duplicate Diary Numbers ---")
    print(
        f"Duplicate diary number groups: "
        f"{len(duplicate_diary_groups)}"
    )
    print(
        f"Duplicate diary number occurrences: "
        f"{duplicate_diary_occurrences}"
    )

    # --------------------------------------------------
    # 2. Duplicate full texts
    # --------------------------------------------------

    text_to_records = defaultdict(list)

    for index, record in enumerate(records):

        text = str(
            record.get("full_text", "")
        ).strip()

        if text:
            text_to_records[text].append(index)

    duplicate_text_groups = {
        text: indexes
        for text, indexes in text_to_records.items()
        if len(indexes) > 1
    }

    duplicate_text_occurrences = sum(
        len(indexes) - 1
        for indexes in duplicate_text_groups.values()
    )

    print("\n--- Duplicate Full Text ---")
    print(
        f"Duplicate text groups: "
        f"{len(duplicate_text_groups)}"
    )
    print(
        f"Duplicate text occurrences: "
        f"{duplicate_text_occurrences}"
    )

    # --------------------------------------------------
    # 3. Field-level missingness
    # --------------------------------------------------

    fields = [
        "diary_no",
        "judgement_type",
        "case_no",
        "pet",
        "res",
        "pet_adv",
        "res_adv",
        "bench",
        "judgement_by",
        "judgment_dates",
        "outcome",
        "sections_cited",
        "articles_cited",
        "full_text",
        "clean_text"
    ]

    print("\n--- Field-Level Missingness ---")

    missing_results = []

    for field in fields:

        missing_count = sum(
            1
            for record in records
            if is_missing(record.get(field))
        )

        present_count = total_records - missing_count

        missing_percentage = (
            missing_count / total_records
        ) * 100

        present_percentage = (
            present_count / total_records
        ) * 100

        missing_results.append({
            "field": field,
            "present_count": present_count,
            "missing_count": missing_count,
            "present_percentage": present_percentage,
            "missing_percentage": missing_percentage
        })

        print(
            f"{field:20} "
            f"Missing: {missing_count:6} "
            f"({missing_percentage:6.2f}%)"
        )

    # --------------------------------------------------
    # 4. Text validity
    # --------------------------------------------------

    empty_full_text = sum(
        1
        for r in records
        if is_missing(r.get("full_text"))
    )

    empty_clean_text = sum(
        1
        for r in records
        if is_missing(r.get("clean_text"))
    )

    print("\n--- Text Validity ---")
    print(f"Empty full_text:  {empty_full_text}")
    print(f"Empty clean_text: {empty_clean_text}")

    # --------------------------------------------------
    # 5. Date validation
    # --------------------------------------------------

    valid_dates = 0
    invalid_dates = 0
    missing_dates = 0

    for record in records:

        value = record.get("judgment_dates")

        if is_missing(value):
            missing_dates += 1

        elif parse_date(value) is not None:
            valid_dates += 1

        else:
            invalid_dates += 1

    print("\n--- Date Validation ---")
    print(f"Valid dates:   {valid_dates}")
    print(f"Invalid dates: {invalid_dates}")
    print(f"Missing dates: {missing_dates}")

    # --------------------------------------------------
    # 6. Metadata completeness BY DECADE
    # --------------------------------------------------

    metadata_fields = [
        "judgement_type",
        "case_no",
        "pet",
        "res",
        "pet_adv",
        "res_adv",
        "bench",
        "judgement_by",
        "judgment_dates",
        "outcome",
        "sections_cited",
        "articles_cited"
    ]

    decade_records = defaultdict(list)

    for record in records:

        decade = get_decade(
            record.get("judgment_dates")
        )

        decade_records[decade].append(record)

    print("\n" + "=" * 70)
    print("METADATA AVAILABILITY BY DECADE")
    print("=" * 70)

    decade_rows = []

    for decade in sorted(decade_records):

        decade_data = decade_records[decade]
        decade_total = len(decade_data)

        print(f"\n--- {decade} ({decade_total} judgments) ---")

        row = {
            "decade": decade,
            "judgment_count": decade_total
        }

        for field in metadata_fields:

            present = sum(
                1
                for record in decade_data
                if not is_missing(record.get(field))
            )

            percentage = (
                present / decade_total
            ) * 100

            row[f"{field}_present"] = present
            row[f"{field}_percentage"] = percentage

            print(
                f"{field:20} "
                f"{percentage:6.2f}%"
            )

        decade_rows.append(row)

    # --------------------------------------------------
    # 7. Save field-level missingness CSV
    # --------------------------------------------------

    missing_file = os.path.join(
        OUTPUT_DIR,
        "missing_value_analysis.csv"
    )

    with open(
        missing_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "field",
                "present_count",
                "missing_count",
                "present_percentage",
                "missing_percentage"
            ]
        )

        writer.writeheader()
        writer.writerows(missing_results)

    # --------------------------------------------------
    # 8. Save metadata-by-decade CSV
    # --------------------------------------------------

    decade_file = os.path.join(
        OUTPUT_DIR,
        "metadata_completeness_by_decade.csv"
    )

    if decade_rows:

        fieldnames = list(
            decade_rows[0].keys()
        )

        with open(
            decade_file,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(decade_rows)

    # --------------------------------------------------
    # 9. Save duplicate text investigation
    # --------------------------------------------------

    duplicate_file = os.path.join(
        OUTPUT_DIR,
        "duplicate_text_analysis.csv"
    )

    with open(
        duplicate_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "duplicate_group",
            "number_of_records",
            "diary_numbers",
            "judgment_dates"
        ])

        for group_number, indexes in enumerate(
            duplicate_text_groups.values(),
            start=1
        ):

            diary_values = [
                str(records[i].get("diary_no", ""))
                for i in indexes
            ]

            date_values = [
                str(records[i].get("judgment_dates", ""))
                for i in indexes
            ]

            writer.writerow([
                group_number,
                len(indexes),
                " | ".join(diary_values),
                " | ".join(date_values)
            ])

    # --------------------------------------------------
    # 10. Save overall summary
    # --------------------------------------------------

    summary_file = os.path.join(
        OUTPUT_DIR,
        "dataset_quality_summary.txt"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("DATASET QUALITY SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write(
            f"Total records: {total_records}\n"
        )

        f.write(
            f"Duplicate diary number groups: "
            f"{len(duplicate_diary_groups)}\n"
        )

        f.write(
            f"Duplicate diary number occurrences: "
            f"{duplicate_diary_occurrences}\n"
        )

        f.write(
            f"Duplicate full-text groups: "
            f"{len(duplicate_text_groups)}\n"
        )

        f.write(
            f"Duplicate full-text occurrences: "
            f"{duplicate_text_occurrences}\n"
        )

        f.write("\nTEXT VALIDITY\n")
        f.write("-" * 60 + "\n")

        f.write(
            f"Empty full_text: {empty_full_text}\n"
        )

        f.write(
            f"Empty clean_text: {empty_clean_text}\n"
        )

        f.write("\nDATE VALIDATION\n")
        f.write("-" * 60 + "\n")

        f.write(
            f"Valid dates: {valid_dates}\n"
        )

        f.write(
            f"Invalid dates: {invalid_dates}\n"
        )

        f.write(
            f"Missing dates: {missing_dates}\n"
        )

    # --------------------------------------------------
    # Final
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 1 REVISED ANALYSIS COMPLETE")
    print("=" * 70)

    print("\nGenerated files:")

    print(
        f"  {missing_file}"
    )

    print(
        f"  {decade_file}"
    )

    print(
        f"  {duplicate_file}"
    )

    print(
        f"  {summary_file}"
    )


if __name__ == "__main__":
    main()