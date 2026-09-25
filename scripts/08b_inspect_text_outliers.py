import json
import re

INPUT_FILE = "data/processed/judgments_processed.jsonl"


def load_data():
    records = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


def word_count(text):
    return len(
        re.findall(
            r"\b[\w'-]+\b",
            str(text)
        )
    )


def preview(text, length=700):

    text = str(text).strip()

    if len(text) <= length * 2:
        return text

    return (
        text[:length]
        + "\n\n...[MIDDLE OMITTED]...\n\n"
        + text[-length:]
    )


def main():

    records = load_data()

    analyzed = []

    for record in records:

        text = record.get(
            "clean_text",
            ""
        )

        analyzed.append({
            "record": record,
            "word_count": word_count(text)
        })

    analyzed.sort(
        key=lambda x: x["word_count"],
        reverse=True
    )

    longest = analyzed[:5]
    shortest = analyzed[-5:]

    # --------------------------------------------------
    # Longest
    # --------------------------------------------------

    print("=" * 80)
    print("INSPECTION OF 5 LONGEST JUDGMENTS")
    print("=" * 80)

    for i, item in enumerate(longest, start=1):

        record = item["record"]
        text = record.get("clean_text", "")

        print("\n")
        print("-" * 80)
        print(f"RANK: {i}")
        print(f"Diary No: {record.get('diary_no')}")
        print(f"Date: {record.get('judgment_dates')}")
        print(f"Case No: {record.get('case_no')}")
        print(f"Judge: {record.get('judgement_by')}")
        print(f"Word Count: {item['word_count']}")
        print("-" * 80)

        print(preview(text))

    # --------------------------------------------------
    # Shortest
    # --------------------------------------------------

    print("\n\n")
    print("=" * 80)
    print("INSPECTION OF 5 SHORTEST JUDGMENTS")
    print("=" * 80)

    for i, item in enumerate(
        reversed(shortest),
        start=1
    ):

        record = item["record"]
        text = record.get("clean_text", "")

        print("\n")
        print("-" * 80)
        print(f"RANK: {i}")
        print(f"Diary No: {record.get('diary_no')}")
        print(f"Date: {record.get('judgment_dates')}")
        print(f"Case No: {record.get('case_no')}")
        print(f"Judge: {record.get('judgement_by')}")
        print(f"Word Count: {item['word_count']}")
        print("-" * 80)

        print(preview(text))


if __name__ == "__main__":
    main()