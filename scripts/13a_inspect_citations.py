import json

INPUT_FILE = "data/processed/judgments_processed.jsonl"


def is_missing(value):

    if value is None:
        return True

    if isinstance(value, str):
        return value.strip().lower() in {
            "",
            "nan",
            "none",
            "null"
        }

    if isinstance(value, list):
        return len(value) == 0

    return False


with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    records = [
        json.loads(line)
        for line in f
    ]


print("=" * 75)
print("CITATION FIELD INSPECTION")
print("=" * 75)


fields = [
    "has_citation",
    "citation_extracted",
    "sections_cited",
    "articles_cited"
]


# ============================================================
# FIELD TYPES
# ============================================================

print("\n--- FIELD TYPES ---\n")

for field in fields:

    types = {}

    for record in records:

        value = record.get(field)

        type_name = type(value).__name__

        types[type_name] = (
            types.get(type_name, 0) + 1
        )

    print(field)
    print(types)
    print()


# ============================================================
# SAMPLE NON-MISSING VALUES
# ============================================================

for field in fields:

    print("\n" + "=" * 75)
    print(f"SAMPLES: {field}")
    print("=" * 75)

    shown = 0

    for record in records:

        value = record.get(field)

        if not is_missing(value):

            print(
                f"\nDiary: {record.get('diary_no')}"
            )

            print(
                f"Date: {record.get('judgment_dates')}"
            )

            print(
                f"Value: {repr(value)}"
            )

            shown += 1

            if shown >= 10:
                break


# ============================================================
# RECORDS WITH has_citation = TRUE
# ============================================================

print("\n" + "=" * 75)
print("RECORDS WHERE has_citation IS TRUE")
print("=" * 75)

shown = 0

for record in records:

    value = record.get(
        "has_citation"
    )

    if str(value).lower() in {
        "true",
        "1",
        "yes"
    }:

        print(
            "\nDiary:",
            record.get("diary_no")
        )

        print(
            "Date:",
            record.get("judgment_dates")
        )

        print(
            "has_citation:",
            repr(
                record.get(
                    "has_citation"
                )
            )
        )

        print(
            "citation_extracted:",
            repr(
                record.get(
                    "citation_extracted"
                )
            )
        )

        print(
            "sections_cited:",
            repr(
                record.get(
                    "sections_cited"
                )
            )
        )

        print(
            "articles_cited:",
            repr(
                record.get(
                    "articles_cited"
                )
            )
        )

        shown += 1

        if shown >= 15:
            break


print("\n" + "=" * 75)
print("INSPECTION COMPLETE")
print("=" * 75)python scripts\13a_inspect_citations.py