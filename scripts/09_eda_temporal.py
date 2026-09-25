import json
import os
import csv
import re
from collections import defaultdict, Counter
from datetime import datetime

import matplotlib.pyplot as plt


INPUT_FILE = "data/processed/judgments_processed.jsonl"

TABLE_DIR = "outputs/tables"
FIGURE_DIR = "outputs/figures"

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


# --------------------------------------------------
# Load data
# --------------------------------------------------

def load_data():

    records = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:

        for line in f:
            records.append(json.loads(line))

    return records


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def get_year(date_value):

    try:
        return datetime.strptime(
            date_value,
            "%d-%m-%Y"
        ).year

    except (ValueError, TypeError):
        return None


def get_decade(year):

    if year is None:
        return "Unknown"

    return f"{(year // 10) * 10}s"


def word_count(text):

    return len(
        re.findall(
            r"\b[\w'-]+\b",
            str(text)
        )
    )


def paragraph_count(text):

    return len([
        p
        for p in re.split(
            r"\n\s*\n",
            str(text)
        )
        if p.strip()
    ])


def sentence_count(text):

    sentences = re.split(
        r"[.!?]+",
        str(text)
    )

    return len([
        s
        for s in sentences
        if s.strip()
    ])


def median(values):

    if not values:
        return 0

    values = sorted(values)

    n = len(values)

    middle = n // 2

    if n % 2 == 0:
        return (
            values[middle - 1]
            + values[middle]
        ) / 2

    return values[middle]


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    records = load_data()

    print("=" * 75)
    print("PHASE 2 - STEP 3: TEMPORAL ANALYSIS")
    print("=" * 75)

    yearly = defaultdict(list)
    decade = defaultdict(list)

    # --------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------

    for record in records:

        year = get_year(
            record.get("judgment_dates")
        )

        if year is None:
            continue

        text = record.get(
            "clean_text",
            ""
        )

        words = word_count(text)
        paragraphs = paragraph_count(text)
        sentences = sentence_count(text)

        unique_words = len(
            set(
                re.findall(
                    r"\b[\w'-]+\b",
                    str(text).lower()
                )
            )
        )

        vocabulary_diversity = (
            unique_words / words
            if words > 0
            else 0
        )

        sentence_length = (
            words / sentences
            if sentences > 0
            else 0
        )

        metrics = {
            "word_count": words,
            "paragraph_count": paragraphs,
            "sentence_count": sentences,
            "vocabulary_diversity":
                vocabulary_diversity,
            "average_sentence_length":
                sentence_length
        }

        yearly[year].append(metrics)
        decade[get_decade(year)].append(metrics)

    # --------------------------------------------------
    # Build temporal table
    # --------------------------------------------------

    rows = []

    for year in sorted(yearly):

        data = yearly[year]

        row = {
            "year": year,
            "judgment_count": len(data),
            "median_words": median([
                x["word_count"]
                for x in data
            ]),
            "median_paragraphs": median([
                x["paragraph_count"]
                for x in data
            ]),
            "median_sentences": median([
                x["sentence_count"]
                for x in data
            ]),
            "median_vocabulary_diversity":
                median([
                    x["vocabulary_diversity"]
                    for x in data
                ]),
            "median_sentence_length":
                median([
                    x["average_sentence_length"]
                    for x in data
                ])
        }

        rows.append(row)

    # --------------------------------------------------
    # Save yearly table
    # --------------------------------------------------

    yearly_file = os.path.join(
        TABLE_DIR,
        "temporal_analysis_by_year.csv"
    )

    with open(
        yearly_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------
    # Decade table
    # --------------------------------------------------

    decade_rows = []

    for d in sorted(decade):

        data = decade[d]

        decade_rows.append({
            "decade": d,
            "judgment_count": len(data),
            "median_words": median([
                x["word_count"]
                for x in data
            ]),
            "median_paragraphs": median([
                x["paragraph_count"]
                for x in data
            ]),
            "median_sentences": median([
                x["sentence_count"]
                for x in data
            ]),
            "median_vocabulary_diversity":
                median([
                    x["vocabulary_diversity"]
                    for x in data
                ]),
            "median_sentence_length":
                median([
                    x["average_sentence_length"]
                    for x in data
                ])
        })

    decade_file = os.path.join(
        TABLE_DIR,
        "temporal_analysis_by_decade.csv"
    )

    with open(
        decade_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                decade_rows[0].keys()
            )
        )

        writer.writeheader()
        writer.writerows(decade_rows)

    # --------------------------------------------------
    # Print decade results
    # --------------------------------------------------

    print("\n--- DECADE SUMMARY ---\n")

    for row in decade_rows:

        print(
            f"{row['decade']:6} | "
            f"N={row['judgment_count']:5} | "
            f"Words={row['median_words']:8.1f} | "
            f"Paragraphs={row['median_paragraphs']:6.1f} | "
            f"Sentences={row['median_sentences']:7.1f} | "
            f"Vocab={row['median_vocabulary_diversity']:.3f} | "
            f"SentLen={row['median_sentence_length']:.2f}"
        )

    # --------------------------------------------------
    # Prepare plotting data
    # --------------------------------------------------

    years = [
        row["year"]
        for row in rows
    ]

    counts = [
        row["judgment_count"]
        for row in rows
    ]

    words = [
        row["median_words"]
        for row in rows
    ]

    paragraphs = [
        row["median_paragraphs"]
        for row in rows
    ]

    vocabulary = [
        row["median_vocabulary_diversity"]
        for row in rows
    ]

    sentence_length = [
        row["median_sentence_length"]
        for row in rows
    ]

    # --------------------------------------------------
    # Figure 1 — Judgment count
    # --------------------------------------------------

    plt.figure(figsize=(12, 6))

    plt.plot(
        years,
        counts,
        marker="o"
    )

    plt.title(
        "Number of Supreme Court Judgments by Year"
    )

    plt.xlabel("Year")
    plt.ylabel("Number of Judgments")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "judgments_by_year.png"
        ),
        dpi=300
    )

    plt.close()

    # --------------------------------------------------
    # Figure 2 — Median judgment length
    # --------------------------------------------------

    plt.figure(figsize=(12, 6))

    plt.plot(
        years,
        words,
        marker="o"
    )

    plt.title(
        "Median Judgment Length by Year"
    )

    plt.xlabel("Year")
    plt.ylabel("Median Word Count")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "median_judgment_length_by_year.png"
        ),
        dpi=300
    )

    plt.close()

    # --------------------------------------------------
    # Figure 3 — Median paragraph count
    # --------------------------------------------------

    plt.figure(figsize=(12, 6))

    plt.plot(
        years,
        paragraphs,
        marker="o"
    )

    plt.title(
        "Median Paragraph Count by Year"
    )

    plt.xlabel("Year")
    plt.ylabel("Median Paragraph Count")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "median_paragraphs_by_year.png"
        ),
        dpi=300
    )

    plt.close()

    # --------------------------------------------------
    # Figure 4 — Vocabulary diversity
    # --------------------------------------------------

    plt.figure(figsize=(12, 6))

    plt.plot(
        years,
        vocabulary,
        marker="o"
    )

    plt.title(
        "Median Vocabulary Diversity by Year"
    )

    plt.xlabel("Year")
    plt.ylabel("Unique Words / Total Words")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "vocabulary_diversity_by_year.png"
        ),
        dpi=300
    )

    plt.close()

    # --------------------------------------------------
    # Figure 5 — Sentence length
    # --------------------------------------------------

    plt.figure(figsize=(12, 6))

    plt.plot(
        years,
        sentence_length,
        marker="o"
    )

    plt.title(
        "Median Sentence Length by Year"
    )

    plt.xlabel("Year")
    plt.ylabel("Words per Sentence")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "sentence_length_by_year.png"
        ),
        dpi=300
    )

    plt.close()

    # --------------------------------------------------
    # Finish
    # --------------------------------------------------

    print("\n" + "=" * 75)
    print("STEP 3 COMPLETE")
    print("=" * 75)

    print("\nTables:")
    print(yearly_file)
    print(decade_file)

    print("\nFigures:")
    print("judgments_by_year.png")
    print("median_judgment_length_by_year.png")
    print("median_paragraphs_by_year.png")
    print("vocabulary_diversity_by_year.png")
    print("sentence_length_by_year.png")


if __name__ == "__main__":
    main()