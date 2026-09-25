import json
import os
import re
import csv
import math
import statistics
from collections import defaultdict
from datetime import datetime

INPUT_FILE = "data/processed/judgments_processed.jsonl"
OUTPUT_DIR = "outputs/tables"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

def load_data():
    records = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


# --------------------------------------------------
# Get decade
# --------------------------------------------------

def get_decade(date_string):

    try:
        year = datetime.strptime(
            date_string,
            "%d-%m-%Y"
        ).year

        return f"{(year // 10) * 10}s"

    except (ValueError, TypeError):
        return "Unknown"


# --------------------------------------------------
# Percentile
# --------------------------------------------------

def percentile(values, p):

    if not values:
        return 0

    values = sorted(values)

    index = (len(values) - 1) * p

    lower = math.floor(index)
    upper = math.ceil(index)

    if lower == upper:
        return values[lower]

    weight = index - lower

    return (
        values[lower] * (1 - weight)
        + values[upper] * weight
    )


# --------------------------------------------------
# Statistics
# --------------------------------------------------

def calculate_statistics(values):

    if not values:
        return {
            "count": 0,
            "mean": 0,
            "median": 0,
            "std": 0,
            "min": 0,
            "q1": 0,
            "q3": 0,
            "iqr": 0,
            "p95": 0,
            "max": 0
        }

    q1 = percentile(values, 0.25)
    q3 = percentile(values, 0.75)

    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": (
            statistics.stdev(values)
            if len(values) > 1
            else 0
        ),
        "min": min(values),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "p95": percentile(values, 0.95),
        "max": max(values)
    }


# --------------------------------------------------
# Text analysis
# --------------------------------------------------

def analyze_text(text):

    text = str(text).strip()

    # Word extraction
    words = re.findall(
        r"\b[\w'-]+\b",
        text
    )

    normalized_words = [
        word.lower()
        for word in words
    ]

    # Sentence detection
    sentences = re.split(
        r"[.!?]+",
        text
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    # Paragraph detection
    paragraphs = re.split(
        r"\n\s*\n",
        text
    )

    paragraphs = [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]

    word_count = len(words)

    unique_word_count = len(
        set(normalized_words)
    )

    vocabulary_diversity = (
        unique_word_count / word_count
        if word_count > 0
        else 0
    )

    average_sentence_length = (
        word_count / len(sentences)
        if sentences
        else 0
    )

    return {
        "word_count": word_count,
        "character_count": len(text),
        "sentence_count": len(sentences),
        "paragraph_count": len(paragraphs),
        "unique_word_count": unique_word_count,
        "vocabulary_diversity": vocabulary_diversity,
        "average_sentence_length": average_sentence_length
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    records = load_data()

    print("=" * 75)
    print("PHASE 2 - STEP 2: TEXT STATISTICS")
    print("=" * 75)

    print(f"\nTotal records analyzed: {len(records)}")

    metrics = [
        "word_count",
        "character_count",
        "sentence_count",
        "paragraph_count",
        "unique_word_count",
        "vocabulary_diversity",
        "average_sentence_length"
    ]

    overall_data = defaultdict(list)
    decade_data = defaultdict(
        lambda: defaultdict(list)
    )

    # --------------------------------------------------
    # Analyze every judgment
    # --------------------------------------------------

    for record in records:

        text = record.get(
            "clean_text",
            ""
        )

        result = analyze_text(text)

        decade = get_decade(
            record.get("judgment_dates")
        )

        for metric in metrics:

            value = result[metric]

            overall_data[metric].append(
                value
            )

            decade_data[decade][metric].append(
                value
            )

    # --------------------------------------------------
    # Overall statistics
    # --------------------------------------------------

    print("\n")
    print("=" * 75)
    print("OVERALL TEXT STATISTICS")
    print("=" * 75)

    overall_results = {}

    for metric in metrics:

        result = calculate_statistics(
            overall_data[metric]
        )

        overall_results[metric] = result

        print(f"\n{metric}")
        print(f"  Mean:       {result['mean']:.2f}")
        print(f"  Median:     {result['median']:.2f}")
        print(f"  Std:        {result['std']:.2f}")
        print(f"  Min:        {result['min']:.2f}")
        print(f"  Q1:         {result['q1']:.2f}")
        print(f"  Q3:         {result['q3']:.2f}")
        print(f"  IQR:        {result['iqr']:.2f}")
        print(f"  95th %ile:  {result['p95']:.2f}")
        print(f"  Max:        {result['max']:.2f}")

    # --------------------------------------------------
    # By decade
    # --------------------------------------------------

    print("\n")
    print("=" * 75)
    print("TEXT STATISTICS BY DECADE")
    print("=" * 75)

    decade_results = {}

    for decade in sorted(decade_data):

        print(
            f"\n--- {decade} "
            f"({len(decade_data[decade]['word_count'])} judgments) ---"
        )

        decade_results[decade] = {}

        for metric in metrics:

            result = calculate_statistics(
                decade_data[decade][metric]
            )

            decade_results[decade][metric] = result

            print(
                f"{metric:28} "
                f"Median: {result['median']:10.2f} | "
                f"Mean: {result['mean']:10.2f} | "
                f"IQR: {result['iqr']:10.2f}"
            )

    # --------------------------------------------------
    # Save detailed CSV
    # --------------------------------------------------

    output_file = os.path.join(
        OUTPUT_DIR,
        "text_statistics_by_decade.csv"
    )

    fieldnames = [
        "decade",
        "judgment_count"
    ]

    for metric in metrics:

        fieldnames.extend([
            f"{metric}_mean",
            f"{metric}_median",
            f"{metric}_std",
            f"{metric}_min",
            f"{metric}_q1",
            f"{metric}_q3",
            f"{metric}_iqr",
            f"{metric}_p95",
            f"{metric}_max"
        ])

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for decade in sorted(decade_results):

            row = {
                "decade": decade,
                "judgment_count": len(
                    decade_data[decade]["word_count"]
                )
            }

            for metric in metrics:

                result = decade_results[
                    decade
                ][metric]

                row[
                    f"{metric}_mean"
                ] = round(
                    result["mean"], 4
                )

                row[
                    f"{metric}_median"
                ] = round(
                    result["median"], 4
                )

                row[
                    f"{metric}_std"
                ] = round(
                    result["std"], 4
                )

                row[
                    f"{metric}_min"
                ] = round(
                    result["min"], 4
                )

                row[
                    f"{metric}_q1"
                ] = round(
                    result["q1"], 4
                )

                row[
                    f"{metric}_q3"
                ] = round(
                    result["q3"], 4
                )

                row[
                    f"{metric}_iqr"
                ] = round(
                    result["iqr"], 4
                )

                row[
                    f"{metric}_p95"
                ] = round(
                    result["p95"], 4
                )

                row[
                    f"{metric}_max"
                ] = round(
                    result["max"], 4
                )

            writer.writerow(row)

    # --------------------------------------------------
    # Save overall statistics CSV
    # --------------------------------------------------

    overall_file = os.path.join(
        OUTPUT_DIR,
        "overall_text_statistics.csv"
    )

    with open(
        overall_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "metric",
            "count",
            "mean",
            "median",
            "std",
            "min",
            "q1",
            "q3",
            "iqr",
            "p95",
            "max"
        ])

        for metric, result in overall_results.items():

            writer.writerow([
                metric,
                result["count"],
                round(result["mean"], 4),
                round(result["median"], 4),
                round(result["std"], 4),
                round(result["min"], 4),
                round(result["q1"], 4),
                round(result["q3"], 4),
                round(result["iqr"], 4),
                round(result["p95"], 4),
                round(result["max"], 4)
            ])

    # --------------------------------------------------
    # Save readable summary
    # --------------------------------------------------

    summary_file = os.path.join(
        OUTPUT_DIR,
        "text_statistics_summary.txt"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("TEXT STATISTICS SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write(
            f"Total records: {len(records)}\n\n"
        )

        f.write("OVERALL STATISTICS\n")
        f.write("-" * 70 + "\n\n")

        for metric, result in overall_results.items():

            f.write(f"{metric}\n")
            f.write(
                f"Mean: {result['mean']:.2f}\n"
            )
            f.write(
                f"Median: {result['median']:.2f}\n"
            )
            f.write(
                f"Std: {result['std']:.2f}\n"
            )
            f.write(
                f"Min: {result['min']:.2f}\n"
            )
            f.write(
                f"Q1: {result['q1']:.2f}\n"
            )
            f.write(
                f"Q3: {result['q3']:.2f}\n"
            )
            f.write(
                f"IQR: {result['iqr']:.2f}\n"
            )
            f.write(
                f"95th percentile: "
                f"{result['p95']:.2f}\n"
            )
            f.write(
                f"Max: {result['max']:.2f}\n\n"
            )

        f.write("\nSTATISTICS BY DECADE\n")
        f.write("=" * 70 + "\n\n")

        for decade in sorted(decade_results):

            f.write(
                f"{decade}\n"
            )

            for metric in metrics:

                result = decade_results[
                    decade
                ][metric]

                f.write(
                    f"  {metric}: "
                    f"median={result['median']:.2f}, "
                    f"mean={result['mean']:.2f}, "
                    f"IQR={result['iqr']:.2f}, "
                    f"P95={result['p95']:.2f}\n"
                )

            f.write("\n")

    # --------------------------------------------------
    # Finish
    # --------------------------------------------------

    print("\n")
    print("=" * 75)
    print("STEP 2 COMPLETE")
    print("=" * 75)

    print("\nGenerated files:")
    print(
        f"  {output_file}"
    )
    print(
        f"  {overall_file}"
    )
    print(
        f"  {summary_file}"
    )


if __name__ == "__main__":
    main()