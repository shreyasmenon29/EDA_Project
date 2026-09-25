import json
import os
import re
import csv
from collections import defaultdict

import matplotlib.pyplot as plt


INPUT_FILE = "data/processed/judgments_processed.jsonl"

TABLE_DIR = "outputs/tables"
FIGURE_DIR = "outputs/figures"

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    records = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


# ============================================================
# DATE
# ============================================================

def get_year(date_value):

    try:
        parts = str(date_value).split("-")

        if len(parts) == 3:
            return int(parts[2])

    except:
        pass

    return None


def get_decade(year):

    if year is None:
        return "Unknown"

    return f"{(year // 10) * 10}s"


# ============================================================
# TEXT METRICS
# ============================================================

def word_count(text):

    return len(
        re.findall(
            r"\b[\w'-]+\b",
            str(text)
        )
    )


def paragraph_count(text):

    paragraphs = re.split(
        r"\n\s*\n",
        str(text)
    )

    return len([
        p for p in paragraphs
        if p.strip()
    ])


# ============================================================
# STRUCTURAL MARKERS
# ============================================================

MARKERS = {

    "coram":
        r"\bCORAM\b",

    "judgment":
        r"\bJUDGMENT\b",

    "order":
        r"\bORDER\b",

    "bench":
        r"\bBENCH\b",

    "petitioner":
        r"\bPETITIONER\b",

    "respondent":
        r"\bRESPONDENT\b",

    "appellant":
        r"\bAPPELLANT\b",

    "facts":
        r"\bFACTS\b",

    "issues":
        r"\bISSUES?\b",

    "held":
        r"\bHELD\b",

    "conclusion":
        r"\bCONCLUSION\b",

    "reasons":
        r"\bREASONS?\b",

    "judgment_by":
        r"\bJUDGMENT\s+BY\b",

    "case_number":
        r"\b(?:CASE|CIVIL|CRIMINAL|WRIT|SPECIAL\s+LEAVE)\b"
        r".{0,50}"
        r"\b(?:NO\.|NOS\.)\b",

    "numbered_paragraphs":
        r"(?m)^\s*\d{1,4}[\.\)]\s+",

    "page_markers":
        r"\bPage\s+\d+\s+of\s+\d+\b",

    "supreme_court":
        r"\bSUPREME\s+COURT\s+OF\s+INDIA\b"
}


# ============================================================
# COUNT MARKERS
# ============================================================

def marker_count(text, pattern):

    return len(
        re.findall(
            pattern,
            str(text),
            flags=re.IGNORECASE
        )
    )


# ============================================================
# SCI
# ============================================================

WEIGHTS = {

    "coram": 1,
    "judgment": 1,
    "order": 1,
    "bench": 1,
    "petitioner": 1,
    "respondent": 1,
    "appellant": 1,
    "facts": 1,
    "issues": 2,
    "held": 2,
    "conclusion": 2,
    "reasons": 1,
    "judgment_by": 2,
    "case_number": 1,
    "numbered_paragraphs": 2,
    "page_markers": 1,
    "supreme_court": 1
}


def calculate_sci(feature_presence):

    score = 0

    for feature, weight in WEIGHTS.items():

        if feature_presence.get(feature):
            score += weight

    maximum = sum(WEIGHTS.values())

    return score, maximum


# ============================================================
# MAIN
# ============================================================

def main():

    records = load_data()

    print("=" * 75)
    print("PHASE 2 - STEP 4B: RIGOROUS STRUCTURAL DRIFT ANALYSIS")
    print("=" * 75)

    processed = []

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

        if words == 0:
            continue

        feature_presence = {}
        feature_counts = {}

        for name, pattern in MARKERS.items():

            count = marker_count(
                text,
                pattern
            )

            feature_counts[name] = count
            feature_presence[name] = count > 0

        sci, max_sci = calculate_sci(
            feature_presence
        )

        row = {
            "year": year,
            "decade": get_decade(year),
            "diary_no": record.get("diary_no"),
            "word_count": words,
            "paragraph_count": paragraphs,

            "paragraphs_per_1000_words":
                paragraphs / words * 1000,

            "sci": sci,

            "sci_percentage":
                sci / max_sci * 100
        }

        # Add marker counts
        for feature in MARKERS:

            row[
                f"{feature}_count"
            ] = feature_counts[feature]

            row[
                f"{feature}_present"
            ] = feature_presence[feature]

            row[
                f"{feature}_per_1000_words"
            ] = (
                feature_counts[feature]
                / words
                * 1000
            )

        processed.append(row)

    # ========================================================
    # SAVE RECORD-LEVEL DATA
    # ========================================================

    record_file = os.path.join(
        TABLE_DIR,
        "structural_drift_record_level.csv"
    )

    fieldnames = list(
        processed[0].keys()
    )

    with open(
        record_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(processed)

    # ========================================================
    # GROUP BY DECADE
    # ========================================================

    decade_groups = defaultdict(list)

    for row in processed:

        decade_groups[
            row["decade"]
        ].append(row)

    decade_rows = []

    for decade in sorted(decade_groups):

        group = decade_groups[decade]

        row = {
            "decade":
                decade,

            "judgment_count":
                len(group),

            "median_words":
                median([
                    x["word_count"]
                    for x in group
                ]),

            "median_paragraphs":
                median([
                    x["paragraph_count"]
                    for x in group
                ]),

            "median_paragraph_density":
                median([
                    x["paragraphs_per_1000_words"]
                    for x in group
                ]),

            "mean_sci":
                sum(
                    x["sci"]
                    for x in group
                ) / len(group),

            "median_sci":
                median([
                    x["sci"]
                    for x in group
                ]),

            "median_sci_percentage":
                median([
                    x["sci_percentage"]
                    for x in group
                ])
        }

        # ----------------------------------------------------
        # Marker prevalence + normalized density
        # ----------------------------------------------------

        for feature in MARKERS:

            presence_rate = (
                sum(
                    x[f"{feature}_present"]
                    for x in group
                )
                / len(group)
                * 100
            )

            density = median([
                x[f"{feature}_per_1000_words"]
                for x in group
            ])

            row[
                f"{feature}_prevalence_pct"
            ] = round(
                presence_rate,
                2
            )

            row[
                f"{feature}_median_density"
            ] = round(
                density,
                4
            )

        decade_rows.append(row)

    # ========================================================
    # MEDIAN HELPER
    # ========================================================

    # Already calculated above through helper below.

    # ========================================================
    # SAVE DECADE TABLE
    # ========================================================

    decade_file = os.path.join(
        TABLE_DIR,
        "structural_drift_by_decade.csv"
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

    # ========================================================
    # PRINT MAIN RESULTS
    # ========================================================

    print("\n--- STRUCTURAL DRIFT BY DECADE ---\n")

    for row in decade_rows:

        print(
            f"{row['decade']:6} | "
            f"N={row['judgment_count']:5} | "
            f"Words={row['median_words']:7.0f} | "
            f"Paragraphs={row['median_paragraphs']:5.1f} | "
            f"Para/1k={row['median_paragraph_density']:6.2f} | "
            f"SCI={row['median_sci']:5.2f}"
        )

    print("\n--- KEY MARKER PREVALENCE ---\n")

    important = [
        "coram",
        "issues",
        "held",
        "numbered_paragraphs",
        "page_markers",
        "judgment_by"
    ]

    for row in decade_rows:

        values = []

        for feature in important:

            values.append(
                f"{feature}="
                f"{row[feature + '_prevalence_pct']:.1f}%"
            )

        print(
            f"{row['decade']:6} | "
            + " | ".join(values)
        )

    # ========================================================
    # FIGURE 1 — SCI
    # ========================================================

    decades = [
        row["decade"]
        for row in decade_rows
    ]

    sci = [
        row["median_sci"]
        for row in decade_rows
    ]

    plt.figure(figsize=(11, 6))

    plt.plot(
        decades,
        sci,
        marker="o"
    )

    plt.title(
        "Median Structural Complexity Index by Decade"
    )

    plt.xlabel("Decade")
    plt.ylabel("Median SCI")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "structural_drift_sci.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # FIGURE 2 — PARAGRAPH DENSITY
    # ========================================================

    paragraph_density = [
        row["median_paragraph_density"]
        for row in decade_rows
    ]

    plt.figure(figsize=(11, 6))

    plt.plot(
        decades,
        paragraph_density,
        marker="o"
    )

    plt.title(
        "Median Paragraph Density by Decade"
    )

    plt.xlabel("Decade")
    plt.ylabel(
        "Paragraphs per 1,000 Words"
    )

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "paragraph_density_by_decade.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # FIGURE 3 — STRUCTURAL MARKER PREVALENCE
    # ========================================================

    plot_features = [
        "coram",
        "issues",
        "held",
        "numbered_paragraphs",
        "page_markers",
        "judgment_by"
    ]

    labels = [
        "CORAM",
        "ISSUES",
        "HELD",
        "Numbered Paragraphs",
        "Page Markers",
        "JUDGMENT BY"
    ]

    plt.figure(figsize=(12, 7))

    for feature, label in zip(
        plot_features,
        labels
    ):

        values = [
            row[
                feature
                + "_prevalence_pct"
            ]
            for row in decade_rows
        ]

        plt.plot(
            decades,
            values,
            marker="o",
            label=label
        )

    plt.title(
        "Structural Marker Prevalence Across Decades"
    )

    plt.xlabel("Decade")
    plt.ylabel(
        "Judgments Containing Marker (%)"
    )

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "structural_marker_drift.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # FIGURE 4 — NORMALIZED MARKER DENSITY
    # ========================================================

    density_features = [
        "coram",
        "issues",
        "held",
        "numbered_paragraphs"
    ]

    labels = [
        "CORAM",
        "ISSUES",
        "HELD",
        "Numbered Paragraphs"
    ]

    plt.figure(figsize=(12, 7))

    for feature, label in zip(
        density_features,
        labels
    ):

        values = [
            row[
                feature
                + "_median_density"
            ]
            for row in decade_rows
        ]

        plt.plot(
            decades,
            values,
            marker="o",
            label=label
        )

    plt.title(
        "Normalized Structural Marker Density"
    )

    plt.xlabel("Decade")
    plt.ylabel(
        "Median Occurrences per 1,000 Words"
    )

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "normalized_structural_marker_density.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 75)
    print("STEP 4B COMPLETE")
    print("=" * 75)

    print("\nTables:")
    print(record_file)
    print(decade_file)

    print("\nFigures:")
    print("structural_drift_sci.png")
    print("paragraph_density_by_decade.png")
    print("structural_marker_drift.png")
    print("normalized_structural_marker_density.png")


# ============================================================
# MEDIAN
# ============================================================

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


# ============================================================

if __name__ == "__main__":
    main()