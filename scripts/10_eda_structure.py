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
# DATE / DECADE
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
# DETECT MARKERS
# ============================================================

def detect_features(text):

    text = str(text)

    results = {}

    for name, pattern in MARKERS.items():

        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        results[name] = len(matches) > 0

    return results


# ============================================================
# STRUCTURAL COMPLEXITY
# ============================================================

# Project-defined analytical index.
# It is NOT an official legal measure.

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


def calculate_sci(features):

    score = 0
    maximum = sum(WEIGHTS.values())

    for feature, weight in WEIGHTS.items():

        if features.get(feature):
            score += weight

    return score, maximum


# ============================================================
# FORMAT CLASSIFICATION
# ============================================================

def classify_format(features):

    structural_count = sum(
        features.values()
    )

    numbered = features[
        "numbered_paragraphs"
    ]

    modern_markers = sum([
        features["coram"],
        features["judgment_by"],
        features["issues"],
        features["held"],
        features["conclusion"],
        features["reasons"],
        numbered
    ])

    old_markers = sum([
        features["bench"],
        features["petitioner"],
        features["respondent"],
        features["appellant"],
        features["case_number"]
    ])

    if structural_count >= 8 and modern_markers >= 4:
        return "Modern Structured"

    if structural_count >= 5 and (
        modern_markers >= 2 or
        old_markers >= 3
    ):
        return "Structured"

    if structural_count >= 2:
        return "Minimal Structure"

    return "Unknown"


# ============================================================
# MAIN
# ============================================================

def main():

    records = load_data()

    print("=" * 75)
    print("PHASE 2 - STEP 4: STRUCTURAL / FORMAT ANALYSIS")
    print("=" * 75)

    processed = []

    for record in records:

        year = get_year(
            record.get("judgment_dates")
        )

        text = record.get(
            "clean_text",
            ""
        )

        features = detect_features(text)

        sci, max_sci = calculate_sci(
            features
        )

        format_class = classify_format(
            features
        )

        result = {
            "diary_no":
                record.get("diary_no"),

            "year":
                year,

            "decade":
                get_decade(year),

            "format_class":
                format_class,

            "sci":
                sci,

            "sci_percentage":
                round(
                    sci / max_sci * 100,
                    2
                )
        }

        result.update(features)

        processed.append(result)

    # ========================================================
    # SAVE RECORD-LEVEL STRUCTURAL FEATURES
    # ========================================================

    feature_file = os.path.join(
        TABLE_DIR,
        "structural_features.csv"
    )

    fieldnames = list(
        processed[0].keys()
    )

    with open(
        feature_file,
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
    # DECADE ANALYSIS
    # ========================================================

    decade_groups = defaultdict(list)

    for row in processed:

        decade_groups[
            row["decade"]
        ].append(row)

    decade_rows = []

    feature_names = list(MARKERS.keys())

    for decade in sorted(decade_groups):

        group = decade_groups[decade]

        row = {
            "decade":
                decade,

            "judgment_count":
                len(group),

            "mean_sci":
                round(
                    sum(x["sci"] for x in group)
                    / len(group),
                    2
                ),

            "median_sci":
                sorted(
                    x["sci"] for x in group
                )[len(group) // 2],

            "mean_sci_percentage":
                round(
                    sum(
                        x["sci_percentage"]
                        for x in group
                    ) / len(group),
                    2
                )
        }

        for feature in feature_names:

            count = sum(
                1
                for x in group
                if x[feature]
            )

            row[
                f"{feature}_pct"
            ] = round(
                count / len(group) * 100,
                2
            )

        decade_rows.append(row)

    # ========================================================
    # SAVE DECADE TABLE
    # ========================================================

    decade_file = os.path.join(
        TABLE_DIR,
        "structural_features_by_decade.csv"
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
    # FORMAT DISTRIBUTION
    # ========================================================

    format_counts = defaultdict(
        lambda: defaultdict(int)
    )

    for row in processed:

        format_counts[
            row["decade"]
        ][
            row["format_class"]
        ] += 1

    format_rows = []

    formats = [
        "Modern Structured",
        "Structured",
        "Minimal Structure",
        "Unknown"
    ]

    for decade in sorted(format_counts):

        total = sum(
            format_counts[decade].values()
        )

        row = {
            "decade": decade,
            "total": total
        }

        for fmt in formats:

            count = format_counts[
                decade
            ].get(fmt, 0)

            row[
                fmt.lower()
                .replace(" ", "_")
            ] = round(
                count / total * 100,
                2
            )

        format_rows.append(row)

    format_file = os.path.join(
        TABLE_DIR,
        "format_distribution_by_decade.csv"
    )

    with open(
        format_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                format_rows[0].keys()
            )
        )

        writer.writeheader()
        writer.writerows(format_rows)

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n--- STRUCTURAL MARKERS BY DECADE ---\n")

    for row in decade_rows:

        print(
            f"{row['decade']:6} | "
            f"N={row['judgment_count']:5} | "
            f"SCI={row['mean_sci']:5.2f} | "
            f"CORAM={row['coram_pct']:6.2f}% | "
            f"JUDGMENT={row['judgment_pct']:6.2f}% | "
            f"ORDER={row['order_pct']:6.2f}% | "
            f"ISSUES={row['issues_pct']:6.2f}% | "
            f"HELD={row['held_pct']:6.2f}% | "
            f"Numbered={row['numbered_paragraphs_pct']:6.2f}%"
        )

    print("\n--- FORMAT DISTRIBUTION ---\n")

    for row in format_rows:

        print(
            f"{row['decade']:6} | "
            f"Modern={row['modern_structured']:6.2f}% | "
            f"Structured={row['structured']:6.2f}% | "
            f"Minimal={row['minimal_structure']:6.2f}% | "
            f"Unknown={row['unknown']:6.2f}%"
        )

    # ========================================================
    # FIGURE 1 — SCI BY DECADE
    # ========================================================

    decades = [
        row["decade"]
        for row in decade_rows
    ]

    sci_values = [
        row["mean_sci"]
        for row in decade_rows
    ]

    plt.figure(figsize=(11, 6))

    plt.plot(
        decades,
        sci_values,
        marker="o"
    )

    plt.title(
        "Structural Complexity Index by Decade"
    )

    plt.xlabel("Decade")
    plt.ylabel("Mean SCI")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "structural_complexity_by_decade.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # FIGURE 2 — STRUCTURAL MARKERS
    # ========================================================

    important_features = [
        "coram_pct",
        "judgment_pct",
        "order_pct",
        "issues_pct",
        "held_pct",
        "numbered_paragraphs_pct"
    ]

    labels = [
        "CORAM",
        "JUDGMENT",
        "ORDER",
        "ISSUES",
        "HELD",
        "Numbered Paragraphs"
    ]

    plt.figure(figsize=(12, 7))

    for feature, label in zip(
        important_features,
        labels
    ):

        values = [
            row[feature]
            for row in decade_rows
        ]

        plt.plot(
            decades,
            values,
            marker="o",
            label=label
        )

    plt.title(
        "Prevalence of Structural Markers by Decade"
    )

    plt.xlabel("Decade")
    plt.ylabel("Percentage of Judgments")

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "structural_markers_by_decade.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # FIGURE 3 — FORMAT DISTRIBUTION
    # ========================================================

    x = list(range(len(format_rows)))

    width = 0.7

    bottom = [0] * len(x)

    for fmt in formats:

        key = fmt.lower().replace(
            " ",
            "_"
        )

        values = [
            row[key]
            for row in format_rows
        ]

        plt.bar(
            x,
            values,
            width,
            bottom=bottom,
            label=fmt
        )

        bottom = [
            b + v
            for b, v in zip(
                bottom,
                values
            )
        ]

    plt.xticks(
        x,
        decades
    )

    plt.ylabel(
        "Percentage of Judgments"
    )

    plt.xlabel("Decade")

    plt.title(
        "Judgment Format Distribution by Decade"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "format_distribution_by_decade.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 75)
    print("STEP 4 COMPLETE")
    print("=" * 75)

    print("\nTables:")
    print(feature_file)
    print(decade_file)
    print(format_file)

    print("\nFigures:")
    print("structural_complexity_by_decade.png")
    print("structural_markers_by_decade.png")
    print("format_distribution_by_decade.png")


if __name__ == "__main__":
    main()