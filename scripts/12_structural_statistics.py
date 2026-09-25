import json
import os
import csv
import re

from collections import defaultdict

from scipy.stats import kruskal, spearmanr


INPUT_FILE = "data/processed/judgments_processed.jsonl"

TABLE_DIR = "outputs/tables"

os.makedirs(TABLE_DIR, exist_ok=True)


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
# HELPERS
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
        p for p in re.split(
            r"\n\s*\n",
            str(text)
        )
        if p.strip()
    ])


# ============================================================
# STRUCTURAL MARKERS
# ============================================================

MARKERS = {

    "coram":
        r"\bCORAM\b",

    "issues":
        r"\bISSUES?\b",

    "held":
        r"\bHELD\b",

    "numbered_paragraphs":
        r"(?m)^\s*\d{1,4}[\.\)]\s+",

    "judgment_by":
        r"\bJUDGMENT\s+BY\b",

    "page_markers":
        r"\bPage\s+\d+\s+of\s+\d+\b"
}


# ============================================================
# MAIN
# ============================================================

def main():

    records = load_data()

    print("=" * 75)
    print("PHASE 2 - STEP 4C: STATISTICAL VALIDATION")
    print("=" * 75)

    rows = []

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

        if words == 0:
            continue

        paragraphs = paragraph_count(text)

        marker_counts = {}

        for name, pattern in MARKERS.items():

            count = len(
                re.findall(
                    pattern,
                    text,
                    flags=re.IGNORECASE
                )
            )

            marker_counts[name] = count

        # Project-defined SCI
        sci = 0

        weights = {
            "coram": 1,
            "issues": 2,
            "held": 2,
            "numbered_paragraphs": 2,
            "judgment_by": 2,
            "page_markers": 1
        }

        for feature, weight in weights.items():

            if marker_counts[feature] > 0:
                sci += weight

        row = {
            "year": year,
            "decade": get_decade(year),

            "word_count": words,

            "paragraph_density":
                paragraphs / words * 1000,

            "sci": sci
        }

        for feature in MARKERS:

            row[
                feature + "_density"
            ] = (
                marker_counts[feature]
                / words
                * 1000
            )

        rows.append(row)

    # ========================================================
    # GROUP BY DECADE
    # ========================================================

    groups = defaultdict(list)

    for row in rows:
        groups[row["decade"]].append(row)

    decades = sorted(groups.keys())

    # ========================================================
    # KRUSKAL-WALLIS
    # ========================================================

    variables = [
        "paragraph_density",
        "sci",
        "coram_density",
        "issues_density",
        "held_density",
        "numbered_paragraphs_density",
        "judgment_by_density",
        "page_markers_density"
    ]

    results = []

    print("\n--- KRUSKAL-WALLIS TESTS ---\n")

    for variable in variables:

        samples = []

        for decade in decades:

            values = [
                row[variable]
                for row in groups[decade]
            ]

            samples.append(values)

        statistic, p_value = kruskal(
            *samples
        )

        result = {
            "variable": variable,
            "kruskal_h": statistic,
            "p_value": p_value,
            "significant_at_0.05":
                p_value < 0.05
        }

        results.append(result)

        print(
            f"{variable:35} "
            f"H={statistic:10.3f} "
            f"p={p_value:.6g} "
            f"{'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'}"
        )

    # ========================================================
    # SPEARMAN YEAR CORRELATIONS
    # ========================================================

    print("\n--- SPEARMAN CORRELATION WITH YEAR ---\n")

    years = [
        row["year"]
        for row in rows
    ]

    correlation_results = []

    for variable in variables:

        values = [
            row[variable]
            for row in rows
        ]

        rho, p_value = spearmanr(
            years,
            values
        )

        result = {
            "variable": variable,
            "spearman_rho": rho,
            "p_value": p_value,
            "significant_at_0.05":
                p_value < 0.05
        }

        correlation_results.append(result)

        direction = (
            "positive"
            if rho > 0
            else "negative"
            if rho < 0
            else "none"
        )

        print(
            f"{variable:35} "
            f"rho={rho:8.4f} "
            f"p={p_value:.6g} "
            f"({direction})"
        )

    # ========================================================
    # SAVE KRUSKAL-WALLIS RESULTS
    # ========================================================

    kw_file = os.path.join(
        TABLE_DIR,
        "structural_kruskal_wallis.csv"
    )

    with open(
        kw_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variable",
                "kruskal_h",
                "p_value",
                "significant_at_0.05"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    # ========================================================
    # SAVE SPEARMAN RESULTS
    # ========================================================

    spearman_file = os.path.join(
        TABLE_DIR,
        "structural_spearman_correlations.csv"
    )

    with open(
        spearman_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variable",
                "spearman_rho",
                "p_value",
                "significant_at_0.05"
            ]
        )

        writer.writeheader()
        writer.writerows(
            correlation_results
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 75)
    print("STEP 4C COMPLETE")
    print("=" * 75)

    print("\nSaved:")
    print(kw_file)
    print(spearman_file)

    print("\nInterpretation rule:")
    print(
        "p < 0.05 indicates statistically significant "
        "differences/association."
    )


if __name__ == "__main__":
    main()