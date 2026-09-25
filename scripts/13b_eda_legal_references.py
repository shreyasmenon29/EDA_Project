import json
import os
import csv
import ast
import re
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
from scipy.stats import spearmanr


INPUT_FILE = "data/processed/judgments_processed.jsonl"

TABLE_DIR = "outputs/tables"
FIGURE_DIR = "outputs/figures"

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


# ============================================================
# LOAD
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
        return int(str(date_value).split("-")[2])
    except:
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


def median(values):

    if not values:
        return 0

    values = sorted(values)

    n = len(values)

    if n % 2 == 0:

        return (
            values[n // 2 - 1]
            + values[n // 2]
        ) / 2

    return values[n // 2]


# ============================================================
# PARSE LIST STRING
# ============================================================

def parse_list(value):

    if value is None:
        return []

    if isinstance(value, list):
        return value

    value = str(value).strip()

    if value in {
        "",
        "[]",
        "nan",
        "None",
        "null"
    }:
        return []

    try:

        parsed = ast.literal_eval(value)

        if isinstance(parsed, (list, tuple, set)):
            return list(parsed)

    except:
        pass

    return []


# ============================================================
# NORMALIZE SECTION
# ============================================================

def normalize_section(value):

    value = str(value)

    value = value.replace(
        "\\n",
        " "
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    # Extract the section identifier.
    # Example:
    # "S.302 Indian Penal Code"
    # -> "S.302"

    match = re.search(
        r"\bS\.?\s*([0-9]+[A-Za-z]?)",
        value,
        flags=re.IGNORECASE
    )

    if match:

        return "S." + match.group(1)

    return value


# ============================================================
# NORMALIZE ARTICLE
# ============================================================

def normalize_article(value):

    value = str(value)

    value = value.replace(
        "\\n",
        " "
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    match = re.search(
        r"\bArt\.?\s*([0-9]+[A-Za-z]?)",
        value,
        flags=re.IGNORECASE
    )

    if match:

        return "Art." + match.group(1)

    return value


# ============================================================
# MAIN
# ============================================================

def main():

    records = load_data()

    print("=" * 75)
    print("PHASE 2 - STEP 5: LEGAL REFERENCE EDA")
    print("=" * 75)

    rows = []

    section_counter = Counter()
    article_counter = Counter()

    decade_groups = defaultdict(list)

    for record in records:

        year = get_year(
            record.get("judgment_dates")
        )

        if year is None:
            continue

        decade = get_decade(year)

        text = record.get(
            "clean_text",
            ""
        )

        words = word_count(text)

        # ----------------------------------------------------
        # Reported judgment citation availability
        # ----------------------------------------------------

        has_citation = bool(
            record.get(
                "has_citation",
                False
            )
        )

        # ----------------------------------------------------
        # Statutory sections
        # ----------------------------------------------------

        raw_sections = parse_list(
            record.get(
                "sections_cited"
            )
        )

        sections = set()

        for item in raw_sections:

            normalized = normalize_section(
                item
            )

            if normalized:
                sections.add(normalized)

        # ----------------------------------------------------
        # Constitutional articles
        # ----------------------------------------------------

        raw_articles = parse_list(
            record.get(
                "articles_cited"
            )
        )

        articles = set()

        for item in raw_articles:

            normalized = normalize_article(
                item
            )

            if normalized:
                articles.add(normalized)

        # ----------------------------------------------------
        # Record
        # ----------------------------------------------------

        row = {

            "diary_no":
                record.get("diary_no"),

            "year":
                year,

            "decade":
                decade,

            "word_count":
                words,

            "has_reported_citation":
                has_citation,

            "section_count":
                len(sections),

            "article_count":
                len(articles),

            "sections_per_1000_words":
                (
                    len(sections) / words * 1000
                    if words > 0
                    else 0
                ),

            "articles_per_1000_words":
                (
                    len(articles) / words * 1000
                    if words > 0
                    else 0
                )
        }

        rows.append(row)

        decade_groups[
            decade
        ].append(row)

        section_counter.update(
            sections
        )

        article_counter.update(
            articles
        )

    # ========================================================
    # RECORD LEVEL TABLE
    # ========================================================

    record_file = os.path.join(
        TABLE_DIR,
        "legal_reference_analysis_record_level.csv"
    )

    with open(
        record_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            )
        )

        writer.writeheader()
        writer.writerows(rows)

    # ========================================================
    # DECADE ANALYSIS
    # ========================================================

    decade_rows = []

    for decade in sorted(decade_groups):

        group = decade_groups[decade]

        n = len(group)

        row = {

            "decade":
                decade,

            "judgment_count":
                n,

            "reported_citation_pct":
                round(
                    sum(
                        x[
                            "has_reported_citation"
                        ]
                        for x in group
                    )
                    / n
                    * 100,
                    2
                ),

            "median_sections":
                median([
                    x["section_count"]
                    for x in group
                ]),

            "median_articles":
                median([
                    x["article_count"]
                    for x in group
                ]),

            "median_sections_per_1000_words":
                median([
                    x[
                        "sections_per_1000_words"
                    ]
                    for x in group
                ]),

            "median_articles_per_1000_words":
                median([
                    x[
                        "articles_per_1000_words"
                    ]
                    for x in group
                ])
        }

        decade_rows.append(row)

    # ========================================================
    # SAVE DECADE TABLE
    # ========================================================

    decade_file = os.path.join(
        TABLE_DIR,
        "legal_reference_analysis_by_decade.csv"
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
        writer.writerows(
            decade_rows
        )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n--- LEGAL REFERENCES BY DECADE ---\n")

    for row in decade_rows:

        print(
            f"{row['decade']:6} | "
            f"N={row['judgment_count']:5} | "
            f"ReportedCitation="
            f"{row['reported_citation_pct']:6.2f}% | "
            f"Sections="
            f"{row['median_sections']:5.1f} | "
            f"Articles="
            f"{row['median_articles']:5.1f} | "
            f"Sec/1k="
            f"{row['median_sections_per_1000_words']:.3f} | "
            f"Art/1k="
            f"{row['median_articles_per_1000_words']:.3f}"
        )

    # ========================================================
    # TOP SECTIONS
    # ========================================================

    top_sections = (
        section_counter
        .most_common(30)
    )

    print("\n--- TOP 30 STATUTORY SECTIONS ---\n")

    for section, count in top_sections:

        print(
            f"{section:12} {count}"
        )

    # ========================================================
    # TOP ARTICLES
    # ========================================================

    top_articles = (
        article_counter
        .most_common(30)
    )

    print("\n--- TOP 30 CONSTITUTIONAL ARTICLES ---\n")

    for article, count in top_articles:

        print(
            f"{article:12} {count}"
        )

    # ========================================================
    # SAVE TOP SECTIONS
    # ========================================================

    sections_file = os.path.join(
        TABLE_DIR,
        "top_statutory_sections.csv"
    )

    with open(
        sections_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "section",
            "judgment_count"
        ])

        writer.writerows(
            top_sections
        )

    # ========================================================
    # SAVE TOP ARTICLES
    # ========================================================

    articles_file = os.path.join(
        TABLE_DIR,
        "top_constitutional_articles.csv"
    )

    with open(
        articles_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "article",
            "judgment_count"
        ])

        writer.writerows(
            top_articles
        )

    # ========================================================
    # TEMPORAL CORRELATIONS
    # ========================================================

    years = [
        row["year"]
        for row in rows
    ]

    variables = [
        "section_count",
        "article_count",
        "sections_per_1000_words",
        "articles_per_1000_words"
    ]

    correlation_rows = []

    print("\n--- TEMPORAL CORRELATIONS ---\n")

    for variable in variables:

        values = [
            row[variable]
            for row in rows
        ]

        rho, p = spearmanr(
            years,
            values
        )

        correlation_rows.append({

            "variable":
                variable,

            "spearman_rho":
                rho,

            "p_value":
                p,

            "significant_at_0.05":
                p < 0.05
        })

        print(
            f"{variable:35} "
            f"rho={rho:8.4f} "
            f"p={p:.6g}"
        )

    correlation_file = os.path.join(
        TABLE_DIR,
        "legal_reference_temporal_correlations.csv"
    )

    with open(
        correlation_file,
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
            correlation_rows
        )

    # ========================================================
    # FIGURE 1 — SECTION REFERENCES
    # ========================================================

    decades = [
        row["decade"]
        for row in decade_rows
    ]

    sections = [
        row["median_sections"]
        for row in decade_rows
    ]

    articles = [
        row["median_articles"]
        for row in decade_rows
    ]

    plt.figure(figsize=(11, 6))

    plt.plot(
        decades,
        sections,
        marker="o",
        label="Statutory Sections"
    )

    plt.plot(
        decades,
        articles,
        marker="o",
        label="Constitutional Articles"
    )

    plt.title(
        "Legal Reference Counts by Decade"
    )

    plt.xlabel("Decade")
    plt.ylabel(
        "Median Distinct References per Judgment"
    )

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "legal_reference_counts_by_decade.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # FIGURE 2 — NORMALIZED REFERENCES
    # ========================================================

    section_density = [
        row[
            "median_sections_per_1000_words"
        ]
        for row in decade_rows
    ]

    article_density = [
        row[
            "median_articles_per_1000_words"
        ]
        for row in decade_rows
    ]

    plt.figure(figsize=(11, 6))

    plt.plot(
        decades,
        section_density,
        marker="o",
        label="Sections"
    )

    plt.plot(
        decades,
        article_density,
        marker="o",
        label="Articles"
    )

    plt.title(
        "Normalized Legal Reference Density by Decade"
    )

    plt.xlabel("Decade")

    plt.ylabel(
        "Median Distinct References per 1,000 Words"
    )

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "legal_reference_density_by_decade.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 75)
    print("STEP 5 COMPLETE")
    print("=" * 75)

    print("\nTables:")
    print(record_file)
    print(decade_file)
    print(sections_file)
    print(articles_file)
    print(correlation_file)

    print("\nFigures:")
    print("legal_reference_counts_by_decade.png")
    print("legal_reference_density_by_decade.png")


if __name__ == "__main__":
    main()