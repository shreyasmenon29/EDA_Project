import json
import os
import csv
import re
import ast

from collections import Counter, defaultdict

import matplotlib.pyplot as plt
from scipy.stats import spearmanr


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


def is_missing(value):

    if value is None:
        return True

    if isinstance(value, str):

        return value.strip().lower() in {
            "",
            "nan",
            "none",
            "null",
            "[]",
            "{}"
        }

    if isinstance(value, list):

        return len(value) == 0

    return False


# ============================================================
# PARSE LIST-LIKE FIELDS
# ============================================================

def parse_items(value):

    if is_missing(value):
        return []

    # Already a Python list
    if isinstance(value, list):

        return [
            str(x).strip()
            for x in value
            if not is_missing(x)
        ]

    # Dictionary
    if isinstance(value, dict):

        return [
            str(k).strip()
            for k in value.keys()
        ]

    value = str(value).strip()

    # Try JSON
    try:

        parsed = json.loads(value)

        if isinstance(parsed, list):

            return [
                str(x).strip()
                for x in parsed
                if str(x).strip()
            ]

        if isinstance(parsed, dict):

            return [
                str(k).strip()
                for k in parsed.keys()
            ]

    except:
        pass

    # Try Python literal
    try:

        parsed = ast.literal_eval(value)

        if isinstance(parsed, (list, tuple, set)):

            return [
                str(x).strip()
                for x in parsed
                if str(x).strip()
            ]

    except:
        pass

    # Common separators
    parts = re.split(
        r"[;,|]\s*|\n+",
        value
    )

    return [
        p.strip(" []'\"")
        for p in parts
        if p.strip(" []'\"")
    ]


# ============================================================
# CITATION PRESENCE
# ============================================================

def citation_present(record):

    value = record.get("has_citation")

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value != 0

    if isinstance(value, str):

        value = value.strip().lower()

        if value in {
            "true",
            "1",
            "yes",
            "y"
        }:
            return True

        if value in {
            "false",
            "0",
            "no",
            "n",
            "",
            "nan",
            "none",
            "null"
        }:
            return False

    # Fall back to extracted citations
    extracted = record.get(
        "citation_extracted"
    )

    return not is_missing(extracted)


# ============================================================
# MAIN
# ============================================================

def main():

    records = load_data()

    print("=" * 75)
    print("PHASE 2 - STEP 5: CITATION & LEGAL-REFERENCE EDA")
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

        citation = citation_present(
            record
        )

        citations = parse_items(
            record.get(
                "citation_extracted"
            )
        )

        sections = parse_items(
            record.get(
                "sections_cited"
            )
        )

        articles = parse_items(
            record.get(
                "articles_cited"
            )
        )

        # Remove obvious placeholders
        citations = [
            x for x in citations
            if x.lower() not in {
                "nan",
                "none",
                "null"
            }
        ]

        sections = [
            x for x in sections
            if x.lower() not in {
                "nan",
                "none",
                "null"
            }
        ]

        articles = [
            x for x in articles
            if x.lower() not in {
                "nan",
                "none",
                "null"
            }
        ]

        row = {

            "diary_no":
                record.get("diary_no"),

            "year":
                year,

            "decade":
                decade,

            "word_count":
                words,

            "has_citation":
                citation,

            "citation_count":
                len(citations),

            "section_count":
                len(sections),

            "article_count":
                len(articles),

            "citation_density_per_1000_words":
                (
                    len(citations) / words * 1000
                    if words > 0
                    else 0
                ),

            "section_density_per_1000_words":
                (
                    len(sections) / words * 1000
                    if words > 0
                    else 0
                ),

            "article_density_per_1000_words":
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
        "citation_analysis_record_level.csv"
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
    # DECADE SUMMARY
    # ========================================================

    decade_rows = []

    for decade in sorted(
        decade_groups
    ):

        group = decade_groups[
            decade
        ]

        total = len(group)

        citation_percentage = (
            sum(
                x["has_citation"]
                for x in group
            )
            / total
            * 100
        )

        row = {

            "decade":
                decade,

            "judgment_count":
                total,

            "citation_presence_pct":
                round(
                    citation_percentage,
                    2
                ),

            "median_citation_count":
                median([
                    x["citation_count"]
                    for x in group
                ]),

            "median_section_count":
                median([
                    x["section_count"]
                    for x in group
                ]),

            "median_article_count":
                median([
                    x["article_count"]
                    for x in group
                ]),

            "median_citation_density":
                median([
                    x[
                        "citation_density_per_1000_words"
                    ]
                    for x in group
                ]),

            "median_section_density":
                median([
                    x[
                        "section_density_per_1000_words"
                    ]
                    for x in group
                ]),

            "median_article_density":
                median([
                    x[
                        "article_density_per_1000_words"
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
        "citation_analysis_by_decade.csv"
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
    # PRINT DECADE RESULTS
    # ========================================================

    print("\n--- CITATION ANALYSIS BY DECADE ---\n")

    for row in decade_rows:

        print(
            f"{row['decade']:6} | "
            f"N={row['judgment_count']:5} | "
            f"HasCitation={row['citation_presence_pct']:6.2f}% | "
            f"Citations={row['median_citation_count']:6.1f} | "
            f"Sections={row['median_section_count']:5.1f} | "
            f"Articles={row['median_article_count']:5.1f} | "
            f"Cit/1k={row['median_citation_density']:.3f}"
        )

    # ========================================================
    # TOP SECTIONS
    # ========================================================

    print("\n--- TOP 20 CITED SECTIONS ---\n")

    top_sections = section_counter.most_common(20)

    for section, count in top_sections:

        print(
            f"{section:30} {count}"
        )

    # ========================================================
    # TOP ARTICLES
    # ========================================================

    print("\n--- TOP 20 CITED ARTICLES ---\n")

    top_articles = article_counter.most_common(20)

    for article, count in top_articles:

        print(
            f"{article:30} {count}"
        )

    # ========================================================
    # SAVE TOP SECTIONS
    # ========================================================

    sections_file = os.path.join(
        TABLE_DIR,
        "top_cited_sections.csv"
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
            "count"
        ])

        writer.writerows(
            top_sections
        )

    # ========================================================
    # SAVE TOP ARTICLES
    # ========================================================

    articles_file = os.path.join(
        TABLE_DIR,
        "top_cited_articles.csv"
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
            "count"
        ])

        writer.writerows(
            top_articles
        )

    # ========================================================
    # SPEARMAN CORRELATIONS
    # ========================================================

    years = [
        row["year"]
        for row in rows
    ]

    correlation_variables = [
        "citation_count",
        "citation_density_per_1000_words",
        "section_count",
        "article_count"
    ]

    correlation_rows = []

    print(
        "\n--- CORRELATION WITH YEAR ---\n"
    )

    for variable in correlation_variables:

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
            f"{variable:40} "
            f"rho={rho:8.4f} "
            f"p={p:.6g}"
        )

    correlation_file = os.path.join(
        TABLE_DIR,
        "citation_temporal_correlations.csv"
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
    # FIGURE 1 — CITATION PRESENCE
    # ========================================================

    decades = [
        row["decade"]
        for row in decade_rows
    ]

    citation_presence = [
        row[
            "citation_presence_pct"
        ]
        for row in decade_rows
    ]

    plt.figure(figsize=(11, 6))

    plt.plot(
        decades,
        citation_presence,
        marker="o"
    )

    plt.title(
        "Judgments Containing Citations by Decade"
    )

    plt.xlabel("Decade")
    plt.ylabel(
        "Judgments with Citations (%)"
    )

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "citation_presence_by_decade.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # FIGURE 2 — CITATION DENSITY
    # ========================================================

    citation_density = [
        row[
            "median_citation_density"
        ]
        for row in decade_rows
    ]

    plt.figure(figsize=(11, 6))

    plt.plot(
        decades,
        citation_density,
        marker="o"
    )

    plt.title(
        "Median Citation Density by Decade"
    )

    plt.xlabel("Decade")

    plt.ylabel(
        "Citations per 1,000 Words"
    )

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "citation_density_by_decade.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # FIGURE 3 — SECTIONS AND ARTICLES
    # ========================================================

    section_values = [
        row[
            "median_section_count"
        ]
        for row in decade_rows
    ]

    article_values = [
        row[
            "median_article_count"
        ]
        for row in decade_rows
    ]

    plt.figure(figsize=(11, 6))

    plt.plot(
        decades,
        section_values,
        marker="o",
        label="Sections"
    )

    plt.plot(
        decades,
        article_values,
        marker="o",
        label="Articles"
    )

    plt.title(
        "Median Legal References per Judgment"
    )

    plt.xlabel("Decade")

    plt.ylabel(
        "Median Number of References"
    )

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "legal_references_by_decade.png"
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
    print("citation_presence_by_decade.png")
    print("citation_density_by_decade.png")
    print("legal_references_by_decade.png")


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