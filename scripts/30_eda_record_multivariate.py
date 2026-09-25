from __future__ import annotations

import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ============================================================
# PATHS
# ============================================================

INPUT = Path("data/processed/judgments_processed.jsonl")

STRUCTURAL_FILE = Path(
    "outputs/tables/structural_features.csv"
)

DRIFT_FILE = Path(
    "outputs/tables/structural_drift_record_level.csv"
)

LEGAL_REF_FILE = Path(
    "outputs/tables/legal_reference_analysis_record_level.csv"
)

TABLE_DIR = Path("outputs/tables")
FIGURE_DIR = Path("outputs/figures")
REPORT_DIR = Path("outputs/reports")

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD PROCESSED DATA
# ============================================================

def load_processed():

    rows = []

    with INPUT.open("r", encoding="utf-8") as f:

        for line in f:

            if line.strip():
                rows.append(json.loads(line))

    df = pd.DataFrame(rows)

    df["date"] = pd.to_datetime(
        df["judgment_dates"],
        format="%d-%m-%Y",
        errors="coerce",
    )

    df["year"] = df["date"].dt.year

    df["decade"] = (
        (df["year"] // 10) * 10
    )

    return df


# ============================================================
# TEXT FEATURES
# ============================================================

def compute_text_features(df):

    text = (
        df["clean_text"]
        .fillna("")
        .astype(str)
    )

    df["word_count_record"] = (
        text.str.split()
        .str.len()
    )

    df["character_count_record"] = (
        text.str.len()
    )

    df["paragraph_count_record"] = (
        text.str.count(r"\n\s*\n") + 1
    )

    # Sentence estimate consistent with the earlier EDA.
    df["sentence_count_record"] = (
        text.str.count(r"[.!?]+(?=\s|$)")
        .clip(lower=1)
    )

    df["unique_word_count_record"] = text.apply(
        lambda x: len(
            set(
                token.lower()
                for token in x.split()
                if token.strip()
            )
        )
    )

    df["vocabulary_diversity_record"] = (
        df["unique_word_count_record"]
        / df["word_count_record"].replace(0, np.nan)
    )

    df["average_sentence_length_record"] = (
        df["word_count_record"]
        / df["sentence_count_record"].replace(0, np.nan)
    )

    df["paragraph_density_record"] = (
        df["paragraph_count_record"]
        / df["word_count_record"].replace(0, np.nan)
        * 1000
    )

    return df


# ============================================================
# LEGAL REFERENCE FEATURES
# ============================================================

def parse_list(value):

    if not isinstance(value, str):
        return []

    try:

        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return parsed

    except Exception:
        pass

    return []


def compute_reference_features(df):

    df["section_count_record"] = (
        df["sections_cited"]
        .apply(lambda x: len(set(parse_list(x))))
    )

    df["article_count_record"] = (
        df["articles_cited"]
        .apply(lambda x: len(set(parse_list(x))))
    )

    df["sections_per_1000_record"] = (
        df["section_count_record"]
        / df["word_count_record"].replace(0, np.nan)
        * 1000
    )

    df["articles_per_1000_record"] = (
        df["article_count_record"]
        / df["word_count_record"].replace(0, np.nan)
        * 1000
    )

    return df


# ============================================================
# STRUCTURAL FEATURES
# ============================================================

def merge_structural_features(df):

    structural = pd.read_csv(
        STRUCTURAL_FILE
    )

    # Keep only reliable binary structural indicators.
    structural_columns = [
        "diary_no",
        "coram",
        "judgment",
        "order",
        "bench",
        "petitioner",
        "respondent",
        "appellant",
        "facts",
        "issues",
        "held",
        "conclusion",
        "reasons",
        "judgment_by",
        "case_number",
        "numbered_paragraphs",
        "page_markers",
        "supreme_court",
    ]

    available = [
        column
        for column in structural_columns
        if column in structural.columns
    ]

    structural = structural[available].copy()

    df = df.merge(
        structural,
        on="diary_no",
        how="left",
        validate="one_to_one",
    )

    return df


# ============================================================
# SPEARMAN CORRELATION
# ============================================================

def calculate_correlations(df):

    variables = [
        "word_count_record",
        "paragraph_count_record",
        "paragraph_density_record",
        "vocabulary_diversity_record",
        "average_sentence_length_record",
        "section_count_record",
        "article_count_record",
        "sections_per_1000_record",
        "articles_per_1000_record",
    ]

    variables = [
        v
        for v in variables
        if v in df.columns
    ]

    rows = []

    for i, x in enumerate(variables):

        for y in variables[i + 1:]:

            subset = df[
                [x, y]
            ].dropna()

            if len(subset) < 30:
                continue

            rho, p = spearmanr(
                subset[x],
                subset[y],
            )

            rows.append(
                {
                    "variable_x": x,
                    "variable_y": y,
                    "n": len(subset),
                    "spearman_rho": rho,
                    "p_value": p,
                    "absolute_rho": abs(rho),
                }
            )

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "absolute_rho",
        ascending=False,
    )

    return result


# ============================================================
# CORRELATION MATRIX
# ============================================================

def create_correlation_matrix(df):

    variables = [
        "word_count_record",
        "paragraph_count_record",
        "paragraph_density_record",
        "vocabulary_diversity_record",
        "average_sentence_length_record",
        "section_count_record",
        "article_count_record",
        "sections_per_1000_record",
        "articles_per_1000_record",
    ]

    variables = [
        v
        for v in variables
        if v in df.columns
    ]

    matrix = df[
        variables
    ].corr(
        method="spearman"
    )

    matrix.to_csv(
        TABLE_DIR
        / "record_level_spearman_matrix.csv"
    )

    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

    plt.figure(
        figsize=(11, 9)
    )

    plt.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
    )

    plt.colorbar(
        label="Spearman correlation"
    )

    plt.xticks(
        range(len(variables)),
        variables,
        rotation=75,
        ha="right",
    )

    plt.yticks(
        range(len(variables)),
        variables,
    )

    plt.title(
        "Record-Level Relationships Between Judgment Characteristics"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "record_level_correlation_matrix.png",
        dpi=220,
    )

    plt.close()


# ============================================================
# STRUCTURAL MARKER PREVALENCE BY DECADE
# ============================================================

def structural_marker_analysis(df):

    markers = [
        "coram",
        "judgment",
        "order",
        "bench",
        "petitioner",
        "respondent",
        "appellant",
        "facts",
        "issues",
        "held",
        "conclusion",
        "reasons",
        "judgment_by",
        "case_number",
        "numbered_paragraphs",
        "page_markers",
        "supreme_court",
    ]

    markers = [
        m
        for m in markers
        if m in df.columns
    ]

    rows = []

    for decade, group in df.groupby(
        "decade",
        dropna=True,
    ):

        row = {
            "decade": int(decade),
            "judgments": len(group),
        }

        for marker in markers:

            values = group[marker]

            row[
                f"{marker}_percentage"
            ] = (
                pd.to_numeric(
                    values,
                    errors="coerce",
                )
                .mean()
                * 100
            )

        rows.append(row)

    result = pd.DataFrame(rows)

    result.to_csv(
        TABLE_DIR
        / "record_level_structural_marker_prevalence.csv",
        index=False,
    )

    return result


# ============================================================
# REPORT
# ============================================================

def create_report(
    df,
    correlations,
):

    report = []

    report.append(
        "# Record-Level Multivariate EDA\n"
    )

    report.append(
        f"Records analysed: **{len(df):,}**\n"
    )

    report.append(
        "This analysis examines relationships between "
        "document length, paragraph structure, vocabulary "
        "diversity, sentence characteristics, and legal-reference "
        "density at the individual judgment level.\n"
    )

    report.append(
        "## Important methodological note\n"
    )

    report.append(
        "Spearman correlations are used because the document "
        "features are highly non-normal and contain extreme "
        "outliers. Correlation indicates association, not causation.\n"
    )

    report.append(
        "## Strongest record-level associations\n"
    )

    for _, row in correlations.head(15).iterrows():

        report.append(
            f"- `{row['variable_x']}` vs "
            f"`{row['variable_y']}`: "
            f"rho = {row['spearman_rho']:.3f}, "
            f"p = {row['p_value']:.3e}, "
            f"n = {int(row['n']):,}"
        )

    report.append(
        "\n## Interpretation cautions\n"
    )

    report.append(
        "- Statistical significance should not be interpreted "
        "as practical importance."
    )

    report.append(
        "- Very large samples can produce small p-values for "
        "modest associations."
    )

    report.append(
        "- Extreme judgments can strongly influence raw "
        "relationships, so robust statistics are preferred."
    )

    report.append(
        "- These are observational relationships and do not "
        "establish causation."
    )

    (REPORT_DIR / "record_level_multivariate_eda.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading processed judgments...")

    df = load_processed()

    print(
        f"Loaded {len(df):,} judgments."
    )

    print("Computing text features...")

    df = compute_text_features(df)

    print("Computing legal-reference features...")

    df = compute_reference_features(df)

    print("Merging structural features...")

    df = merge_structural_features(df)

    print("Calculating record-level correlations...")

    correlations = calculate_correlations(
        df
    )

    correlations.to_csv(
        TABLE_DIR
        / "record_level_spearman_correlations.csv",
        index=False,
    )

    print("Creating correlation matrix...")

    create_correlation_matrix(
        df
    )

    print("Analysing structural markers...")

    structural_marker_analysis(
        df
    )

    print("Creating report...")

    create_report(
        df,
        correlations,
    )

    print()
    print(
        "Record-level multivariate EDA complete."
    )

    print(
        "Top associations:"
    )

    print(
        correlations.head(10).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()