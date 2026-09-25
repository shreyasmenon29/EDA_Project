import json
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


BASE = Path(__file__).resolve().parent.parent

DATA = BASE / "data" / "processed" / "judgments_processed.jsonl"
TOPIC_FILE = BASE / "outputs" / "tables" / "topic_prevalence_by_decade.csv"

OUT_TABLES = BASE / "outputs" / "tables"
OUT_FIGURES = BASE / "outputs" / "figures"

OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGURES.mkdir(parents=True, exist_ok=True)


print("=" * 70)
print("STEP 10 - TOPIC × STRUCTURE × LEGAL REFERENCES")
print("=" * 70)


# ---------------------------------------------------------
# 1. Load topic prevalence
# ---------------------------------------------------------

topic_df = pd.read_csv(TOPIC_FILE)

# The topic prevalence CSV stores decade as its index
if "decade_label" not in topic_df.columns:
    first_col = topic_df.columns[0]
    topic_df = topic_df.rename(columns={first_col: "decade_label"})

topic_df["decade_label"] = topic_df["decade_label"].astype(str).str.strip()

print(f"\nTopic prevalence loaded: {len(topic_df)} decades")


# ---------------------------------------------------------
# 2. Load processed judgments
# ---------------------------------------------------------

records = []

with open(DATA, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

df = pd.DataFrame(records)

print(f"Judgments loaded: {len(df):,}")


# ---------------------------------------------------------
# 3. Basic temporal variables
# ---------------------------------------------------------

df["date"] = pd.to_datetime(
    df["judgment_dates"],
    format="%d-%m-%Y",
    errors="coerce"
)

df["year"] = df["date"].dt.year

df["decade"] = (df["year"] // 10) * 10

df["decade_label"] = (
    df["decade"].astype(int).astype(str) + "s"
)


# ---------------------------------------------------------
# 4. Text statistics
# ---------------------------------------------------------

text = df["clean_text"].fillna("")

df["word_count"] = text.str.split().str.len()

df["paragraph_count"] = text.str.count(r"\n\s*\n") + 1

df["sentence_count"] = (
    text.str.count(r"[.!?]+")
    .clip(lower=1)
)

df["vocabulary_diversity"] = (
    text.apply(
        lambda x:
        len(set(re.findall(r"\b[a-zA-Z]+\b", x.lower())))
        / max(len(re.findall(r"\b[a-zA-Z]+\b", x.lower())), 1)
    )
)

df["paragraph_density"] = (
    df["paragraph_count"] / df["word_count"] * 1000
)

df["sentence_length"] = (
    df["word_count"] / df["sentence_count"]
)


# ---------------------------------------------------------
# 5. Legal-reference extraction
# ---------------------------------------------------------

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


def normalize_section(x):
    x = str(x).strip().upper()

    match = re.search(r"S\.\s*([0-9]+[A-Z]?)", x)

    if match:
        return "S." + match.group(1)

    return None


def normalize_article(x):
    x = str(x).strip().upper()

    match = re.search(r"ART\.?\s*([0-9]+[A-Z]?)", x)

    if match:
        return "Art." + match.group(1)

    return None


df["sections"] = df["sections_cited"].apply(parse_list)
df["articles"] = df["articles_cited"].apply(parse_list)

df["section_count"] = df["sections"].apply(
    lambda xs: len(
        set(
            filter(
                None,
                [normalize_section(x) for x in xs]
            )
        )
    )
)

df["article_count"] = df["articles"].apply(
    lambda xs: len(
        set(
            filter(
                None,
                [normalize_article(x) for x in xs]
            )
        )
    )
)

df["sections_per_1000"] = (
    df["section_count"] / df["word_count"] * 1000
)

df["articles_per_1000"] = (
    df["article_count"] / df["word_count"] * 1000
)


# ---------------------------------------------------------
# 6. Aggregate structural/legal statistics by decade
# ---------------------------------------------------------

decade_stats = (
    df.groupby("decade_label")
    .agg(
        documents=("diary_no", "count"),
        median_words=("word_count", "median"),
        median_paragraphs=("paragraph_count", "median"),
        median_paragraph_density=("paragraph_density", "median"),
        median_vocab_diversity=("vocabulary_diversity", "median"),
        median_sentence_length=("sentence_length", "median"),
        median_sections=("section_count", "median"),
        median_articles=("article_count", "median"),
        median_sections_per_1000=("sections_per_1000", "median"),
        median_articles_per_1000=("articles_per_1000", "median"),
    )
    .reset_index()
)


# ---------------------------------------------------------
# 7. Merge with topic prevalence
# ---------------------------------------------------------

merged = pd.merge(
    topic_df,
    decade_stats,
    on="decade_label",
    how="inner"
)

print("\nMerged topic + structural dataset:")
print(merged.to_string(index=False))


merged.to_csv(
    OUT_TABLES / "topic_structure_legal_references_by_decade.csv",
    index=False
)


# ---------------------------------------------------------
# 8. Topic × structural correlations
# ---------------------------------------------------------

topic_columns = [
    c for c in topic_df.columns
    if c.startswith("topic_")
]

structural_columns = [
    "median_words",
    "median_paragraphs",
    "median_paragraph_density",
    "median_vocab_diversity",
    "median_sentence_length",
    "median_sections",
    "median_articles",
    "median_sections_per_1000",
    "median_articles_per_1000",
]


results = []

for topic in topic_columns:

    for variable in structural_columns:

        x = merged[topic]
        y = merged[variable]

        rho, p = spearmanr(x, y)

        results.append({
            "topic": topic,
            "variable": variable,
            "spearman_rho": rho,
            "p_value": p
        })


correlations = pd.DataFrame(results)

correlations["abs_rho"] = correlations["spearman_rho"].abs()

correlations = correlations.sort_values(
    "abs_rho",
    ascending=False
)

correlations.to_csv(
    OUT_TABLES / "topic_structure_correlations.csv",
    index=False
)


# ---------------------------------------------------------
# 9. Print strongest relationships
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("STRONGEST TOPIC × STRUCTURE RELATIONSHIPS")
print("=" * 70)

print(
    correlations.head(25).to_string(index=False)
)


# ---------------------------------------------------------
# 10. Dominant topic by decade
# ---------------------------------------------------------

dominant_rows = []

for _, row in merged.iterrows():

    topic_values = row[topic_columns]

    dominant_topic = topic_values.idxmax()

    dominant_rows.append({
        "decade": row["decade_label"],
        "dominant_topic": dominant_topic,
        "prevalence": topic_values.max(),
        "median_words": row["median_words"],
        "median_paragraph_density":
            row["median_paragraph_density"],
        "median_vocab_diversity":
            row["median_vocab_diversity"],
        "median_sections_per_1000":
            row["median_sections_per_1000"],
        "median_articles_per_1000":
            row["median_articles_per_1000"],
    })


dominant_df = pd.DataFrame(dominant_rows)

dominant_df.to_csv(
    OUT_TABLES / "dominant_topic_structural_profile.csv",
    index=False
)


print("\n" + "=" * 70)
print("DOMINANT TOPIC STRUCTURAL PROFILE")
print("=" * 70)

print(dominant_df.to_string(index=False))


# ---------------------------------------------------------
# 11. Heatmap
# ---------------------------------------------------------

try:

    import matplotlib.pyplot as plt
    import seaborn as sns

    heatmap_data = correlations.pivot(
        index="topic",
        columns="variable",
        values="spearman_rho"
    )

    plt.figure(figsize=(14, 8))

    sns.heatmap(
        heatmap_data,
        center=0,
        cmap="coolwarm",
        annot=False
    )

    plt.title(
        "Topic Prevalence vs Structural and Legal-Reference Variables"
    )

    plt.tight_layout()

    plt.savefig(
        OUT_FIGURES / "topic_structure_correlation_heatmap.png",
        dpi=300
    )

    plt.close()

    print(
        "\nHeatmap saved:"
        " outputs/figures/topic_structure_correlation_heatmap.png"
    )

except Exception as e:

    print("\nHeatmap skipped:", e)


print("\n" + "=" * 70)
print("STEP 10 COMPLETE")
print("=" * 70)