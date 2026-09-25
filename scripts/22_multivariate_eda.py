import json
import re
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


BASE = Path(__file__).resolve().parent.parent

DATA = BASE / "data" / "processed" / "judgments_processed.jsonl"

OUT_TABLES = BASE / "outputs" / "tables"
OUT_FIGURES = BASE / "outputs" / "figures"

OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGURES.mkdir(parents=True, exist_ok=True)


print("=" * 70)
print("STEP 11 - MULTIVARIATE EDA / PCA")
print("=" * 70)


# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------

records = []

with open(DATA, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

df = pd.DataFrame(records)

print(f"Documents: {len(df):,}")


# ---------------------------------------------------------
# 2. Dates / decades
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
# 3. Text statistics
# ---------------------------------------------------------

text = df["clean_text"].fillna("")

df["word_count"] = text.str.split().str.len()

df["paragraph_count"] = (
    text.str.count(r"\n\s*\n") + 1
)

df["sentence_count"] = (
    text.str.count(r"[.!?]+").clip(lower=1)
)

df["paragraph_density"] = (
    df["paragraph_count"] /
    df["word_count"] * 1000
)

df["sentence_length"] = (
    df["word_count"] /
    df["sentence_count"]
)


def vocabulary_diversity(x):

    words = re.findall(
        r"\b[a-zA-Z]+\b",
        x.lower()
    )

    if not words:
        return 0

    return len(set(words)) / len(words)


df["vocabulary_diversity"] = text.apply(
    vocabulary_diversity
)


# ---------------------------------------------------------
# 4. Legal-reference statistics
# ---------------------------------------------------------

def parse_list(value):

    if not isinstance(value, str):
        return []

    try:
        value = ast.literal_eval(value)

        if isinstance(value, list):
            return value

    except Exception:
        pass

    return []


def normalize_section(x):

    match = re.search(
        r"S\.\s*([0-9]+[A-Z]?)",
        str(x).upper()
    )

    if match:
        return "S." + match.group(1)

    return None


def normalize_article(x):

    match = re.search(
        r"ART\.?\s*([0-9]+[A-Z]?)",
        str(x).upper()
    )

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
    df["section_count"] /
    df["word_count"] * 1000
)

df["articles_per_1000"] = (
    df["article_count"] /
    df["word_count"] * 1000
)


# ---------------------------------------------------------
# 5. Build multivariate feature matrix
# ---------------------------------------------------------

features = pd.DataFrame({

    "log_word_count":
        np.log1p(df["word_count"]),

    "paragraph_density":
        df["paragraph_density"],

    "vocabulary_diversity":
        df["vocabulary_diversity"],

    "sentence_length":
        df["sentence_length"],

    "sections_per_1000":
        df["sections_per_1000"],

    "articles_per_1000":
        df["articles_per_1000"],

    "paragraph_count":
        np.log1p(df["paragraph_count"])
})


# ---------------------------------------------------------
# 6. Remove invalid values
# ---------------------------------------------------------

features = features.replace(
    [np.inf, -np.inf],
    np.nan
)

valid = features.notna().all(axis=1)

features = features.loc[valid]

metadata = df.loc[valid].copy()

print(f"Valid documents for PCA: {len(features):,}")


# ---------------------------------------------------------
# 7. Standardization
# ---------------------------------------------------------

scaler = StandardScaler()

X = scaler.fit_transform(features)


# ---------------------------------------------------------
# 8. PCA
# ---------------------------------------------------------

pca = PCA()

components = pca.fit_transform(X)


explained = pd.DataFrame({

    "component": [
        f"PC{i+1}"
        for i in range(len(pca.explained_variance_ratio_))
    ],

    "explained_variance_ratio":
        pca.explained_variance_ratio_,

    "cumulative_variance":
        np.cumsum(
            pca.explained_variance_ratio_
        )
})


explained.to_csv(
    OUT_TABLES /
    "pca_explained_variance.csv",
    index=False
)


# ---------------------------------------------------------
# 9. PCA loadings
# ---------------------------------------------------------

loadings = pd.DataFrame(
    pca.components_.T,
    index=features.columns,
    columns=[
        f"PC{i+1}"
        for i in range(
            pca.n_components_
        )
    ]
)

loadings.to_csv(
    OUT_TABLES /
    "pca_loadings.csv"
)


# ---------------------------------------------------------
# 10. Print PCA results
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("PCA EXPLAINED VARIANCE")
print("=" * 70)

print(
    explained.to_string(index=False)
)


print("\n" + "=" * 70)
print("PCA LOADINGS")
print("=" * 70)

print(
    loadings.iloc[:, :4].to_string()
)


# ---------------------------------------------------------
# 11. Document-level PCA scores
# ---------------------------------------------------------

score_df = pd.DataFrame(
    components[:, :4],
    columns=[
        "PC1",
        "PC2",
        "PC3",
        "PC4"
    ]
)

score_df["year"] = metadata["year"].values
score_df["decade"] = metadata["decade_label"].values


# ---------------------------------------------------------
# 12. PCA scores by decade
# ---------------------------------------------------------

decade_pca = (
    score_df
    .groupby("decade")
    [["PC1", "PC2", "PC3", "PC4"]]
    .mean()
    .reset_index()
)

decade_pca.to_csv(
    OUT_TABLES /
    "pca_scores_by_decade.csv",
    index=False
)


print("\n" + "=" * 70)
print("PCA SCORES BY DECADE")
print("=" * 70)

print(
    decade_pca.to_string(index=False)
)


# ---------------------------------------------------------
# 13. PCA vs year
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("PCA DIMENSION vs YEAR")
print("=" * 70)

year_results = []

for pc in ["PC1", "PC2", "PC3", "PC4"]:

    rho, p = spearmanr(
        score_df[pc],
        score_df["year"]
    )

    year_results.append({

        "component": pc,

        "spearman_rho":
            rho,

        "p_value":
            p
    })


year_results = pd.DataFrame(year_results)

print(
    year_results.to_string(index=False)
)

year_results.to_csv(
    OUT_TABLES /
    "pca_year_correlations.csv",
    index=False
)


# ---------------------------------------------------------
# 14. PCA plots
# ---------------------------------------------------------

try:

    import matplotlib.pyplot as plt

    # PC1 explained variance
    plt.figure(figsize=(8, 5))

    plt.bar(
        explained["component"],
        explained["explained_variance_ratio"]
    )

    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("PCA Explained Variance")

    plt.tight_layout()

    plt.savefig(
        OUT_FIGURES /
        "pca_explained_variance.png",
        dpi=300
    )

    plt.close()


    # PC1 vs PC2
    plt.figure(figsize=(9, 6))

    plt.scatter(
        score_df["PC1"],
        score_df["PC2"],
        s=5,
        alpha=0.25
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(
        "Judgments in Multivariate Structural Space"
    )

    plt.tight_layout()

    plt.savefig(
        OUT_FIGURES /
        "pca_pc1_pc2.png",
        dpi=300
    )

    plt.close()


    # Decade trajectory
    plt.figure(figsize=(9, 6))

    plt.plot(
        decade_pca["PC1"],
        decade_pca["PC2"],
        marker="o"
    )

    for _, row in decade_pca.iterrows():

        plt.annotate(
            str(row["decade"]),
            (row["PC1"], row["PC2"])
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(
        "Decadal Movement in Multivariate Judgment Structure"
    )

    plt.tight_layout()

    plt.savefig(
        OUT_FIGURES /
        "pca_decade_trajectory.png",
        dpi=300
    )

    plt.close()

    print("\nPCA figures saved.")

except Exception as e:

    print("\nPlotting skipped:", e)


print("\n" + "=" * 70)
print("STEP 11 COMPLETE")
print("=" * 70)