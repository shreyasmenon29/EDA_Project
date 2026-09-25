import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

INPUT = Path("data/processed/judgments_vocab.jsonl")

TABLE_DIR = Path("outputs/tables")
FIGURE_DIR = Path("outputs/figures")

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

records = []

with INPUT.open("r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

df = pd.DataFrame(records)

print("=" * 70)
print("STEP 7 - DOCUMENT-LEVEL TF-IDF ANALYSIS")
print("=" * 70)

print(f"Documents: {len(df):,}")

# ---------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------

vectorizer = TfidfVectorizer(
    min_df=5,
    max_df=0.85,
    max_features=30000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X = vectorizer.fit_transform(df["vocab_text"])

terms = vectorizer.get_feature_names_out()

print(f"Vocabulary size: {len(terms):,}")
print(f"TF-IDF matrix: {X.shape}")

# ---------------------------------------------------------
# GLOBAL TF-IDF TERMS
# ---------------------------------------------------------

mean_tfidf = X.mean(axis=0).A1

global_tfidf = pd.DataFrame({
    "term": terms,
    "mean_tfidf": mean_tfidf
}).sort_values(
    "mean_tfidf",
    ascending=False
)

global_tfidf.head(100).to_csv(
    TABLE_DIR / "tfidf_global_top100.csv",
    index=False
)

# ---------------------------------------------------------
# DECADE TF-IDF
# ---------------------------------------------------------

decades = sorted(df["decade"].unique())

decade_results = []

for decade in decades:

    indices = df.index[df["decade"] == decade].tolist()

    X_decade = X[indices]

    mean_scores = X_decade.mean(axis=0).A1

    temp = pd.DataFrame({
        "term": terms,
        "mean_tfidf": mean_scores
    })

    temp = temp.sort_values(
        "mean_tfidf",
        ascending=False
    ).head(50)

    temp["decade"] = decade

    decade_results.append(temp)

decade_tfidf = pd.concat(
    decade_results,
    ignore_index=True
)

decade_tfidf.to_csv(
    TABLE_DIR / "tfidf_top_terms_by_decade.csv",
    index=False
)

# ---------------------------------------------------------
# TERM TEMPORAL PROFILE
# ---------------------------------------------------------

# Calculate mean TF-IDF for every term in every decade

temporal_rows = []

for decade in decades:

    indices = df.index[df["decade"] == decade].tolist()

    mean_scores = X[indices].mean(axis=0).A1

    temporal_rows.append(mean_scores)

temporal_matrix = pd.DataFrame(
    temporal_rows,
    index=decades,
    columns=terms
)

# ---------------------------------------------------------
# TERMS WITH STRONGEST TEMPORAL CHANGE
# ---------------------------------------------------------

change_results = []

for term in terms:

    values = temporal_matrix[term]

    max_value = values.max()
    min_value = values.min()

    max_decade = values.idxmax()
    min_decade = values.idxmin()

    change = max_value - min_value

    change_results.append({
        "term": term,
        "max_decade": max_decade,
        "min_decade": min_decade,
        "max_tfidf": max_value,
        "min_tfidf": min_value,
        "tfidf_range": change
    })

change_df = pd.DataFrame(change_results)

change_df = change_df.sort_values(
    "tfidf_range",
    ascending=False
)

change_df.head(200).to_csv(
    TABLE_DIR / "tfidf_temporal_change.csv",
    index=False
)

# ---------------------------------------------------------
# VOCABULARY SIZE
# ---------------------------------------------------------

vocab_stats = []

for decade in decades:

    subset = df[df["decade"] == decade]

    all_tokens = []

    for text in subset["vocab_text"]:
        all_tokens.extend(text.split())

    unique_tokens = len(set(all_tokens))
    total_tokens = len(all_tokens)

    vocab_stats.append({
        "decade": decade,
        "documents": len(subset),
        "tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "type_token_ratio": (
            unique_tokens / total_tokens
            if total_tokens > 0 else 0
        )
    })

vocab_stats_df = pd.DataFrame(vocab_stats)

vocab_stats_df.to_csv(
    TABLE_DIR / "tfidf_vocabulary_statistics.csv",
    index=False
)

# ---------------------------------------------------------
# PLOT
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    vocab_stats_df["decade"],
    vocab_stats_df["type_token_ratio"],
    marker="o"
)

plt.xlabel("Decade")
plt.ylabel("Type-Token Ratio")
plt.title("Vocabulary Diversity Across Decades")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "tfidf_vocabulary_diversity.png",
    dpi=300
)

plt.close()

# ---------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------

print()
print("Top global TF-IDF terms:")
print(global_tfidf.head(20).to_string(index=False))

print()
print("Top TF-IDF terms by decade:")
for decade in decades:
    print(f"\n{decade}")
    print(
        decade_tfidf[
            decade_tfidf["decade"] == decade
        ].head(10)[["term", "mean_tfidf"]].to_string(index=False)
    )

print()
print("Strongest temporal vocabulary changes:")
print(
    change_df.head(30).to_string(index=False)
)

print()
print("Files created:")
print(TABLE_DIR / "tfidf_global_top100.csv")
print(TABLE_DIR / "tfidf_top_terms_by_decade.csv")
print(TABLE_DIR / "tfidf_temporal_change.csv")
print(TABLE_DIR / "tfidf_vocabulary_statistics.csv")
print(FIGURE_DIR / "tfidf_vocabulary_diversity.png")

print()
print("=" * 70)
print("STEP 7 COMPLETE")
print("=" * 70)