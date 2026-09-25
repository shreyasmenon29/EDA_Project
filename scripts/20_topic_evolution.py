import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

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
print("STEP 9 - TOPIC EVOLUTION BY DECADE")
print("=" * 70)

print(f"Documents: {len(df):,}")

# ---------------------------------------------------------
# DECADE
# ---------------------------------------------------------

def decade_from_date(date_string):
    year = int(date_string[-4:])
    return f"{(year // 10) * 10}s"

df["decade"] = df["judgment_dates"].apply(decade_from_date)

# ---------------------------------------------------------
# BAG OF WORDS
# ---------------------------------------------------------

vectorizer = CountVectorizer(
    min_df=10,
    max_df=0.80,
    max_features=20000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(df["vocab_text"])

feature_names = vectorizer.get_feature_names_out()

print(f"Vocabulary size: {len(feature_names):,}")

# ---------------------------------------------------------
# FINAL LDA MODEL
# ---------------------------------------------------------

K = 12

print()
print(f"Training final LDA model with K={K}...")

lda = LatentDirichletAllocation(
    n_components=K,
    random_state=42,
    learning_method="batch",
    max_iter=40,
    n_jobs=-1
)

lda.fit(X)

# ---------------------------------------------------------
# TOP WORDS FOR EACH TOPIC
# ---------------------------------------------------------

topic_rows = []

for topic_id, weights in enumerate(lda.components_):

    top_indices = weights.argsort()[-20:][::-1]

    top_words = [
        feature_names[i]
        for i in top_indices
    ]

    topic_rows.append({
        "topic_id": topic_id + 1,
        "top_words": ", ".join(top_words)
    })

topic_df = pd.DataFrame(topic_rows)

topic_df.to_csv(
    TABLE_DIR / "final_topic_words.csv",
    index=False
)

# ---------------------------------------------------------
# TOPIC DISTRIBUTION
# ---------------------------------------------------------

print("Calculating document-topic distributions...")

topic_distribution = lda.transform(X)

topic_columns = [
    f"topic_{i+1}"
    for i in range(K)
]

topic_dist_df = pd.DataFrame(
    topic_distribution,
    columns=topic_columns
)

# Add metadata
topic_dist_df["decade"] = df["decade"].values
topic_dist_df["diary_no"] = df["diary_no"].values

# ---------------------------------------------------------
# TOPIC PREVALENCE BY DECADE
# ---------------------------------------------------------

decade_topic = (
    topic_dist_df
    .groupby("decade")[topic_columns]
    .mean()
)

decade_topic.to_csv(
    TABLE_DIR / "topic_prevalence_by_decade.csv"
)

# ---------------------------------------------------------
# DOMINANT TOPIC
# ---------------------------------------------------------

dominant_topic = topic_distribution.argmax(axis=1) + 1

topic_dist_df["dominant_topic"] = dominant_topic

dominant_counts = (
    topic_dist_df
    .groupby(["decade", "dominant_topic"])
    .size()
    .reset_index(name="documents")
)

dominant_counts["percentage"] = (
    dominant_counts
    .groupby("decade")["documents"]
    .transform(
        lambda x: x / x.sum() * 100
    )
)

dominant_counts.to_csv(
    TABLE_DIR / "dominant_topic_by_decade.csv",
    index=False
)

# ---------------------------------------------------------
# PRINT TOPIC WORDS
# ---------------------------------------------------------

print()
print("=" * 70)
print("FINAL TOPICS")
print("=" * 70)

for _, row in topic_df.iterrows():

    print(
        f"\nTopic {int(row['topic_id'])}:"
    )

    print(
        f"  {row['top_words']}"
    )

# ---------------------------------------------------------
# PRINT DECADE PREVALENCE
# ---------------------------------------------------------

print()
print("=" * 70)
print("TOPIC PREVALENCE BY DECADE")
print("=" * 70)

print(
    decade_topic.round(4).to_string()
)

# ---------------------------------------------------------
# PLOT TOPIC EVOLUTION
# ---------------------------------------------------------

plt.figure(figsize=(14, 8))

for topic in topic_columns:

    plt.plot(
        decade_topic.index,
        decade_topic[topic],
        marker="o",
        label=topic
    )

plt.xlabel("Decade")
plt.ylabel("Average Topic Proportion")
plt.title("Evolution of Legal Topic Prevalence Across Decades")

plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "topic_evolution_by_decade.png",
    dpi=300
)

plt.close()

# ---------------------------------------------------------
# HEATMAP
# ---------------------------------------------------------

plt.figure(figsize=(14, 8))

plt.imshow(
    decade_topic.values,
    aspect="auto"
)

plt.colorbar(
    label="Average Topic Proportion"
)

plt.xticks(
    range(len(topic_columns)),
    topic_columns,
    rotation=45
)

plt.yticks(
    range(len(decade_topic.index)),
    decade_topic.index
)

plt.xlabel("Topic")
plt.ylabel("Decade")

plt.title(
    "Heatmap of Legal Topic Prevalence by Decade"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "topic_prevalence_heatmap.png",
    dpi=300
)

plt.close()

# ---------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------

print()
print("=" * 70)
print("FILES CREATED")
print("=" * 70)

print(TABLE_DIR / "final_topic_words.csv")
print(TABLE_DIR / "topic_prevalence_by_decade.csv")
print(TABLE_DIR / "dominant_topic_by_decade.csv")

print(FIGURE_DIR / "topic_evolution_by_decade.png")
print(FIGURE_DIR / "topic_prevalence_heatmap.png")

print()
print("=" * 70)
print("STEP 9 COMPLETE")
print("=" * 70)