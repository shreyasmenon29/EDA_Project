import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

INPUT = Path("data/processed/judgments_vocab.jsonl")
OUTPUT = Path("outputs/tables/topic_coherence_analysis.csv")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

records = []

with INPUT.open("r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

df = pd.DataFrame(records)

print("=" * 70)
print("STEP 8B - TOPIC COHERENCE & INTERPRETABILITY")
print("=" * 70)

print(f"Documents: {len(df):,}")

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
# CO-OCCURRENCE BASED COHERENCE
# ---------------------------------------------------------
#
# For each topic:
#   - take top 15 words
#   - calculate how often pairs of those words
#     occur together across documents
#
# This is a simple UMass-style coherence measure.
# Higher = words tend to occur together more often.
# ---------------------------------------------------------

binary_X = (X > 0).astype(int)

document_frequency = np.asarray(
    binary_X.sum(axis=0)
).ravel()

df_lookup = {
    i: document_frequency[i]
    for i in range(len(document_frequency))
}

topic_counts = [8, 10, 12, 15, 18, 20]

results = []

all_topic_details = []

for k in topic_counts:

    print()
    print(f"Training LDA with K={k}...")

    lda = LatentDirichletAllocation(
        n_components=k,
        random_state=42,
        learning_method="batch",
        max_iter=30,
        n_jobs=-1
    )

    lda.fit(X)

    topic_coherences = []

    for topic_id, topic_weights in enumerate(lda.components_):

        top_indices = topic_weights.argsort()[-15:][::-1]

        top_words = [
            feature_names[i]
            for i in top_indices
        ]

        pair_scores = []

        for i in range(1, len(top_indices)):

            word_i = top_indices[i]

            df_i = df_lookup[word_i]

            if df_i == 0:
                continue

            for j in range(i):

                word_j = top_indices[j]

                # Number of documents containing both words
                cooccur = (
                    binary_X[:, word_i]
                    .multiply(binary_X[:, word_j])
                    .sum()
                )

                # UMass-style log coherence
                score = np.log(
                    (cooccur + 1) / df_i
                )

                pair_scores.append(score)

        coherence = (
            np.mean(pair_scores)
            if pair_scores
            else 0
        )

        topic_coherences.append(coherence)

        all_topic_details.append({
            "num_topics": k,
            "topic_id": topic_id + 1,
            "coherence": coherence,
            "top_words": ", ".join(top_words)
        })

    # Average coherence across topics
    average_coherence = np.mean(topic_coherences)

    # Topic coherence spread
    coherence_std = np.std(topic_coherences)

    results.append({
        "num_topics": k,
        "average_coherence": average_coherence,
        "coherence_std": coherence_std,
        "minimum_topic_coherence": min(topic_coherences),
        "maximum_topic_coherence": max(topic_coherences)
    })

    print(
        f"Average coherence: {average_coherence:.4f}"
    )

# ---------------------------------------------------------
# SAVE SUMMARY
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "average_coherence",
    ascending=False
)

results_df.to_csv(
    OUTPUT,
    index=False
)

# ---------------------------------------------------------
# SAVE TOPIC DETAILS
# ---------------------------------------------------------

topic_details_df = pd.DataFrame(all_topic_details)

topic_details_df.to_csv(
    "outputs/tables/topic_interpretability.csv",
    index=False
)

# ---------------------------------------------------------
# PRINT RESULTS
# ---------------------------------------------------------

print()
print("=" * 70)
print("COHERENCE RESULTS")
print("=" * 70)

print(
    results_df.to_string(index=False)
)

# ---------------------------------------------------------
# PRINT TOPICS FOR EACH K
# ---------------------------------------------------------

for k in topic_counts:

    print()
    print("=" * 70)
    print(f"TOPIC INTERPRETABILITY — K={k}")
    print("=" * 70)

    subset = topic_details_df[
        topic_details_df["num_topics"] == k
    ].sort_values(
        "coherence",
        ascending=False
    )

    for _, row in subset.iterrows():

        print(
            f"\nTopic {int(row['topic_id'])} "
            f"(coherence={row['coherence']:.4f})"
        )

        print(
            f"  {row['top_words']}"
        )

print()
print("=" * 70)
print("FILES CREATED")
print("=" * 70)

print(OUTPUT)
print("outputs/tables/topic_interpretability.csv")

print()
print("=" * 70)
print("STEP 8B COMPLETE")
print("=" * 70)