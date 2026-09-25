import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

INPUT = Path("data/processed/judgments_vocab.jsonl")
OUTPUT = Path("outputs/tables/topic_model_selection.csv")

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
print("STEP 8 - TOPIC MODEL SELECTION")
print("=" * 70)

print(f"Documents: {len(df):,}")

# ---------------------------------------------------------
# BAG-OF-WORDS REPRESENTATION
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
print(f"Document-term matrix: {X.shape}")

# ---------------------------------------------------------
# TEST DIFFERENT NUMBERS OF TOPICS
# ---------------------------------------------------------

topic_counts = [6, 8, 10, 12, 15, 18, 20]

results = []

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

    # Perplexity
    perplexity = lda.perplexity(X)

    # Approximate topic separation:
    # average maximum topic probability
    topic_distribution = lda.transform(X)

    avg_max_topic_probability = (
        topic_distribution.max(axis=1).mean()
    )

    results.append({
        "num_topics": k,
        "perplexity": perplexity,
        "avg_max_topic_probability": avg_max_topic_probability
    })

    print(f"Perplexity: {perplexity:.2f}")
    print(
        f"Average max topic probability: "
        f"{avg_max_topic_probability:.4f}"
    )

# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT,
    index=False
)

print()
print("=" * 70)
print("TOPIC MODEL SELECTION RESULTS")
print("=" * 70)

print(results_df.to_string(index=False))

print()
print(f"Saved: {OUTPUT}")

print()
print("=" * 70)
print("STEP 8A COMPLETE")
print("=" * 70)