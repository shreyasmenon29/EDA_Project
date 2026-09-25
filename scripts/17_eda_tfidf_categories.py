import json
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

INPUT = Path("data/processed/judgments_vocab.jsonl")

OUTPUT = Path("outputs/tables/tfidf_category_analysis.csv")

LEGAL_TERMS = {
    "constitution",
    "constitutional",
    "statute",
    "statutory",
    "provision",
    "provisions",
    "section",
    "article",
    "petition",
    "appeal",
    "appellant",
    "respondent",
    "petitioner",
    "jurisdiction",
    "legislation",
    "legislative",
    "fundamental",
    "rights",
    "liberty",
    "equality",
    "writ",
    "mandamus",
    "certiorari",
    "prohibition",
    "habeas",
    "criminal",
    "civil",
    "evidence",
    "offence",
    "offense",
    "conviction",
    "acquittal",
    "bail",
    "murder",
    "property",
    "contract",
    "tax",
    "income",
    "land",
    "labour",
    "labor",
    "employment",
    "arbitration",
    "contempt",
    "jurisdiction",
    "precedent",
    "ratio",
    "doctrine",
    "interpretation",
    "review",
    "judicial",
}

STRUCTURAL_TERMS = {
    "headnote",
    "reportable",
    "digitally",
    "signed",
    "signature",
    "verified",
    "bench",
    "coram",
    "page",
    "pages",
    "dated",
    "diary",
    "report",
    "citation",
    "citations",
    "scc",
    "scr",
    "air",
    "versus",
    "vs",
    "slp",
    "connected",
    "pronounced",
}

def decade_from_date(date_string):
    year = int(date_string[-4:])
    return f"{(year // 10) * 10}s"

# ---------------------------------------------------------
# LOAD
# ---------------------------------------------------------

records = []

with INPUT.open("r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

df = pd.DataFrame(records)

df["decade"] = df["judgment_dates"].apply(decade_from_date)

print("=" * 70)
print("STEP 7B - TF-IDF CATEGORY ANALYSIS")
print("=" * 70)

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

# ---------------------------------------------------------
# CATEGORY CLASSIFICATION
# ---------------------------------------------------------

def classify_term(term):

    words = set(term.split())

    legal_match = words.intersection(LEGAL_TERMS)
    structural_match = words.intersection(STRUCTURAL_TERMS)

    if structural_match:
        return "structural_format"

    if legal_match:
        return "legal"

    return "general"

categories = {
    term: classify_term(term)
    for term in terms
}

# ---------------------------------------------------------
# DECADE ANALYSIS
# ---------------------------------------------------------

rows = []

for decade in sorted(df["decade"].unique()):

    indices = df.index[df["decade"] == decade].tolist()

    X_decade = X[indices]

    scores = X_decade.mean(axis=0).A1

    temp = pd.DataFrame({
        "term": terms,
        "tfidf": scores
    })

    temp["category"] = temp["term"].map(categories)

    temp["decade"] = decade

    for category in [
        "legal",
        "structural_format",
        "general"
    ]:

        subset = temp[
            temp["category"] == category
        ].sort_values(
            "tfidf",
            ascending=False
        ).head(20)

        for _, row in subset.iterrows():

            rows.append({
                "decade": decade,
                "category": category,
                "term": row["term"],
                "tfidf": row["tfidf"]
            })

result = pd.DataFrame(rows)

result.to_csv(
    OUTPUT,
    index=False
)

# ---------------------------------------------------------
# PRINT
# ---------------------------------------------------------

for decade in sorted(df["decade"].unique()):

    print()
    print("-" * 60)
    print(decade)

    for category in [
        "legal",
        "structural_format",
        "general"
    ]:

        print(f"\n{category.upper()}")

        subset = result[
            (result["decade"] == decade) &
            (result["category"] == category)
        ].head(10)

        print(
            subset[
                ["term", "tfidf"]
            ].to_string(index=False)
        )

print()
print("=" * 70)
print("STEP 7B COMPLETE")
print("=" * 70)

print(f"Output: {OUTPUT}")