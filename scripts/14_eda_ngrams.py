import json
import os
import csv
import re
from collections import Counter, defaultdict

import matplotlib.pyplot as plt


INPUT_FILE = "data/processed/judgments_processed.jsonl"

TABLE_DIR = "outputs/tables"
FIGURE_DIR = "outputs/figures"

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

# Very common English/legal-function words.
# These are removed mainly so that decade-specific vocabulary
# is more informative.

STOPWORDS = {
    "the", "of", "and", "to", "in", "a", "is", "that",
    "for", "on", "it", "as", "was", "with", "by", "this",
    "be", "are", "from", "or", "an", "at", "which", "but",
    "not", "have", "has", "had", "were", "their", "they",
    "he", "she", "his", "her", "we", "you", "I", "its",
    "there", "been", "would", "could", "should", "may",
    "can", "will", "shall", "also", "than", "then",
    "these", "those", "such", "more", "other", "some",
    "any", "all", "each", "both", "where", "when",
    "while", "who", "whom", "what", "how", "if", "so",
    "only", "very", "into", "over", "under", "after",
    "before", "between", "through", "during", "against",
    "being", "because", "therefore", "however"
}


# Common legal boilerplate that otherwise dominates
# the frequency tables.

LEGAL_STOPWORDS = {
    "court",
    "supreme",
    "india",
    "judgment",
    "judgement",
    "order",
    "appeal",
    "case",
    "petition",
    "petitioner",
    "respondent",
    "appellant",
    "learned",
    "honble",
    "honble",
    "justice",
    "judge",
    "judges"
}


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    records = []

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            records.append(
                json.loads(line)
            )

    return records


# ============================================================
# HELPERS
# ============================================================

def get_year(date_value):

    try:
        return int(
            str(date_value).split("-")[2]
        )

    except:
        return None


def get_decade(year):

    return f"{(year // 10) * 10}s"


def tokenize(text):

    tokens = re.findall(
        r"\b[a-zA-Z][a-zA-Z'-]{2,}\b",
        str(text).lower()
    )

    return tokens


def filtered_tokens(text):

    tokens = tokenize(text)

    return [
        token
        for token in tokens
        if token not in STOPWORDS
    ]


def informative_tokens(text):

    tokens = tokenize(text)

    return [
        token
        for token in tokens
        if (
            token not in STOPWORDS
            and token not in LEGAL_STOPWORDS
        )
    ]


def make_ngrams(tokens, n):

    return [
        " ".join(
            tokens[i:i+n]
        )
        for i in range(
            len(tokens) - n + 1
        )
    ]


# ============================================================
# MAIN
# ============================================================

def main():

    records = load_data()

    print("=" * 75)
    print("PHASE 2 - STEP 6: N-GRAM & VOCABULARY ANALYSIS")
    print("=" * 75)

    overall_unigrams = Counter()
    overall_bigrams = Counter()
    overall_trigrams = Counter()

    decade_unigrams = defaultdict(Counter)
    decade_bigrams = defaultdict(Counter)
    decade_trigrams = defaultdict(Counter)

    decade_documents = defaultdict(int)

    # ========================================================
    # PROCESS DOCUMENTS
    # ========================================================

    for record in records:

        year = get_year(
            record.get(
                "judgment_dates"
            )
        )

        if year is None:
            continue

        decade = get_decade(year)

        text = record.get(
            "clean_text",
            ""
        )

        tokens = informative_tokens(text)

        if not tokens:
            continue

        decade_documents[
            decade
        ] += 1

        unigrams = Counter(tokens)

        bigrams = Counter(
            make_ngrams(
                tokens,
                2
            )
        )

        trigrams = Counter(
            make_ngrams(
                tokens,
                3
            )
        )

        overall_unigrams.update(
            unigrams
        )

        overall_bigrams.update(
            bigrams
        )

        overall_trigrams.update(
            trigrams
        )

        decade_unigrams[
            decade
        ].update(
            unigrams
        )

        decade_bigrams[
            decade
        ].update(
            bigrams
        )

        decade_trigrams[
            decade
        ].update(
            trigrams
        )

    # ========================================================
    # SAVE TOP N-GRAMS
    # ========================================================

    def save_counter(
        counter,
        filename,
        top_n=100
    ):

        path = os.path.join(
            TABLE_DIR,
            filename
        )

        with open(
            path,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "term",
                "frequency"
            ])

            for term, count in counter.most_common(
                top_n
            ):

                writer.writerow([
                    term,
                    count
                ])

        return path


    unigram_file = save_counter(
        overall_unigrams,
        "top_unigrams.csv"
    )

    bigram_file = save_counter(
        overall_bigrams,
        "top_bigrams.csv"
    )

    trigram_file = save_counter(
        overall_trigrams,
        "top_trigrams.csv"
    )

    # ========================================================
    # PRINT TOP N-GRAMS
    # ========================================================

    print("\n--- TOP 30 UNIGRAMS ---\n")

    for term, count in (
        overall_unigrams
        .most_common(30)
    ):

        print(
            f"{term:30} {count}"
        )

    print("\n--- TOP 30 BIGRAMS ---\n")

    for term, count in (
        overall_bigrams
        .most_common(30)
    ):

        print(
            f"{term:45} {count}"
        )

    print("\n--- TOP 30 TRIGRAMS ---\n")

    for term, count in (
        overall_trigrams
        .most_common(30)
    ):

        print(
            f"{term:60} {count}"
        )

    # ========================================================
    # DECADE-SPECIFIC VOCABULARY
    # ========================================================

    decades = sorted(
        decade_unigrams.keys()
    )

    # --------------------------------------------------------
    # Calculate corpus-wide document frequency
    # --------------------------------------------------------

    document_frequency = Counter()

    for decade in decades:

        # Approximation at decade level:
        # number of decades containing term

        for term in decade_unigrams[
            decade
        ]:

            document_frequency[
                term
            ] += 1

    # ========================================================
    # DECADE DISTINCTIVE TERMS
    # ========================================================

    distinctive_rows = []

    for decade in decades:

        current = decade_unigrams[
            decade
        ]

        total_tokens = sum(
            current.values()
        )

        for term, count in current.items():

            if count < 10:
                continue

            # Term frequency within decade
            tf = count / total_tokens

            # Penalize terms appearing in many decades
            df = document_frequency[
                term
            ]

            idf = (
                __import__("math").log(
                    len(decades) /
                    (1 + df)
                )
                + 1
            )

            score = tf * idf

            distinctive_rows.append({

                "decade":
                    decade,

                "term":
                    term,

                "frequency":
                    count,

                "tf":
                    tf,

                "decade_frequency":
                    df,

                "distinctiveness_score":
                    score
            })

    distinctive_file = os.path.join(
        TABLE_DIR,
        "distinctive_vocabulary_by_decade.csv"
    )

    with open(
        distinctive_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "decade",
                "term",
                "frequency",
                "tf",
                "decade_frequency",
                "distinctiveness_score"
            ]
        )

        writer.writeheader()
        writer.writerows(
            distinctive_rows
        )

    # ========================================================
    # PRINT TOP DISTINCTIVE TERMS
    # ========================================================

    print(
        "\n--- DISTINCTIVE TERMS BY DECADE ---"
    )

    for decade in decades:

        candidates = [
            row
            for row in distinctive_rows
            if row["decade"] == decade
        ]

        candidates.sort(
            key=lambda x:
                x["distinctiveness_score"],
            reverse=True
        )

        print(
            f"\n{decade}"
        )

        for row in candidates[:15]:

            print(
                f"  {row['term']:25} "
                f"freq={row['frequency']:6} "
                f"score={row['distinctiveness_score']:.6f}"
            )

    # ========================================================
    # DECADE TOP BIGRAMS
    # ========================================================

    decade_bigram_rows = []

    for decade in decades:

        for term, count in (
            decade_bigrams[
                decade
            ].most_common(30)
        ):

            decade_bigram_rows.append({

                "decade":
                    decade,

                "bigram":
                    term,

                "frequency":
                    count
            })

    bigram_decade_file = os.path.join(
        TABLE_DIR,
        "top_bigrams_by_decade.csv"
    )

    with open(
        bigram_decade_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "decade",
                "bigram",
                "frequency"
            ]
        )

        writer.writeheader()
        writer.writerows(
            decade_bigram_rows
        )

    # ========================================================
    # DECADE TOP TRIGRAMS
    # ========================================================

    decade_trigram_rows = []

    for decade in decades:

        for term, count in (
            decade_trigrams[
                decade
            ].most_common(30)
        ):

            decade_trigram_rows.append({

                "decade":
                    decade,

                "trigram":
                    term,

                "frequency":
                    count
            })

    trigram_decade_file = os.path.join(
        TABLE_DIR,
        "top_trigrams_by_decade.csv"
    )

    with open(
        trigram_decade_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "decade",
                "trigram",
                "frequency"
            ]
        )

        writer.writeheader()
        writer.writerows(
            decade_trigram_rows
        )

    # ========================================================
    # VOCABULARY SIZE BY DECADE
    # ========================================================

    vocabulary_rows = []

    for decade in decades:

        counter = decade_unigrams[
            decade
        ]

        total_tokens = sum(
            counter.values()
        )

        unique_terms = len(counter)

        vocabulary_rows.append({

            "decade":
                decade,

            "total_tokens":
                total_tokens,

            "unique_terms":
                unique_terms,

            "type_token_ratio":
                (
                    unique_terms /
                    total_tokens
                    if total_tokens
                    else 0
                )
        })

    vocabulary_file = os.path.join(
        TABLE_DIR,
        "vocabulary_statistics_by_decade.csv"
    )

    with open(
        vocabulary_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "decade",
                "total_tokens",
                "unique_terms",
                "type_token_ratio"
            ]
        )

        writer.writeheader()
        writer.writerows(
            vocabulary_rows
        )

    # ========================================================
    # PRINT VOCABULARY
    # ========================================================

    print(
        "\n--- VOCABULARY STATISTICS ---\n"
    )

    for row in vocabulary_rows:

        print(
            f"{row['decade']:6} | "
            f"Tokens={row['total_tokens']:10} | "
            f"Unique={row['unique_terms']:8} | "
            f"TTR={row['type_token_ratio']:.5f}"
        )

    # ========================================================
    # FIGURE — VOCABULARY SIZE
    # ========================================================

    x = list(range(
        len(vocabulary_rows)
    ))

    unique_terms = [
        row["unique_terms"]
        for row in vocabulary_rows
    ]

    plt.figure(figsize=(11, 6))

    plt.plot(
        x,
        unique_terms,
        marker="o"
    )

    plt.xticks(
        x,
        [
            row["decade"]
            for row in vocabulary_rows
        ]
    )

    plt.title(
        "Unique Vocabulary by Decade"
    )

    plt.xlabel("Decade")

    plt.ylabel(
        "Number of Unique Terms"
    )

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "unique_vocabulary_by_decade.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 75)
    print("STEP 6 COMPLETE")
    print("=" * 75)

    print("\nTables:")

    print(unigram_file)
    print(bigram_file)
    print(trigram_file)
    print(distinctive_file)
    print(bigram_decade_file)
    print(trigram_decade_file)
    print(vocabulary_file)

    print("\nFigure:")
    print(
        "unique_vocabulary_by_decade.png"
    )


if __name__ == "__main__":
    main()