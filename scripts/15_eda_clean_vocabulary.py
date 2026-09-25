import json
import re
from pathlib import Path
from collections import Counter, defaultdict

INPUT = Path("data/processed/judgments_processed.jsonl")
OUTPUT = Path("data/processed/judgments_vocab.jsonl")

# Dataset/source boilerplate observed during Step 6
BOILERPLATE_TERMS = {
    "http",
    "https",
    "judis",
    "nic",
    "page",
    "pages",
    "xxx",
    "diary",
}

STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "which",
    "were", "have", "has", "had", "been", "being", "are", "was",
    "will", "would", "could", "should", "shall", "may", "might",
    "their", "there", "these", "those", "they", "them", "then",
    "than", "into", "upon", "such", "also", "only", "more", "most",
    "some", "any", "all", "not", "but", "its", "our", "out", "one",
    "two", "three", "here", "where", "when", "what", "who", "whom",
    "how", "why", "each", "other", "other", "same", "said"
}

LEGAL_FUNCTION_WORDS = {
    "court",
    "courts",
    "supreme",
    "india",
    "indian",
    "judgment",
    "judgments",
    "judgement",
    "judgements",
    "order",
    "orders",
    "appeal",
    "appeals",
    "case",
    "cases",
    "petition",
    "petitions",
    "petitioner",
    "petitioners",
    "respondent",
    "respondents",
    "appellant",
    "appellants",
    "justice",
    "judge",
    "judges",
    "learned",
}

def tokenize(text):
    return re.findall(r"\b[a-zA-Z][a-zA-Z'-]{2,}\b", text.lower())

def decade_from_date(date_string):
    year = int(date_string[-4:])
    return f"{(year // 10) * 10}s"

with INPUT.open("r", encoding="utf-8") as f_in, \
     OUTPUT.open("w", encoding="utf-8") as f_out:

    for line in f_in:
        record = json.loads(line)

        text = record.get("clean_text", "")

        tokens = tokenize(text)

        # Remove source boilerplate and ordinary stopwords
        vocab_tokens = [
            token
            for token in tokens
            if token not in BOILERPLATE_TERMS
            and token not in STOPWORDS
            and token not in LEGAL_FUNCTION_WORDS
        ]

        record["vocab_text"] = " ".join(vocab_tokens)
        record["vocab_token_count"] = len(vocab_tokens)
        record["decade"] = decade_from_date(record["judgment_dates"])

        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

print("=" * 70)
print("STEP 6A - VOCABULARY CLEANING")
print("=" * 70)
print(f"Input : {INPUT}")
print(f"Output: {OUTPUT}")
print("Boilerplate removed:", ", ".join(sorted(BOILERPLATE_TERMS)))
print("Vocabulary-only representation created successfully.")