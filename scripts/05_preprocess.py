import json
import re
import unicodedata
from pathlib import Path

INPUT_PATH = Path("../data/raw/sampled_judgments.jsonl")
OUTPUT_PATH = Path("../data/processed/judgments_processed.jsonl")


def clean_text(text):
    if not isinstance(text, str):
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Convert escaped whitespace into actual whitespace
    text = text.replace("\\n", "\n")
    text = text.replace("\\t", " ")

    # Remove control characters except newline and tab
    text = "".join(
        char
        for char in text
        if char in "\n\t" or not unicodedata.category(char).startswith("C")
    )

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove excessive spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

processed_count = 0

with open(INPUT_PATH, "r", encoding="utf-8") as infile, \
     open(OUTPUT_PATH, "w", encoding="utf-8") as outfile:

    for line in infile:

        record = json.loads(line)

        original_text = record.get("full_text", "")

        record["clean_text"] = clean_text(original_text)

        outfile.write(
            json.dumps(
                record,
                ensure_ascii=False
            ) + "\n"
        )

        processed_count += 1

print("Preprocessing complete!")
print("Records processed:", processed_count)
print("Output:", OUTPUT_PATH)