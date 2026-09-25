import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "retrieval_questions.jsonl"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

EXAMPLES = [
    {
        "question_id": "q_001",
        "question": "What principles govern remoteness of damages in contract disputes?",
        "category": "contract",
        "era": "1950s",
        "relevant_diary_nos": [],
        "relevant_chunk_ids": [],
        "notes": "REQUIRES HUMAN ANNOTATION: This benchmark entry is a schema placeholder, not ground truth."
    },
    {
        "question_id": "q_002",
        "question": "How has the Supreme Court treated constitutional challenges under Article 21 and Article 19?",
        "category": "constitutional",
        "era": "2010s",
        "relevant_diary_nos": [],
        "relevant_chunk_ids": [],
        "notes": "REQUIRES HUMAN ANNOTATION: Relevance labels must be created by a human annotation workflow."
    },
    {
        "question_id": "q_003",
        "question": "What are the procedural issues concerning criminal appeals and bail?",
        "category": "criminal_procedure",
        "era": "2020s",
        "relevant_diary_nos": [],
        "relevant_chunk_ids": [],
        "notes": "REQUIRES HUMAN ANNOTATION: No synthetic labels are permitted."
    },
]

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for item in EXAMPLES:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Wrote evaluation schema to: {OUTPUT_PATH}")
print(f"Records: {len(EXAMPLES)}")
print("Schema: question_id, question, category, era, relevant_diary_nos, relevant_chunk_ids, notes")
