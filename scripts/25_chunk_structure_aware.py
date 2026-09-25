import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.chunking import structure_aware_chunk_document
from src.rag.config_loader import load_config

CONFIG = load_config(PROJECT_ROOT / "config" / "rag_config.json")

INPUT_PATH = PROJECT_ROOT / CONFIG["processed_dataset_path"]
OUTPUT_PATH = PROJECT_ROOT / CONFIG["output_directories"]["chunks"] / "structure_aware_chunks.jsonl"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

records = []
with open(INPUT_PATH, "r", encoding="utf-8") as infile:
    for line in infile:
        if line.strip():
            records.append(json.loads(line))

all_chunks = []
for record in records:
    chunks = structure_aware_chunk_document(record, CONFIG)
    all_chunks.extend(chunks)

with open(OUTPUT_PATH, "w", encoding="utf-8") as outfile:
    for chunk in all_chunks:
        outfile.write(json.dumps(chunk, ensure_ascii=False) + "\n")

print(f"Structure-aware chunks written: {len(all_chunks)}")
print(f"Output: {OUTPUT_PATH}")
