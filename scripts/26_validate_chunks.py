import csv
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.config_loader import load_config

CONFIG = load_config(PROJECT_ROOT / "config" / "rag_config.json")

TOOL_OUTPUT_DIR = PROJECT_ROOT / CONFIG["output_directories"]["tables"]
REPORT_DIR = PROJECT_ROOT / CONFIG["output_directories"]["reports"]
TOOL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize(chunks: list[dict], strategy: str):
    judgments = set(c["diary_no"] for c in chunks)
    word_counts = [c["word_count"] for c in chunks]
    per_judgment = {}
    for c in chunks:
        per_judgment.setdefault(c["diary_no"], 0)
        per_judgment[c["diary_no"]] += 1
    if not word_counts:
        return {
            "total_judgments": len(judgments),
            "total_chunks": 0,
            "mean_chunks_per_judgment": 0,
            "median_chunks_per_judgment": 0,
            "p95_chunks_per_judgment": 0,
            "min_chunk_words": 0,
            "max_chunk_words": 0,
            "mean_chunk_words": 0,
            "median_chunk_words": 0,
            "p95_chunk_words": 0,
            "empty_chunks": 0,
            "zero_chunk_judgments": 0,
            "tiny_chunks": 0,
            "large_chunks": 0,
            "duplicate_chunk_ids": 0,
            "duplicate_chunk_text": 0,
            "missing_provenance": 0,
            "invalid_source_offsets": 0,
            "strategy_distribution": {strategy: len(chunks)},
            "decade_distribution": {},
            "representative_chunks": [],
        }

    duplicate_chunk_ids = 0
    duplicate_chunk_text = 0
    chunk_id_counts = {}
    text_counts = {}
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        text = json.dumps(chunk["text"], ensure_ascii=False)
        chunk_id_counts[chunk_id] = chunk_id_counts.get(chunk_id, 0) + 1
        text_counts[text] = text_counts.get(text, 0) + 1

    duplicate_chunk_ids = sum(1 for count in chunk_id_counts.values() if count > 1)
    duplicate_chunk_text = sum(1 for count in text_counts.values() if count > 1)

    stats = {
        "total_judgments": len(judgments),
        "total_chunks": len(chunks),
        "mean_chunks_per_judgment": sum(per_judgment.values()) / len(per_judgment) if per_judgment else 0,
        "median_chunks_per_judgment": statistics.median(per_judgment.values()) if per_judgment else 0,
        "p95_chunks_per_judgment": sorted(per_judgment.values())[int(len(per_judgment) * 0.95) - 1] if per_judgment else 0,
        "min_chunk_words": min(word_counts),
        "max_chunk_words": max(word_counts),
        "mean_chunk_words": statistics.mean(word_counts),
        "median_chunk_words": statistics.median(word_counts),
        "p95_chunk_words": sorted(word_counts)[int(len(word_counts) * 0.95) - 1],
        "empty_chunks": sum(1 for c in chunks if not c["text"].strip()),
        "zero_chunk_judgments": sum(1 for j in judgments if j not in per_judgment),
        "tiny_chunks": sum(1 for c in chunks if c["word_count"] < 25),
        "large_chunks": sum(1 for c in chunks if c["word_count"] > 1800),
        "duplicate_chunk_ids": duplicate_chunk_ids,
        "duplicate_chunk_text": duplicate_chunk_text,
        "missing_provenance": sum(1 for c in chunks if not c.get("metadata", {}).get("provenance")),
        "invalid_source_offsets": sum(1 for c in chunks if c["start_position"] < 0 or c["end_position"] < c["start_position"]),
        "strategy_distribution": {strategy: len(chunks)},
        "decade_distribution": {},
        "representative_chunks": [],
    }
    decade_counter = {}
    for c in chunks:
        decade = c["decade"]
        decade_counter[decade] = decade_counter.get(decade, 0) + 1
    stats["decade_distribution"] = decade_counter
    sample_decades = [1950, 1970, 1990, 2010, 2020]
    for year in sample_decades:
        matches = [c for c in chunks if c["year"] == year]
        if matches:
            stats["representative_chunks"].append({"year": year, "sample": matches[0]["text"][:300]})
    return stats


def write_csv(path: Path, stats: dict):
    fieldnames = [
        "metric", "value"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key, value in stats.items():
            if key in {"strategy_distribution", "decade_distribution", "representative_chunks"}:
                writer.writerow({"metric": key, "value": json.dumps(value, ensure_ascii=False)})
            else:
                writer.writerow({"metric": key, "value": value})


def write_markdown(path: Path, stats: dict, title: str):
    lines = [
        f"# {title}",
        "",
        f"- Total judgments processed: {stats['total_judgments']}",
        f"- Total chunks: {stats['total_chunks']}",
        f"- Mean chunks/judgment: {stats['mean_chunks_per_judgment']:.2f}",
        f"- Median chunks/judgment: {stats['median_chunks_per_judgment']}",
        f"- P95 chunks/judgment: {stats['p95_chunks_per_judgment']}",
        f"- Min chunk word count: {stats['min_chunk_words']}",
        f"- Max chunk word count: {stats['max_chunk_words']}",
        f"- Mean chunk word count: {stats['mean_chunk_words']:.2f}",
        f"- Median chunk word count: {stats['median_chunk_words']}",
        f"- P95 chunk word count: {stats['p95_chunk_words']}",
        f"- Empty chunks: {stats['empty_chunks']}",
        f"- Zero-chunk judgments: {stats['zero_chunk_judgments']}",
        f"- Extremely tiny chunks: {stats['tiny_chunks']}",
        f"- Extremely large chunks: {stats['large_chunks']}",
        f"- Duplicate chunk IDs: {stats['duplicate_chunk_ids']}",
        f"- Duplicate chunk text: {stats['duplicate_chunk_text']}",
        f"- Missing provenance: {stats['missing_provenance']}",
        f"- Invalid source offsets: {stats['invalid_source_offsets']}",
        "",
        "## Strategy distribution",
        f"- {stats['strategy_distribution']}",
        "",
        "## Decade distribution",
        f"- {stats['decade_distribution']}",
        "",
        "## Representative chunks",
    ]
    for item in stats["representative_chunks"]:
        lines.append(f"- {item['year']}: {item['sample']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    generic_path = PROJECT_ROOT / CONFIG["output_directories"]["chunks"] / "generic_chunks.jsonl"
    structure_path = PROJECT_ROOT / CONFIG["output_directories"]["chunks"] / "structure_aware_chunks.jsonl"

    generic_chunks = load_jsonl(generic_path)
    structure_chunks = load_jsonl(structure_path)

    generic_stats = summarize(generic_chunks, "generic")
    structure_stats = summarize(structure_chunks, "structure_aware")

    write_csv(TOOL_OUTPUT_DIR / "chunk_validation_generic.csv", generic_stats)
    write_csv(TOOL_OUTPUT_DIR / "chunk_validation_structure_aware.csv", structure_stats)
    write_markdown(REPORT_DIR / "chunk_validation_generic.md", generic_stats, "Generic Chunk Validation")
    write_markdown(REPORT_DIR / "chunk_validation_structure_aware.md", structure_stats, "Structure-Aware Chunk Validation")

    print("Generic validation summary:")
    print(json.dumps(generic_stats, ensure_ascii=False, indent=2))
    print("\nStructure-aware validation summary:")
    print(json.dumps(structure_stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
