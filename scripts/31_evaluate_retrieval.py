from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from src.rag.config_loader import load_config
from src.rag.embeddings import load_embedding_model
from src.rag.faiss_index import FaissIndex, load_mapping
from src.rag.retrieval import Retriever


PROJECT_ROOT = Path(".")
QUESTIONS_PATH = PROJECT_ROOT / "data/evaluation/retrieval_questions.jsonl"
FAISS_DIR = PROJECT_ROOT / "data/faiss"
EMBEDDINGS_DIR = PROJECT_ROOT / "data/embeddings"
OUTPUT_DIR = PROJECT_ROOT / "outputs/evaluation"


TOP_K_VALUES = [1, 3, 5, 10]


def load_questions(path: Path) -> list[dict]:
    """Load benchmark questions from JSONL."""
    questions = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            questions.append(record)

    return questions


def normalize_ids(values) -> set[str]:
    """Convert diary/chunk IDs into a normalized string set."""
    if values is None:
        return set()

    if isinstance(values, str):
        values = [values]

    return {
        str(value).strip()
        for value in values
        if str(value).strip()
    }


def unique_results_by_diary(results: list[dict]) -> list[dict]:
    """
    Collapse multiple retrieved chunks from the same judgment.

    This makes the primary evaluation judgment-level rather than
    allowing one judgment with many chunks to dominate the ranking.
    """
    unique = []
    seen = set()

    for result in results:
        diary_no = str(result.get("diary_no", "")).strip()

        if not diary_no:
            continue

        if diary_no in seen:
            continue

        seen.add(diary_no)
        unique.append(result)

    return unique


def recall_at_k(results: list[dict], relevant_diaries: set[str], k: int) -> float:
    """Binary judgment-level Recall@K."""
    if not relevant_diaries:
        return float("nan")

    retrieved = {
        str(result.get("diary_no", "")).strip()
        for result in results[:k]
    }

    return float(bool(retrieved & relevant_diaries))


def reciprocal_rank(results: list[dict], relevant_diaries: set[str]) -> float:
    """Reciprocal rank of the first relevant judgment."""
    if not relevant_diaries:
        return float("nan")

    for rank, result in enumerate(results, start=1):
        diary_no = str(result.get("diary_no", "")).strip()

        if diary_no in relevant_diaries:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(results: list[dict], relevant_diaries: set[str], k: int) -> float:
    """
    Binary-relevance nDCG@K.

    Since the current annotation schema identifies relevant judgments,
    each relevant judgment receives relevance=1 and every other judgment
    receives relevance=0.
    """
    if not relevant_diaries:
        return float("nan")

    retrieved = results[:k]

    dcg = 0.0

    for rank, result in enumerate(retrieved, start=1):
        diary_no = str(result.get("diary_no", "")).strip()

        if diary_no in relevant_diaries:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_relevant_count = min(len(relevant_diaries), k)

    if ideal_relevant_count == 0:
        return float("nan")

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant_count + 1)
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


def evaluate_question(
    retriever: Retriever,
    question: dict,
    strategy: str,
    top_k: int,
) -> dict:
    """Run one benchmark question through one retrieval strategy."""

    question_id = question.get("question_id", "")
    query = question.get("question", "")
    category = question.get("category", "")
    era = question.get("era", "")

    relevant_diaries = normalize_ids(
        question.get("relevant_diary_nos", [])
    )

    if not relevant_diaries:
        return {
            "question_id": question_id,
            "strategy": strategy,
            "category": category,
            "era": era,
            "status": "SKIPPED_UNANNOTATED",
            "recall_at_1": "",
            "recall_at_3": "",
            "recall_at_5": "",
            "recall_at_10": "",
            "mrr": "",
            "ndcg_at_5": "",
            "ndcg_at_10": "",
            "retrieved_diaries": "",
            "top_score": "",
        }

    results = retriever.retrieve(
        query,
        strategy,
        top_k,
    )

    # Evaluate unique judgments rather than counting repeated chunks
    # from the same judgment multiple times.
    judgment_results = unique_results_by_diary(results)

    metrics = {
        "question_id": question_id,
        "strategy": strategy,
        "category": category,
        "era": era,
        "status": "EVALUATED",
        "recall_at_1": recall_at_k(
            judgment_results,
            relevant_diaries,
            1,
        ),
        "recall_at_3": recall_at_k(
            judgment_results,
            relevant_diaries,
            3,
        ),
        "recall_at_5": recall_at_k(
            judgment_results,
            relevant_diaries,
            5,
        ),
        "recall_at_10": recall_at_k(
            judgment_results,
            relevant_diaries,
            10,
        ),
        "mrr": reciprocal_rank(
            judgment_results,
            relevant_diaries,
        ),
        "ndcg_at_5": ndcg_at_k(
            judgment_results,
            relevant_diaries,
            5,
        ),
        "ndcg_at_10": ndcg_at_k(
            judgment_results,
            relevant_diaries,
            10,
        ),
        "retrieved_diaries": " | ".join(
            str(result.get("diary_no", ""))
            for result in judgment_results[:10]
        ),
        "top_score": (
            f"{results[0]['score']:.6f}"
            if results
            else ""
        ),
    }

    return metrics


def mean(values: list[float]) -> float:
    """Mean ignoring NaN values."""
    valid = [
        value
        for value in values
        if isinstance(value, (int, float))
        and not math.isnan(value)
    ]

    if not valid:
        return float("nan")

    return sum(valid) / len(valid)


def aggregate_metrics(rows: list[dict]) -> dict:
    """Calculate overall metrics for one group."""
    evaluated = [
        row
        for row in rows
        if row["status"] == "EVALUATED"
    ]

    return {
        "questions_evaluated": len(evaluated),
        "recall_at_1": mean(
            [float(row["recall_at_1"]) for row in evaluated]
        ),
        "recall_at_3": mean(
            [float(row["recall_at_3"]) for row in evaluated]
        ),
        "recall_at_5": mean(
            [float(row["recall_at_5"]) for row in evaluated]
        ),
        "recall_at_10": mean(
            [float(row["recall_at_10"]) for row in evaluated]
        ),
        "mrr": mean(
            [float(row["mrr"]) for row in evaluated]
        ),
        "ndcg_at_5": mean(
            [float(row["ndcg_at_5"]) for row in evaluated]
        ),
        "ndcg_at_10": mean(
            [float(row["ndcg_at_10"]) for row in evaluated]
        ),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    """Write rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def format_metric(value: float) -> str:
    if isinstance(value, float) and math.isnan(value):
        return ""

    return f"{value:.4f}"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate generic vs structure-aware retrieval "
            "using human-annotated judgment relevance."
        )
    )

    parser.add_argument(
        "--questions",
        default=str(QUESTIONS_PATH),
        help="Path to retrieval benchmark JSONL.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of retrieved chunks per query.",
    )

    args = parser.parse_args()

    questions_path = Path(args.questions)

    if not questions_path.exists():
        raise FileNotFoundError(
            f"Benchmark file not found: {questions_path}"
        )

    generic_index_path = FAISS_DIR / "generic.index"
    structure_index_path = FAISS_DIR / "structure_aware.index"

    generic_mapping_path = (
        EMBEDDINGS_DIR / "generic_mapping.json"
    )

    structure_mapping_path = (
        EMBEDDINGS_DIR / "structure_aware_mapping.json"
    )

    required_files = [
        generic_index_path,
        structure_index_path,
        generic_mapping_path,
        structure_mapping_path,
    ]

    missing = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing:
        print("Missing required retrieval artifacts:")

        for path in missing:
            print(f"  - {path}")

        print()
        print(
            "Complete embedding/FAISS construction before "
            "running the evaluation."
        )

        return

    config = load_config(
        Path("config/rag_config.json")
    )

    print("Loading embedding model...")
    model = load_embedding_model(config)

    print("Loading FAISS indexes...")

    indexes = {
        "generic": FaissIndex.load(
            generic_index_path
        ),
        "structure_aware": FaissIndex.load(
            structure_index_path
        ),
    }

    print("Loading mappings...")

    mappings = {
        "generic": load_mapping(
            generic_mapping_path
        ),
        "structure_aware": load_mapping(
            structure_mapping_path
        ),
    }

    retriever = Retriever(
        model,
        indexes,
        mappings,
    )

    questions = load_questions(
        questions_path
    )

    print()
    print(f"Loaded benchmark questions: {len(questions)}")

    annotated_questions = [
        question
        for question in questions
        if normalize_ids(
            question.get("relevant_diary_nos", [])
        )
    ]

    print(
        f"Human-annotated questions: "
        f"{len(annotated_questions)}"
    )

    print(
        f"Unannotated questions: "
        f"{len(questions) - len(annotated_questions)}"
    )

    if not annotated_questions:
        print()
        print(
            "No human-annotated questions are available yet."
        )
        print(
            "No retrieval metrics will be fabricated."
        )
        return

    all_rows = []

    for index, question in enumerate(
        questions,
        start=1,
    ):
        question_id = question.get(
            "question_id",
            f"Q{index:03d}",
        )

        print(
            f"[{index}/{len(questions)}] "
            f"{question_id}"
        )

        for strategy in [
            "generic",
            "structure_aware",
        ]:
            row = evaluate_question(
                retriever,
                question,
                strategy,
                args.top_k,
            )

            all_rows.append(row)

    # ---------------------------------------------------------
    # Per-question results
    # ---------------------------------------------------------

    detail_fields = [
        "question_id",
        "strategy",
        "category",
        "era",
        "status",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "ndcg_at_5",
        "ndcg_at_10",
        "retrieved_diaries",
        "top_score",
    ]

    detail_path = (
        OUTPUT_DIR / "retrieval_results.csv"
    )

    write_csv(
        detail_path,
        all_rows,
        detail_fields,
    )

    # ---------------------------------------------------------
    # Overall comparison
    # ---------------------------------------------------------

    overall_rows = []

    for strategy in [
        "generic",
        "structure_aware",
    ]:
        strategy_rows = [
            row
            for row in all_rows
            if row["strategy"] == strategy
        ]

        metrics = aggregate_metrics(
            strategy_rows
        )

        overall_rows.append(
            {
                "strategy": strategy,
                "questions_evaluated": metrics[
                    "questions_evaluated"
                ],
                "recall_at_1": format_metric(
                    metrics["recall_at_1"]
                ),
                "recall_at_3": format_metric(
                    metrics["recall_at_3"]
                ),
                "recall_at_5": format_metric(
                    metrics["recall_at_5"]
                ),
                "recall_at_10": format_metric(
                    metrics["recall_at_10"]
                ),
                "mrr": format_metric(
                    metrics["mrr"]
                ),
                "ndcg_at_5": format_metric(
                    metrics["ndcg_at_5"]
                ),
                "ndcg_at_10": format_metric(
                    metrics["ndcg_at_10"]
                ),
            }
        )

    overall_fields = [
        "strategy",
        "questions_evaluated",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "ndcg_at_5",
        "ndcg_at_10",
    ]

    overall_path = (
        OUTPUT_DIR / "retrieval_metrics.csv"
    )

    write_csv(
        overall_path,
        overall_rows,
        overall_fields,
    )

    # ---------------------------------------------------------
    # Era-wise comparison
    # ---------------------------------------------------------

    eras = sorted(
        {
            str(row["era"])
            for row in all_rows
            if row["status"] == "EVALUATED"
        }
    )

    era_rows = []

    for era in eras:
        for strategy in [
            "generic",
            "structure_aware",
        ]:
            group = [
                row
                for row in all_rows
                if row["status"] == "EVALUATED"
                and row["era"] == era
                and row["strategy"] == strategy
            ]

            metrics = aggregate_metrics(group)

            era_rows.append(
                {
                    "era": era,
                    "strategy": strategy,
                    "questions_evaluated": metrics[
                        "questions_evaluated"
                    ],
                    "recall_at_1": format_metric(
                        metrics["recall_at_1"]
                    ),
                    "recall_at_3": format_metric(
                        metrics["recall_at_3"]
                    ),
                    "recall_at_5": format_metric(
                        metrics["recall_at_5"]
                    ),
                    "recall_at_10": format_metric(
                        metrics["recall_at_10"]
                    ),
                    "mrr": format_metric(
                        metrics["mrr"]
                    ),
                    "ndcg_at_5": format_metric(
                        metrics["ndcg_at_5"]
                    ),
                    "ndcg_at_10": format_metric(
                        metrics["ndcg_at_10"]
                    ),
                }
            )

    era_path = (
        OUTPUT_DIR / "retrieval_by_era.csv"
    )

    write_csv(
        era_path,
        era_rows,
        [
            "era",
            "strategy",
            "questions_evaluated",
            "recall_at_1",
            "recall_at_3",
            "recall_at_5",
            "recall_at_10",
            "mrr",
            "ndcg_at_5",
            "ndcg_at_10",
        ],
    )

    # ---------------------------------------------------------
    # Console summary
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("RETRIEVAL EVALUATION COMPLETE")
    print("=" * 70)

    for row in overall_rows:
        print()
        print(
            f"Strategy: {row['strategy']}"
        )
        print(
            f"Questions: {row['questions_evaluated']}"
        )
        print(
            f"Recall@1:  {row['recall_at_1']}"
        )
        print(
            f"Recall@3:  {row['recall_at_3']}"
        )
        print(
            f"Recall@5:  {row['recall_at_5']}"
        )
        print(
            f"Recall@10: {row['recall_at_10']}"
        )
        print(
            f"MRR:       {row['mrr']}"
        )
        print(
            f"nDCG@5:    {row['ndcg_at_5']}"
        )
        print(
            f"nDCG@10:   {row['ndcg_at_10']}"
        )

    print()
    print(f"Saved: {detail_path}")
    print(f"Saved: {overall_path}")
    print(f"Saved: {era_path}")


if __name__ == "__main__":
    main()