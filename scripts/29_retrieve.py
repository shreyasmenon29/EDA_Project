from __future__ import annotations

import argparse
from pathlib import Path

from src.rag.config_loader import load_config
from src.rag.embeddings import load_embedding_model
from src.rag.faiss_index import (
    FaissIndex,
    load_mapping,
)
from src.rag.retrieval import Retriever


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        required=True,
    )

    parser.add_argument(
        "--strategy",
        choices=["generic", "structure_aware"],
        required=True,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    config = load_config(
        Path("config/rag_config.json")
    )

    model = load_embedding_model(config)

    indexes = {}
    mappings = {}

    for strategy in [
        "generic",
        "structure_aware",
    ]:
        indexes[strategy] = FaissIndex.load(
            Path("data/faiss")
            / f"{strategy}.index"
        )

        mappings[strategy] = load_mapping(
            Path("data/embeddings")
            / f"{strategy}_mapping.json"
        )

    retriever = Retriever(
        model,
        indexes,
        mappings,
    )

    results = retriever.retrieve(
        args.query,
        args.strategy,
        args.top_k,
    )

    for result in results:
        print("=" * 80)
        print(
            f"Rank: {result['rank']}"
        )
        print(
            f"Score: {result['score']:.4f}"
        )
        print(
            f"Diary: {result['diary_no']}"
        )
        print(
            f"Date: {result['judgment_date']}"
        )
        print(
            f"Chunk: {result['chunk_id']}"
        )
        print(
            f"Section: {result['section_label']}"
        )
        print()
        print(result["text"][:1000])


if __name__ == "__main__":
    main()