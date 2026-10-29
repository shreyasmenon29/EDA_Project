from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import argparse
from pathlib import Path

import numpy as np

from src.rag.config_loader import load_config
from src.rag.faiss_index import (
    FaissIndex,
    load_mapping,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--strategy",
        choices=["generic", "structure_aware"],
        required=True,
    )

    args = parser.parse_args()

    config = load_config(
        Path("config/rag_config.json")
    )

    strategy = args.strategy

    embedding_dir = Path("data/embeddings")

    embeddings = np.load(
        embedding_dir
        / f"{strategy}_embeddings.npy"
    )

    mapping = load_mapping(
        embedding_dir
        / f"{strategy}_mapping.json"
    )

    if len(mapping) != len(embeddings):
        raise ValueError(
            "Embedding count does not match mapping count."
        )

    index = FaissIndex(
        dimension=embeddings.shape[1]
    )

    index.add(embeddings)

    output_dir = Path("data/faiss")
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = (
        output_dir / f"{strategy}.index"
    )

    index.save(index_path)

    print(f"Strategy: {strategy}")
    print(f"Vectors: {index.size:,}")
    print(
        f"Dimension: {embeddings.shape[1]}"
    )
    print(
        f"FAISS index: {index_path}"
    )


if __name__ == "__main__":
    main()