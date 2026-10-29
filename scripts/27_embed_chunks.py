from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import argparse
import json
from pathlib import Path

import numpy as np

from src.rag.config_loader import load_config
from src.rag.embeddings import load_embedding_model


def load_chunks(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


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

    chunk_path = Path(
        config["output_directories"]["chunks"]
    ) / f"{strategy}_chunks.jsonl"

    output_dir = Path("data/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_path = (
        output_dir / f"{strategy}_embeddings.npy"
    )

    mapping_path = (
        output_dir / f"{strategy}_mapping.json"
    )

    print(f"Loading chunks: {chunk_path}")

    chunks = list(load_chunks(chunk_path))

    print(f"Chunks: {len(chunks):,}")

    model = load_embedding_model(config)

    print(
        f"Embedding model: "
        f"{config['embedding_model_name']}"
    )

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode_documents(
        texts,
        batch_size=config["embedding_batch_size"],
        show_progress_bar=True,
    )

    np.save(
        embedding_path,
        embeddings,
    )

    mapping = [
        {
            key: chunk[key]
            for key in [
                "chunk_id",
                "diary_no",
                "judgment_date",
                "year",
                "decade",
                "chunking_strategy",
                "chunk_index",
                "text",
                "start_position",
                "end_position",
                "section_label",
            ]
        }
        for chunk in chunks
    ]

    with mapping_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            mapping,
            f,
            ensure_ascii=False,
        )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    print(
        f"Saved embeddings: {embedding_path}"
    )

    print(
        f"Saved mapping: {mapping_path}"
    )


if __name__ == "__main__":
    main()