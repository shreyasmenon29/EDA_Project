from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


class FaissIndex:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)

    def add(self, embeddings: np.ndarray) -> None:
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected dimension {self.dimension}, "
                f"got {embeddings.shape[1]}"
            )

        self.index.add(
            np.asarray(embeddings, dtype=np.float32)
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
    ):
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.shape[1] != self.dimension:
            raise ValueError("Query embedding dimension mismatch.")

        scores, indices = self.index.search(
            np.asarray(query_embedding, dtype=np.float32),
            top_k,
        )

        return scores, indices

    @property
    def size(self) -> int:
        return self.index.ntotal

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path):
        index = faiss.read_index(str(path))

        obj = cls(index.d)
        obj.index = index

        return obj


def save_mapping(mapping: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def load_mapping(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)