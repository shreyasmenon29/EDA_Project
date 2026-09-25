from __future__ import annotations

from pathlib import Path

from src.rag.embeddings import EmbeddingModel
from src.rag.faiss_index import FaissIndex, load_mapping


class Retriever:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        indexes: dict[str, FaissIndex],
        mappings: dict[str, list[dict]],
    ):
        self.embedding_model = embedding_model
        self.indexes = indexes
        self.mappings = mappings

    def retrieve(
        self,
        query: str,
        strategy: str,
        top_k: int = 5,
    ) -> list[dict]:

        if strategy not in self.indexes:
            raise ValueError(
                f"Unknown strategy: {strategy}"
            )

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        query_embedding = self.embedding_model.encode_query(query)

        scores, indices = self.indexes[strategy].search(
            query_embedding,
            top_k,
        )

        mapping = self.mappings[strategy]

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):
            if index < 0:
                continue

            item = mapping[int(index)]

            results.append(
                {
                    "rank": len(results) + 1,
                    "score": float(score),
                    "chunk_id": item["chunk_id"],
                    "diary_no": item["diary_no"],
                    "judgment_date": item["judgment_date"],
                    "year": item["year"],
                    "decade": item["decade"],
                    "chunking_strategy": item[
                        "chunking_strategy"
                    ],
                    "chunk_index": item["chunk_index"],
                    "text": item["text"],
                    "start_position": item[
                        "start_position"
                    ],
                    "end_position": item[
                        "end_position"
                    ],
                    "section_label": item.get(
                        "section_label"
                    ),
                }
            )

        return results