from __future__ import annotations

from pathlib import Path

from src.rag.embeddings import EmbeddingModel
from src.rag.faiss_index import FaissIndex, load_mapping
from src.rag.metadata import get_document_metadata


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
            diary_no = item.get("diary_no", "")
            doc_meta = get_document_metadata(diary_no)

            results.append(
                {
                    "rank": len(results) + 1,
                    "score": float(score),
                    "chunk_id": item["chunk_id"],
                    "diary_no": diary_no,
                    "case_name": doc_meta.get("case_name", "Case name unavailable"),
                    "citation": doc_meta.get("citation", "Citation unavailable"),
                    "url": doc_meta.get("url"),
                    "judgment_date": (
                        doc_meta.get("judgment_date")
                        if doc_meta.get("judgment_date") != "Date unavailable"
                        else item.get("judgment_date", "Unknown")
                    ),
                    "year": (
                        doc_meta.get("year")
                        if doc_meta.get("year") != "Year unavailable"
                        else item.get("year", "Unknown")
                    ),
                    "decade": item.get("decade", "Unknown"),
                    "chunking_strategy": item.get("chunking_strategy", strategy),
                    "chunk_index": item.get("chunk_index", 0),
                    "text": item["text"],
                    "start_position": item.get("start_position", 0),
                    "end_position": item.get("end_position", 0),
                    "section_label": item.get("section_label", "general"),
                }
            )

        return results
