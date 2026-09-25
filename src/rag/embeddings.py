from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Thin wrapper around SentenceTransformer.

    The model name is supplied through configuration so that the
    embedding model can be changed without modifying the pipeline.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        """Return the embedding dimension produced by the model."""
        return self.model.get_embedding_dimension()

    def encode_documents(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        """
        Encode document/chunk text into normalized embeddings.

        Normalization makes inner-product similarity equivalent to
        cosine similarity when using FAISS IndexFlatIP.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single retrieval query.

        Returns shape: (1, embedding_dimension)
        """
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.asarray(embedding, dtype=np.float32)


def load_embedding_model(config: dict) -> EmbeddingModel:
    """
    Construct the embedding model from rag_config.json-style config.
    """
    model_name = config.get("embedding_model_name")

    if not model_name:
        raise ValueError(
            "Missing 'embedding_model_name' in RAG configuration."
        )

    return EmbeddingModel(model_name)