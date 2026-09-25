from pathlib import Path

import numpy as np

from src.rag.config_loader import load_config
from src.rag.embeddings import load_embedding_model


CONFIG = load_config(Path("config/rag_config.json"))


def test_embedding_model_loads():
    model = load_embedding_model(CONFIG)

    assert model.dimension > 0


def test_document_embedding_shape_and_normalization():
    model = load_embedding_model(CONFIG)

    embeddings = model.encode_documents(
        [
            "The Supreme Court considered a constitutional question.",
            "The court examined the legal provisions applicable to the case.",
        ],
        batch_size=2,
        show_progress_bar=False,
    )

    assert embeddings.shape == (2, model.dimension)
    assert embeddings.dtype == np.float32

    norms = np.linalg.norm(embeddings, axis=1)

    assert np.allclose(norms, 1.0, atol=1e-5)


def test_query_embedding_shape_and_normalization():
    model = load_embedding_model(CONFIG)

    embedding = model.encode_query(
        "What constitutional issue did the court consider?"
    )

    assert embedding.shape == (1, model.dimension)

    norm = np.linalg.norm(embedding[0])

    assert np.isclose(norm, 1.0, atol=1e-5)