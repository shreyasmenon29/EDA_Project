from __future__ import annotations

from pathlib import Path
from typing import Any

from src.rag.config_loader import load_config
from src.rag.embeddings import load_embedding_model
from src.rag.faiss_index import FaissIndex, load_mapping
from src.rag.retrieval import Retriever
from src.rag.context_builder import ContextBuilder
from src.rag.generator import RAGGenerator


class RAGPipeline:
    """
    End-to-end RAG pipeline:

        Question
            ↓
        Retrieval
            ↓
        Relevance assessment
            ↓
        ┌───────────────────────┐
        │ Relevant?             │
        ├───────────┬───────────┤
        │ YES       │ NO        │
        ↓           ↓
        Grounded    General
        RAG         Legal Answer
        └───────────┴───────────┘
    """

    def __init__(
        self,
        config_path: str | Path = "config/rag_config.json",
        max_chunks: int = 8,
        max_chars: int = 24000,
    ):
        self.config = load_config(Path(config_path))

        self.model = load_embedding_model(self.config)

        self.indexes: dict[str, FaissIndex] = {}
        self.mappings: dict[str, Any] = {}

        self._load_available_indexes()

        self.retriever = Retriever(
            self.model,
            self.indexes,
            self.mappings,
        )

        self.context_builder = ContextBuilder(
            max_chunks=max_chunks,
            max_chars=max_chars,
        )

        self.generator = RAGGenerator()

        # ----------------------------------------------------
        # RELEVANCE SETTINGS
        # ----------------------------------------------------
        #
        # A retrieved passage with a similarity score below
        # this threshold is treated as insufficiently relevant.
        #
        # 0.45 is intentionally conservative for the current
        # MiniLM + FAISS retrieval setup.
        #
        validation_config = self.config.get(
            "validation",
            {},
        )

        self.relevance_threshold = float(
            validation_config.get(
                "retrieval_relevance_threshold",
                0.45,
            )
        )

        self.min_relevant_results = int(
            validation_config.get(
                "min_relevant_results",
                1,
            )
        )

    # ========================================================
    # LOAD FAISS INDEXES
    # ========================================================

    def _load_available_indexes(self) -> None:
        """
        Load all available FAISS indexes.
        """

        faiss_dir = Path("data/faiss")
        embedding_dir = Path("data/embeddings")

        for strategy in [
            "generic",
            "structure_aware",
        ]:

            index_path = (
                faiss_dir /
                f"{strategy}.index"
            )

            mapping_path = (
                embedding_dir /
                f"{strategy}_mapping.json"
            )

            if (
                index_path.exists()
                and mapping_path.exists()
            ):

                self.indexes[strategy] = (
                    FaissIndex.load(
                        index_path
                    )
                )

                self.mappings[strategy] = (
                    load_mapping(
                        mapping_path
                    )
                )

    # ========================================================
    # AVAILABLE STRATEGIES
    # ========================================================

    def available_strategies(
        self,
    ) -> list[str]:

        return sorted(
            self.indexes.keys()
        )

    # ========================================================
    # RETRIEVAL
    # ========================================================

    def retrieve(
        self,
        question: str,
        strategy: str = "generic",
        top_k: int = 5,
    ) -> list[dict[str, Any]]:

        if strategy not in self.indexes:

            available = ", ".join(
                self.available_strategies()
            )

            raise ValueError(
                f"Strategy '{strategy}' is not available. "
                f"Available strategies: "
                f"{available or 'none'}"
            )

        return self.retriever.retrieve(
            question,
            strategy,
            top_k,
        )

    # ========================================================
    # RELEVANCE ASSESSMENT
    # ========================================================

    def _assess_relevance(
        self,
        retrieval_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Determine whether the retrieved context is relevant
        enough to support a grounded RAG answer.
        """

        if not retrieval_results:

            return {
                "is_relevant": False,
                "max_score": None,
                "relevant_count": 0,
            }

        scores: list[float] = []

        for result in retrieval_results:

            score = result.get(
                "score"
            )

            if score is None:
                continue

            try:

                scores.append(
                    float(score)
                )

            except (
                TypeError,
                ValueError,
            ):

                continue

        if not scores:

            return {
                "is_relevant": False,
                "max_score": None,
                "relevant_count": 0,
            }

        relevant_scores = [
            score
            for score in scores
            if score >= self.relevance_threshold
        ]

        max_score = max(scores)

        is_relevant = (
            len(relevant_scores)
            >= self.min_relevant_results
        )

        return {
            "is_relevant": is_relevant,
            "max_score": max_score,
            "relevant_count": len(
                relevant_scores
            ),
        }

    # ========================================================
    # PREPARE
    # ========================================================

    def prepare(
        self,
        question: str,
        strategy: str = "generic",
        top_k: int = 5,
    ) -> dict[str, Any]:

        retrieval_results = self.retrieve(
            question=question,
            strategy=strategy,
            top_k=top_k,
        )

        relevance = self._assess_relevance(
            retrieval_results
        )

        # ----------------------------------------------------
        # RELEVANT RETRIEVAL
        # ----------------------------------------------------

        if relevance["is_relevant"]:

            context_result = (
                self.context_builder.build(
                    retrieval_results
                )
            )

            prompt = (
                self.generator.build_prompt(
                    question=question,
                    context_result=context_result,
                )
            )

            return {
                "question": question,
                "strategy": strategy,
                "retrieval_results": (
                    retrieval_results
                ),
                "context": context_result,
                "answer_mode": "grounded_rag",
                "relevance": relevance,
                "prompt": {
                    "system": (
                        prompt.system_prompt
                    ),
                    "user": (
                        prompt.user_prompt
                    ),
                },
            }

        # ----------------------------------------------------
        # IRRELEVANT RETRIEVAL
        # ----------------------------------------------------
        #
        # We still preserve the retrieval results so the UI
        # can show what the retrieval system found.
        #
        # However, those passages are NOT passed to Gemini
        # as evidence for the answer.
        # ----------------------------------------------------

        return {
            "question": question,
            "strategy": strategy,
            "retrieval_results": (
                retrieval_results
            ),
            "context": {
                "context": "",
                "sources": [],
            },
            "answer_mode": "general_legal",
            "relevance": relevance,
            "prompt": None,
        }

    # ========================================================
    # ASK
    # ========================================================

    def ask(
        self,
        question: str,
        strategy: str = "structure_aware",
        top_k: int = 5,
    ) -> dict[str, Any]:

        prepared = self.prepare(
            question=question,
            strategy=strategy,
            top_k=top_k,
        )

        # ----------------------------------------------------
        # GROUNDED RAG
        # ----------------------------------------------------

        if (
            prepared["answer_mode"]
            == "grounded_rag"
        ):

            answer = (
                self.generator.generate(
                    question=question,
                    context_result=(
                        prepared["context"]
                    ),
                )
            )

        # ----------------------------------------------------
        # GENERAL LEGAL FALLBACK
        # ----------------------------------------------------

        else:

            answer = (
                self.generator.generate_general(
                    question=question
                )
            )

        prepared["answer"] = answer

        return prepared