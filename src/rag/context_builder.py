from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from src.rag.metadata import get_document_metadata


@dataclass
class ContextChunk:
    rank: int
    score: float
    diary_no: str
    case_name: str
    citation: str
    url: str | None
    year: str
    judgment_date: str
    chunk_id: str
    section_label: str
    text: str


class ContextBuilder:
    """
    Converts retrieval results into a clean context block for
    RAG generation.

    Important:

    - Does NOT generate an answer.
    - Does NOT alter the retrieved text.
    - Preserves provenance for every chunk.
    - Removes duplicate chunks/judgments when possible.
    - Applies a context-size limit.
    - Exposes retrieval statistics for the application.
    """

    def __init__(
        self,
        max_chunks: int = 8,
        max_chars: int = 24000,
    ):
        self.max_chunks = max_chunks
        self.max_chars = max_chars

    def _normalize_result(
        self,
        result: dict[str, Any],
    ) -> ContextChunk:

        diary_no = str(result.get("diary_no", "")).strip()
        doc_meta = get_document_metadata(diary_no)

        case_name = result.get("case_name") or doc_meta.get("case_name", "Case name unavailable")
        citation = result.get("citation") or doc_meta.get("citation", "Citation unavailable")
        url = result.get("url") if "url" in result else doc_meta.get("url")
        year = str(result.get("year") or doc_meta.get("year", "Year unavailable"))
        judgment_date = str(result.get("judgment_date") or doc_meta.get("judgment_date", "Date unavailable"))

        return ContextChunk(
            rank=int(result.get("rank", 0)),
            score=float(result.get("score", 0.0)),
            diary_no=diary_no,
            case_name=case_name,
            citation=citation,
            url=url,
            year=year,
            judgment_date=judgment_date,
            chunk_id=str(result.get("chunk_id", "")),
            section_label=str(result.get("section_label", "")),
            text=str(result.get("text", "")).strip(),
        )

    def _deduplicate(
        self,
        chunks: Iterable[ContextChunk],
    ) -> list[ContextChunk]:

        seen_chunk_ids: set[str] = set()
        seen_text: set[str] = set()

        unique_chunks: list[ContextChunk] = []

        for chunk in chunks:

            if not chunk.text:
                continue

            if (
                chunk.chunk_id
                and chunk.chunk_id in seen_chunk_ids
            ):
                continue

            normalized_text = " ".join(
                chunk.text.split()
            )

            if normalized_text in seen_text:
                continue

            if chunk.chunk_id:
                seen_chunk_ids.add(
                    chunk.chunk_id
                )

            seen_text.add(
                normalized_text
            )

            unique_chunks.append(
                chunk
            )

        return unique_chunks

    def build(
        self,
        retrieval_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Build structured context from retrieval results.

        Returns:

            {
                "context": str,
                "sources": list[dict],
                "num_chunks": int,
                "num_judgments": int,
                "total_characters": int,
                "best_score": float,
                "average_score": float
            }
        """

        chunks = [
            self._normalize_result(result)
            for result in retrieval_results
        ]

        chunks = self._deduplicate(
            chunks
        )

        # Retriever results are already ranked.
        # Keep their order rather than re-ranking them.
        chunks = chunks[: self.max_chunks]

        context_parts: list[str] = []

        sources: list[dict[str, Any]] = []

        current_chars = 0

        for index, chunk in enumerate(
            chunks,
            start=1,
        ):

            url_display = chunk.url if chunk.url else "Judgment link unavailable"

            header = (
                f"[SOURCE {index}]\n"
                f"Case: {chunk.case_name} ({chunk.year})\n"
                f"Citation: {chunk.citation}\n"
                f"Judgment URL: {url_display}\n"
                f"Diary Number: {chunk.diary_no}\n"
                f"Judgment Date: {chunk.judgment_date}\n"
                f"Section: {chunk.section_label}\n"
                f"Retrieval Score: {chunk.score:.4f}\n"
                f"Chunk ID: {chunk.chunk_id}"
            )

            body = f"\n{chunk.text}"

            block = header + body

            remaining = (
                self.max_chars
                - current_chars
            )

            if remaining <= 0:
                break

            if len(block) > remaining:

                block = (
                    block[:remaining]
                    .rstrip()
                )

                if not block:
                    break

                block += "\n[TRUNCATED]"

            context_parts.append(
                block
            )

            current_chars += len(block)

            sources.append(
                {
                    "source_number": index,
                    "diary_no": chunk.diary_no,
                    "case_name": chunk.case_name,
                    "citation": chunk.citation,
                    "url": chunk.url,
                    "year": chunk.year,
                    "judgment_date": chunk.judgment_date,
                    "section_label": chunk.section_label,
                    "chunk_id": chunk.chunk_id,
                    "score": chunk.score,
                    "text": chunk.text,
                }
            )

        if context_parts:

            context = (
                "\n\n"
                + (
                    "\n\n"
                    + ("-" * 70)
                    + "\n\n"
                ).join(
                    context_parts
                )
            )

        else:
            context = ""

        unique_diaries = {
            source["diary_no"]
            for source in sources
            if source["diary_no"]
        }

        scores = [
            float(source["score"])
            for source in sources
        ]

        best_score = (
            max(scores)
            if scores
            else 0.0
        )

        average_score = (
            sum(scores) / len(scores)
            if scores
            else 0.0
        )

        final_context = context.strip()

        return {
            "context": final_context,
            "sources": sources,
            "num_chunks": len(sources),
            "num_judgments": len(
                unique_diaries
            ),
            "total_characters": len(
                final_context
            ),
            "best_score": best_score,
            "average_score": average_score,
        }
