import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.pipeline import RAGPipeline


def main():

    question = (
        "What principles govern remoteness of damages "
        "in contract disputes?"
    )

    print("=" * 80)
    print("INITIALIZING RAG PIPELINE")
    print("=" * 80)

    pipeline = RAGPipeline()

    print("\nAvailable strategies:")
    for strategy in pipeline.available_strategies():
        print(f"  - {strategy}")

    print("\n" + "=" * 80)
    print("RUNNING RETRIEVAL")
    print("=" * 80)

    result = pipeline.prepare(
        question=question,
        strategy="generic",
        top_k=5,
    )

    print(
        f"\nRetrieved chunks: "
        f"{len(result['retrieval_results'])}"
    )

    print(
        f"Unique judgments: "
        f"{result['context']['num_judgments']}"
    )

    print(
        f"Context characters: "
        f"{result['context']['total_characters']}"
    )

    print("\n" + "=" * 80)
    print("SOURCES")
    print("=" * 80)

    for source in result["context"]["sources"]:
        print(
            f"[Source {source['source_number']}] "
            f"Diary={source['diary_no']} | "
            f"Date={source['judgment_date']} | "
            f"Score={source['score']:.4f}"
        )

    print("\n" + "=" * 80)
    print("GENERATED USER PROMPT")
    print("=" * 80)

    print(result["prompt"]["user"])


if __name__ == "__main__":
    main()