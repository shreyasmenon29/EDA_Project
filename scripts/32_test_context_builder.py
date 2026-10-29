import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.context_builder import ContextBuilder


def main():
    fake_results = [
        {
            "rank": 1,
            "score": 0.8123,
            "diary_no": "12345",
            "judgment_date": "12-04-2018",
            "chunk_id": "chunk_001",
            "section_label": "Held",
            "text": (
                "The court considered the principles governing "
                "remoteness of damages in contractual disputes."
            ),
        },
        {
            "rank": 2,
            "score": 0.7911,
            "diary_no": "67890",
            "judgment_date": "21-08-2015",
            "chunk_id": "chunk_002",
            "section_label": "Reasoning",
            "text": (
                "Damages must be sufficiently connected to the "
                "breach and cannot be purely remote consequences."
            ),
        },
    ]

    builder = ContextBuilder(
        max_chunks=8,
        max_chars=24000,
    )

    result = builder.build(fake_results)

    print("=" * 80)
    print("CONTEXT BUILDER TEST")
    print("=" * 80)

    print(f"Chunks: {result['num_chunks']}")
    print(f"Judgments: {result['num_judgments']}")
    print(f"Characters: {result['total_characters']}")

    print("\n" + "=" * 80)
    print("GENERATED CONTEXT")
    print("=" * 80)

    print(result["context"])

    print("\n" + "=" * 80)
    print("SOURCES")
    print("=" * 80)

    for source in result["sources"]:
        print(source)


if __name__ == "__main__":
    main()