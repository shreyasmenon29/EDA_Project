import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.generator import RAGGenerator


def main():
    generator = RAGGenerator()

    context_result = {
        "context": """
[SOURCE 1]
Diary Number: 12345
Judgment Date: 12-04-2018
Section: Held
Retrieval Score: 0.8123
Chunk ID: chunk_001

The court considered the principles governing remoteness
of damages in contractual disputes.

[SOURCE 2]
Diary Number: 67890
Judgment Date: 21-08-2015
Section: Reasoning
Retrieval Score: 0.7911
Chunk ID: chunk_002

Damages must be sufficiently connected to the breach and
cannot be purely remote consequences.
""",
        "sources": [
            {
                "source_number": 1,
                "diary_no": "12345",
                "judgment_date": "12-04-2018",
                "section_label": "Held",
                "chunk_id": "chunk_001",
                "score": 0.8123,
            },
            {
                "source_number": 2,
                "diary_no": "67890",
                "judgment_date": "21-08-2015",
                "section_label": "Reasoning",
                "chunk_id": "chunk_002",
                "score": 0.7911,
            },
        ],
    }

    question = "What principles govern remoteness of damages?"

    result = generator.build_prompt(
        question=question,
        context_result=context_result,
    )

    print("=" * 80)
    print("SYSTEM PROMPT")
    print("=" * 80)
    print(result.system_prompt)

    print("\n" + "=" * 80)
    print("USER PROMPT")
    print("=" * 80)
    print(result.user_prompt)


if __name__ == "__main__":
    main()