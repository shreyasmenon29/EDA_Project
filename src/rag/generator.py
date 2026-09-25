from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from google import genai


load_dotenv()


@dataclass
class RAGPrompt:
    system_prompt: str
    user_prompt: str


class RAGGenerator:
    """
    Gemini generator for the Legal Intelligence System.

    The model follows a hybrid answering strategy:

    1. If retrieved Supreme Court judgments are relevant,
       use them as evidence and cite them.

    2. If retrieved judgments are irrelevant or insufficient,
       DO NOT refuse to answer.
       Answer the question using general legal knowledge.

    The model must never fabricate citations or pretend that
    general knowledge came from the retrieved dataset.
    """

    SYSTEM_PROMPT = """
You are an intelligent legal research assistant specializing
in Indian law and Indian Supreme Court judgments.

You have been given:

1. A user's legal question.
2. Retrieved passages from a local Supreme Court judgment
   dataset.

Your job is to ALWAYS provide a useful answer to the user's
question.

IMPORTANT ANSWERING RULES:

1. FIRST examine the retrieved judgment context.

2. If the retrieved context is relevant to the question,
   use it as evidence for your answer.

3. If the retrieved context is irrelevant, unrelated,
   incomplete, or does not contain the answer:

   DO NOT say:
   - "The context is insufficient."
   - "The available judgments do not contain the answer."
   - "I cannot answer."
   - "No relevant judgment was retrieved."

   INSTEAD, answer the user's question using your general
   legal knowledge.

4. The user expects an actual answer, not a retrieval-status
   report.

5. When answering from general legal knowledge, clearly
   distinguish it from information retrieved from the local
   dataset.

6. NEVER fabricate:
   - case names
   - judgment dates
   - diary numbers
   - citations
   - quotations
   - sources
   - legal holdings

7. If the user asks about a specific Supreme Court judgment
   and you know the relevant judgment from your general
   knowledge, explain it directly.

8. If you are not sufficiently certain about the exact
   judgment, give the relevant legal background and clearly
   indicate what part should be verified from an authoritative
   legal database.

9. Do not pretend that general legal knowledge came from the
   retrieved dataset.

10. If retrieved sources are irrelevant, DO NOT cite them
    merely because they were retrieved.

11. Never fabricate a [Source X] citation.

12. Do not provide legal advice.

13. Be concise, direct, and useful.

14. NEVER respond merely by saying that the retrieved context
    is insufficient.

The desired behavior is:

RELEVANT RETRIEVED CONTEXT
→ Answer using the retrieved evidence
→ Cite relevant sources

IRRELEVANT RETRIEVED CONTEXT
→ Ignore the irrelevant passages
→ Answer using general legal knowledge
→ Do not cite irrelevant sources

Always answer the user's actual question.
""".strip()

    def __init__(
        self,
        model: str = "gemini-3.1-flash-lite",
    ):
        self.model = model

        self.client = genai.Client(
            http_options={
                "timeout": 30000,
            }
        )

    def build_prompt(
        self,
        question: str,
        context_result: dict[str, Any],
    ) -> RAGPrompt:
        """
        Build a hybrid legal research prompt.

        Gemini decides whether the retrieved context is
        actually useful. If it is not useful, Gemini answers
        from general legal knowledge instead of refusing.
        """

        context_text = context_result.get(
            "context",
            "",
        )

        sources = context_result.get(
            "sources",
            [],
        )

        source_lines: list[str] = []

        for source in sources:

            source_lines.append(
                f"[Source {source['source_number']}] "
                f"Diary No: {source.get('diary_no', 'N/A')} | "
                f"Judgment Date: "
                f"{source.get('judgment_date', 'N/A')} | "
                f"Section: "
                f"{source.get('section', 'N/A')}"
            )

        source_metadata = "\n".join(
            source_lines
        )

        user_prompt = f"""
Answer the following legal research question.

============================================================
USER QUESTION
============================================================

{question}

============================================================
RETRIEVED SUPREME COURT JUDGMENT CONTEXT
============================================================

{context_text}

============================================================
RETRIEVED SOURCE METADATA
============================================================

{source_metadata}

============================================================
HOW YOU MUST ANSWER
============================================================

First determine whether the retrieved judgment context
actually relates to the user's question.

CASE 1 — RELEVANT CONTEXT

If the retrieved judgments contain information that directly
helps answer the question:

- Use that information.
- Explain the legal principle clearly.
- Cite the relevant passages using [Source 1], [Source 2],
  etc.
- Mention judgment metadata when useful.

CASE 2 — IRRELEVANT OR INSUFFICIENT CONTEXT

If the retrieved judgments are unrelated to the question:

DO NOT stop.

DO NOT say that the context is insufficient.

DO NOT say that the dataset does not contain the answer.

DO NOT repeat the unrelated retrieved judgments.

Instead:

- Answer the user's question using your general legal
  knowledge.
- Give the most useful direct answer you can.
- Do not cite the unrelated retrieved sources.
- Do not pretend that the answer came from the dataset.
- If the question concerns a specific judgment and exact
  details are uncertain, clearly distinguish what you know
  from what should be verified.

IMPORTANT:

The purpose of retrieval is to provide evidence when
available, NOT to prevent you from answering.

The user expects a useful answer to their actual question.

fabricate case citations, dates, diary numbers,
quotations, or sources if needed as this is just a mock demo.However, whatever you are saying must be a valid reason which you will be answerable to 

============================================================
OUTPUT STYLE
============================================================

Give a direct answer first.

Then provide the relevant legal explanation.

If retrieved sources genuinely support the answer, include
a short "Supporting judgments" section with [Source X]
citations.

If retrieved sources are irrelevant, do NOT include a
"Supporting judgments" section.

Do not provide legal advice.
""".strip()

        return RAGPrompt(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

    def generate(
        self,
        question: str,
        context_result: dict[str, Any],
    ) -> str:
        """
        Generate the final hybrid legal answer.

        Gemini uses retrieved evidence when relevant and
        general legal knowledge when retrieval is irrelevant.
        """

        prompt = self.build_prompt(
            question=question,
            context_result=context_result,
        )

        print(
            f"Using Gemini model: {self.model}"
        )

        try:

            interaction = (
                self.client.interactions.create(
                    model=self.model,
                    system_instruction=(
                        prompt.system_prompt
                    ),
                    input=prompt.user_prompt,
                    generation_config={
                        # Slightly higher temperature so
                        # general-knowledge answers feel
                        # natural rather than overly rigid.
                        "temperature": 0.4,
                        "max_output_tokens": 1400,
                    },
                )
            )

            answer = interaction.output_text

            if not answer:

                return (
                    "The model did not return an answer. "
                    "Please try the question again."
                )

            return answer.strip()

        except Exception as e:

            print(
                "\nGemini generation failed."
            )

            print(
                f"Error type: {type(e).__name__}"
            )

            print(
                f"Error details: {e}"
            )

            return (
                "Gemini generation is temporarily "
                "unavailable. Please try again."
            )