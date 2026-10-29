from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from dotenv import load_dotenv
from google import genai


load_dotenv()


@dataclass
class RAGPrompt:
    system_prompt: str
    user_prompt: str


def format_answer_citations(answer: str, num_sources: int) -> str:
    """
    Replace generic [Source X] and [X] citations in the answer text with
    clickable anchor links pointing directly to the corresponding source
    in the supporting judgments section.
    """
    if not answer:
        return answer

    def _replace_source_label(match: re.Match) -> str:
        idx = match.group(1)
        return (
            f'<a href="#source-{idx}" '
            f'style="color: #d9b662; font-weight: 600; text-decoration: underline;">'
            f'[{idx}]</a>'
        )

    # Replace [Source 1], [Source 2], etc.
    formatted = re.sub(
        r"\[Source\s+(\d+)\]",
        _replace_source_label,
        answer,
        flags=re.IGNORECASE,
    )

    # Replace standalone [1], [2], etc. where 1 <= idx <= num_sources
    def _replace_standalone_num(match: re.Match) -> str:
        val = match.group(1)
        try:
            idx = int(val)
            if 1 <= idx <= num_sources:
                return (
                    f'<a href="#source-{idx}" '
                    f'style="color: #d9b662; font-weight: 600; text-decoration: underline;">'
                    f'[{idx}]</a>'
                )
        except ValueError:
            pass
        return match.group(0)

    formatted = re.sub(
        r"(?<![>\[\(#])\[(\d{1,2})\](?![<\]\)])",
        _replace_standalone_num,
        formatted,
    )

    return formatted


class RAGGenerator:
    """
    Gemini generator for the Legal Intelligence System.

    The model follows a hybrid answering strategy:

    1. If retrieved Supreme Court judgments are relevant,
       use them as evidence and cite them with precise case names,
       years, and numbered citations.

    2. If retrieved judgments are irrelevant or insufficient,
       DO NOT refuse to answer.
       Answer the question using general legal knowledge.

    The model must never fabricate citations, case names, or pretend that
    general knowledge came from the retrieved dataset.
    """

    SYSTEM_PROMPT = """
You are an intelligent legal research assistant specializing
in Indian law and Indian Supreme Court judgments.

You have been given:

1. A user's legal question.
2. Retrieved passages from a local Supreme Court judgment
   dataset with verified metadata (Case Name, Year, Citation,
   Official URL, Diary Number).

Your job is to ALWAYS provide a useful answer to the user's
question.

IMPORTANT ANSWERING RULES:

1. FIRST examine the retrieved judgment context.

2. If the retrieved context is relevant to the question:
   - Use it as primary evidence for your answer.
   - Refer to cases by their real case name and year as provided in the metadata.
   - Cite each supporting passage using numbered citations like [1], [2] (matching [Source 1], [Source 2]).
   - Ensure each citation points strictly to the actual document that supports the relevant statement.
   - Do NOT assume a source refers to a case without checking its metadata.

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
   - URLs

   If a citation or URL is noted as unavailable in the metadata,
   do not invent one.

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

11. Never fabricate a citation number or source link.

12. Do not provide legal advice.

13. Be concise, direct, and useful.

14. NEVER respond merely by saying that the retrieved context
    is insufficient.

The desired behavior is:

RELEVANT RETRIEVED CONTEXT
→ Answer using the retrieved evidence
→ Cite relevant sources using [1], [2] with real case names
→ Accurately reflect the legal holdings

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
        Build a hybrid legal research prompt with verified source metadata.
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
            num = source.get("source_number", 1)
            case_name = source.get("case_name", "Case name unavailable")
            year = source.get("year", "Year unavailable")
            citation = source.get("citation", "Citation unavailable")
            url = source.get("url") or "Judgment link unavailable"
            diary_no = source.get("diary_no", "N/A")
            judgment_date = source.get("judgment_date", "N/A")
            section = source.get("section_label", "N/A")

            source_lines.append(
                f"[Source {num}] {case_name} ({year})\n"
                f"   - Verified Citation: {citation}\n"
                f"   - Official URL: {url}\n"
                f"   - Diary No: {diary_no} | Date: {judgment_date} | Section: {section}"
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
- Refer to each supporting judgment by its real case name and year as given above.
- Cite the relevant passages using numbered citations like [1], [2] (corresponding to [Source 1], [Source 2]).
- Ensure each citation corresponds to the actual document that supports the relevant statement.
- Do not invent citations or case details if they are marked unavailable.

CASE 2 — IRRELEVANT OR INSUFFICIENT CONTEXT

If the retrieved judgments are unrelated to the question:

DO NOT stop.
DO NOT say that the context is insufficient.
DO NOT say that the dataset does not contain the answer.
DO NOT repeat the unrelated retrieved judgments.

Instead:
- Answer the user's question using your general legal knowledge.
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

============================================================
OUTPUT STYLE
============================================================

Give a direct answer first.

Then provide the relevant legal explanation citing supporting cases.

If retrieved sources genuinely support the answer, include
a short "Supporting judgments" section citing the cases by name with [1], [2] tags.

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

            raw_answer = interaction.output_text

            if not raw_answer:

                return (
                    "The model did not return an answer. "
                    "Please try the question again."
                )

            # Linkify citations [Source X] and [X] to jump to the source cards
            sources = context_result.get("sources", [])
            formatted_answer = format_answer_citations(
                raw_answer.strip(),
                num_sources=len(sources),
            )

            return formatted_answer

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
