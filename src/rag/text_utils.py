import re
from typing import List, Tuple


def normalize_whitespace(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\\n", "\n").replace("\\t", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def safe_segment(text: str, start: int, end: int) -> str:
    return text[start:end].strip()


def split_on_paragraphs(text: str) -> List[str]:
    blocks = re.split(r"\n\s*\n+", text)
    return [b.strip() for b in blocks if b.strip()]


def find_numbered_paragraph_boundaries(text: str) -> List[int]:
    positions: List[int] = []
    for match in re.finditer(r"(?m)^(?:\s*\d{1,4}[\.)]\s+)", text):
        positions.append(match.start())
    return positions


def find_header_boundaries(text: str) -> List[int]:
    markers = [
        "CORAM",
        "JUDGMENT",
        "ORDER",
        "HEADNOTE",
        "PETITIONER",
        "RESPONDENT",
        "BENCH",
        "FACTS",
        "ISSUES",
        "HELD",
    ]
    positions: List[int] = []
    for marker in markers:
        for match in re.finditer(rf"(?im)\b{re.escape(marker)}\b", text):
            positions.append(match.start())
    return sorted(set(positions))
