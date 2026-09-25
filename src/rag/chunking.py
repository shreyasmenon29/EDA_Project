import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .chunk_models import ChunkRecord, chunk_metadata_for, make_chunk_id
from .text_utils import normalize_whitespace, split_on_paragraphs, word_count


SECTION_MARKERS = [
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


def _parse_year(date_value: Any) -> int:
    if date_value is None:
        return 0
    try:
        text = str(date_value).strip()
        if len(text) >= 4:
            return int(text[-4:])
    except Exception:
        pass
    return 0


def _decade_for_year(year: int) -> str:
    if year <= 0:
        return "Unknown"
    return f"{(year // 10) * 10}s"


def _token_spans(text: str) -> List[Tuple[int, int]]:
    spans = []
    for m in re.finditer(r"\S+", text):
        spans.append((m.start(), m.end()))
    return spans


def _chunk_from_offsets(
    text: str,
    diary_no: str,
    judgment_date: str,
    year: int,
    strategy: str,
    index: int,
    start_pos: int,
    end_pos: int,
    section_label: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    chunk_text = text[start_pos:end_pos].strip()
    if not chunk_text:
        return None
    chunk_word_count = word_count(chunk_text)
    if chunk_word_count <= 0:
        return None
    chunk_id = make_chunk_id(diary_no, strategy, index, chunk_text)
    record = ChunkRecord(
        chunk_id=chunk_id,
        diary_no=str(diary_no),
        judgment_date=judgment_date,
        year=year,
        decade=_decade_for_year(year),
        chunking_strategy=strategy,
        chunk_index=index,
        text=chunk_text,
        word_count=chunk_word_count,
        start_position=start_pos,
        end_position=end_pos,
        section_label=section_label,
        metadata={
            **chunk_metadata_for(
                {"diary_no": diary_no, "judgment_dates": judgment_date},
                strategy,
                index,
                start_pos,
                end_pos,
                len(text),
                section_label,
                **metadata,
            ),
            "provenance": {
                "source_text_length": len(text),
                "source_diary_no": diary_no,
                "source_judgment_date": judgment_date,
                "source_year": year,
                "source_decade": _decade_for_year(year),
            },
        },
    )
    return record.to_dict()


def generic_chunk_document(record: Dict[str, Any], config: Dict[str, Any], strategy: str = "generic") -> List[Dict[str, Any]]:
    text = normalize_whitespace(record.get("clean_text", ""))
    if not text:
        return []
    diary_no = record.get("diary_no", "unknown")
    judgment_date = record.get("judgment_dates", "")
    year = _parse_year(judgment_date)
    chunk_size = int(config.get("generic_chunk_size", 500))
    overlap = int(config.get("generic_overlap", 100))
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 2)
    tokens = re.findall(r"\S+", text)
    if not tokens:
        return []
    char_positions = []
    cursor = 0
    for token in tokens:
        start = text.find(token, cursor)
        if start < 0:
            start = cursor
        end = start + len(token)
        char_positions.append((start, end))
        cursor = end
    chunks: List[Dict[str, Any]] = []
    index = 0
    i = 0
    while i < len(tokens):
        j = min(len(tokens), i + chunk_size)
        start_pos = char_positions[i][0]
        end_pos = char_positions[j - 1][1]
        chunk = _chunk_from_offsets(
            text,
            diary_no,
            judgment_date,
            year,
            strategy,
            index,
            start_pos,
            end_pos,
            "generic",
            {
                "window_size_words": chunk_size,
                "overlap_words": overlap,
                "source_strategy": "generic",
            },
        )
        if chunk is not None:
            chunks.append(chunk)
        index += 1
        if j == len(tokens):
            break
        i = max(i + (chunk_size - overlap), 1)
    return chunks


def _candidate_boundaries(text: str) -> List[int]:
    boundaries = set()
    for marker in SECTION_MARKERS:
        for match in re.finditer(rf"(?im)\b{re.escape(marker)}\b", text):
            boundaries.add(match.start())
    for match in re.finditer(r"(?m)^(?:\s*\d{1,4}[\.)]\s+)", text):
        boundaries.add(match.start())
    for match in re.finditer(r"\n\s*\n+", text):
        boundaries.add(match.start())
    return sorted(boundaries)


def _section_label_for(text: str, start: int, end: int) -> str:
    chunk = text[start:end].strip()
    if not chunk:
        return "unknown"
    for marker in SECTION_MARKERS:
        if re.search(rf"(?im)^\s*{re.escape(marker)}\b", chunk):
            return marker.lower()
    if re.search(r"(?m)^\s*\d{1,4}[\.)]\s+", chunk):
        return "numbered_paragraph"
    if "\n\n" in chunk:
        return "paragraph_block"
    return "general"


def _paragraph_spans(text: str) -> List[Tuple[int, int, int]]:
    spans = []
    for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n+|$)", text, re.DOTALL):
        paragraph = match.group(0).strip()
        if paragraph:
            start = match.start() + (len(match.group(0)) - len(match.group(0).lstrip()))
            end = start + len(paragraph)
            spans.append((start, end, word_count(paragraph)))
    return spans


def _apply_structure_overlap(
    chunks: List[Dict[str, Any]],
    text: str,
    diary_no: str,
    judgment_date: str,
    year: int,
    overlap: int,
) -> List[Dict[str, Any]]:
    if overlap <= 0 or len(chunks) < 2:
        return chunks

    paragraph_spans = _paragraph_spans(text)
    overlapped = []
    for index, chunk in enumerate(chunks):
        start = chunk["start_position"]
        if index > 0:
            overlap_words = 0
            overlap_start = start
            for paragraph_start, paragraph_end, paragraph_words in reversed(paragraph_spans):
                if paragraph_end > start:
                    continue
                if overlap_words + paragraph_words > overlap:
                    break
                overlap_words += paragraph_words
                overlap_start = paragraph_start
            if overlap_words:
                start = overlap_start

        if start == chunk["start_position"]:
            overlapped.append(chunk)
            continue

        metadata = dict(chunk.get("metadata", {}))
        metadata["overlap_words"] = overlap
        for generated_key in {
            "source_text_length",
            "source_diary_no",
            "judgment_date",
            "section_label",
            "chunk_index",
            "start_position",
            "end_position",
            "strategy",
            "provenance",
        }:
            metadata.pop(generated_key, None)
        rebuilt = _chunk_from_offsets(
            text,
            diary_no,
            judgment_date,
            year,
            "structure_aware",
            chunk["chunk_index"],
            start,
            chunk["end_position"],
            chunk["section_label"],
            metadata,
        )
        if rebuilt is not None:
            overlapped.append(rebuilt)
    return overlapped


def structure_aware_chunk_document(record: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    text = normalize_whitespace(record.get("clean_text", ""))
    if not text:
        return []

    diary_no = record.get("diary_no", "unknown")
    judgment_date = record.get("judgment_dates", "")
    year = _parse_year(judgment_date)
    target = int(config.get("structure_aware_target_chunk_size", 600))
    overlap = int(config.get("structure_aware_overlap", 0))
    min_words = int(config.get("structure_aware_min_chunk_words", 120))
    max_words = int(config.get("structure_aware_max_chunk_words", 1200))

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not paragraphs:
        return generic_chunk_document(record, config, strategy="structure_aware_fallback")

    chunks: List[Dict[str, Any]] = []
    current_start = None
    current_end = None
    current_label = "general"
    current_words = 0

    def flush_current():
        nonlocal current_start, current_end, current_label, current_words
        if current_start is None or current_end is None:
            return
        body = text[current_start:current_end].strip()
        if not body:
            current_start = None
            current_end = None
            current_label = "general"
            current_words = 0
            return
        chunk = _chunk_from_offsets(
            text,
            diary_no,
            judgment_date,
            year,
            "structure_aware",
            len(chunks),
            current_start,
            current_end,
            current_label,
            {
                "window_size_words": target,
                "overlap_words": overlap,
                "source_strategy": "structure_aware",
                "boundary_type": "section_or_paragraph",
            },
        )
        if chunk is not None:
            chunks.append(chunk)
        current_start = None
        current_end = None
        current_label = "general"
        current_words = 0

    for para in paragraphs:
        start = text.find(para, current_start if current_start is not None else 0)
        if start < 0:
            start = len(text) if current_end is None else current_end
        end = start + len(para)
        label = "general"
        for marker in SECTION_MARKERS:
            if re.search(rf"(?im)^\s*{re.escape(marker)}\b", para):
                label = marker.lower()
                break
        if label == "general" and re.search(r"(?m)^\s*\d{1,4}[\.)]\s+", para):
            label = "numbered_paragraph"

        block_words = word_count(para)
        if current_start is None:
            current_start = start
            current_end = end
            current_label = label
            current_words = block_words
            continue

        same_anchor = (current_label != "general" and label == current_label)
        new_structural_block = (current_label != "general" and label != "general" and label != current_label)
        boundary_hit = (label != "general") or (current_words + block_words > target)

        if boundary_hit and (not same_anchor or new_structural_block):
            flush_current()
            current_start = start
            current_end = end
            current_label = label
            current_words = block_words
            continue

        if current_words + block_words > target:
            flush_current()
            current_start = start
            current_end = end
            current_label = label
            current_words = block_words
            continue

        current_end = end
        current_words += block_words
        if label != "general":
            current_label = label

    flush_current()

    if not chunks:
        return generic_chunk_document(record, config, strategy="structure_aware_fallback")

    normalized: List[Dict[str, Any]] = []
    for chunk in chunks:
        if chunk["word_count"] > max_words:
            subchunks = generic_chunk_document({**record, "clean_text": chunk["text"]}, config, strategy="structure_aware")
            normalized.extend(subchunks)
        elif chunk["word_count"] < min_words and len(chunks) > 1:
            continue
        else:
            normalized.append(chunk)
    if normalized:
        return _apply_structure_overlap(normalized, text, diary_no, judgment_date, year, overlap)
    return chunks
