import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ChunkRecord:
    chunk_id: str
    diary_no: str
    judgment_date: str
    year: int
    decade: str
    chunking_strategy: str
    chunk_index: int
    text: str
    word_count: int
    start_position: int
    end_position: int
    section_label: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ChunkRecord":
        return cls(**payload)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "diary_no": self.diary_no,
            "judgment_date": self.judgment_date,
            "year": self.year,
            "decade": self.decade,
            "chunking_strategy": self.chunking_strategy,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "word_count": self.word_count,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "section_label": self.section_label,
            "metadata": self.metadata,
        }


def make_chunk_id(diary_no: str, strategy: str, index: int, text: str) -> str:
    seed = f"{diary_no}|{strategy}|{index}|{text[:120]}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:24]


def chunk_metadata_for(
    record: Dict[str, Any],
    strategy: str,
    chunk_index: int,
    start_position: int,
    end_position: int,
    source_text_length: int,
    section_label: str,
    **extra: Any,
) -> Dict[str, Any]:
    return {
        "source_text_length": source_text_length,
        "source_diary_no": record.get("diary_no"),
        "judgment_date": record.get("judgment_dates"),
        "section_label": section_label,
        "chunk_index": chunk_index,
        "start_position": start_position,
        "end_position": end_position,
        "strategy": strategy,
        **extra,
    }
