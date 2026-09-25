import json
import importlib.util
import time
from pathlib import Path

from src.rag.chunk_models import ChunkRecord
from src.rag.chunking import generic_chunk_document, structure_aware_chunk_document
from src.rag.config_loader import load_config
from src.rag.text_utils import normalize_whitespace


def _load_validation_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "26_validate_chunks.py"
    spec = importlib.util.spec_from_file_location("validate_chunks_26", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validate_chunks = _load_validation_module()


CONFIG = load_config(Path("config/rag_config.json"))


def test_chunk_metadata_fields():
    record = {
        "diary_no": "123-2020",
        "judgment_dates": "15-05-2020",
        "clean_text": "Alpha beta gamma delta. Epsilon zeta eta theta. Iota kappa lambda mu."
    }
    chunks = generic_chunk_document(record, CONFIG, "generic")
    assert chunks
    chunk = chunks[0]
    assert isinstance(chunk, dict)
    assert set(
        [
            "chunk_id",
            "diary_no",
            "judgment_date",
            "year",
            "decade",
            "chunking_strategy",
            "chunk_index",
            "text",
            "word_count",
            "start_position",
            "end_position",
            "section_label",
            "metadata",
        ]
    ).issubset(chunk.keys())
    assert chunk["chunking_strategy"] == "generic"
    assert chunk["word_count"] > 0
    assert chunk["metadata"]["source_text_length"] == len(record["clean_text"])


def test_chunk_ids_are_unique():
    record = {
        "diary_no": "123-2020",
        "judgment_dates": "15-05-2020",
        "clean_text": " ".join([f"word{i}" for i in range(1000)])
    }
    chunks = generic_chunk_document(record, CONFIG, "generic")
    chunk_ids = [c["chunk_id"] for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_source_offsets_are_valid():
    record = {
        "diary_no": "123-2020",
        "judgment_dates": "15-05-2020",
        "clean_text": "First second third fourth fifth. Another sentence in the same record."
    }
    chunks = generic_chunk_document(record, CONFIG, "generic")
    for chunk in chunks:
        assert chunk["start_position"] >= 0
        assert chunk["end_position"] >= chunk["start_position"]
        assert chunk["end_position"] <= len(record["clean_text"])
        assert record["clean_text"][chunk["start_position"]:chunk["end_position"]].strip() == chunk["text"]


def test_no_empty_chunks():
    record = {
        "diary_no": "123-2020",
        "judgment_dates": "15-05-2020",
        "clean_text": " ".join([f"w{i}" for i in range(700)])
    }
    chunks = generic_chunk_document(record, CONFIG, "generic")
    assert all(c["text"].strip() for c in chunks)
    assert all(c["word_count"] > 0 for c in chunks)


def test_generic_chunk_size_and_overlap_behavior():
    record = {
        "diary_no": "123-2020",
        "judgment_dates": "15-05-2020",
        "clean_text": " ".join([f"w{i}" for i in range(1500)])
    }
    chunks = generic_chunk_document(record, CONFIG, "generic")
    assert len(chunks) > 1
    assert all(c["metadata"]["window_size_words"] == CONFIG["generic_chunk_size"] for c in chunks)
    assert all(c["metadata"]["overlap_words"] == CONFIG["generic_overlap"] for c in chunks)
    assert all(
        chunk["word_count"] <= CONFIG["generic_chunk_size"]
        for chunk in chunks[:-1]
    )


def test_generic_baseline_respects_configured_window_for_normal_input():
    record = {
        "diary_no": "normal-2020",
        "judgment_dates": "15-05-2020",
        "clean_text": " ".join([f"word{i}" for i in range(2400)]),
    }
    chunks = generic_chunk_document(record, CONFIG, "generic")
    assert len(chunks) == 6
    assert all(chunk["word_count"] == CONFIG["generic_chunk_size"] for chunk in chunks[:-1])
    assert chunks[-1]["word_count"] <= CONFIG["generic_chunk_size"]


def test_structure_aware_uses_markers_when_present():
    record = {
        "diary_no": "123-2020",
        "judgment_dates": "15-05-2020",
        "clean_text": "\n\nHEADNOTE\nA short headnote.\n\nFACTS\nThis is a facts section.\n\nISSUES\nIssue one.\n\nHELD\nThe court held in favor.\n\n1. The first numbered paragraph is clear.\n2. The second numbered paragraph follows.\n"
    }
    chunks = structure_aware_chunk_document(record, CONFIG)
    assert chunks
    assert any("headnote" in c["section_label"].lower() or "facts" in c["section_label"].lower() for c in chunks)
    assert any("numbered" in c["section_label"].lower() for c in chunks) or any("issue" in c["section_label"].lower() for c in chunks)


def test_structure_aware_provenance_and_configured_paragraph_overlap():
    record = {
        "diary_no": "overlap-2020",
        "judgment_dates": "15-05-2020",
        "clean_text": "\n\nFACTS\n" + " ".join([f"fact{i}" for i in range(30)])
        + "\n\nHELD\n" + " ".join([f"held{i}" for i in range(30)])
        + "\n\nORDER\n" + " ".join([f"order{i}" for i in range(30)]),
    }
    config = {
        **CONFIG,
        "structure_aware_target_chunk_size": 35,
        "structure_aware_overlap": 35,
        "structure_aware_min_chunk_words": 0,
    }
    chunks = structure_aware_chunk_document(record, config)
    assert len(chunks) > 1
    source_text = normalize_whitespace(record["clean_text"])
    for chunk in chunks:
        assert source_text[chunk["start_position"]:chunk["end_position"]].strip() == chunk["text"]
    assert chunks[1]["start_position"] < chunks[0]["end_position"]


def test_structure_aware_default_has_no_artificial_overlap():
    record = {
        "diary_no": "no-overlap-2020",
        "judgment_dates": "15-05-2020",
        "clean_text": "\n\nFACTS\n" + " ".join([f"fact{i}" for i in range(30)])
        + "\n\nHELD\n" + " ".join([f"held{i}" for i in range(30)]),
    }
    config = {
        **CONFIG,
        "structure_aware_target_chunk_size": 35,
        "structure_aware_overlap": 0,
        "structure_aware_min_chunk_words": 0,
    }
    chunks = structure_aware_chunk_document(record, config)
    assert all(c["metadata"]["overlap_words"] == 0 for c in chunks)
    assert chunks[1]["start_position"] >= chunks[0]["end_position"]


def test_structure_aware_falls_back_for_weak_structure():
    record = {
        "diary_no": "123-2020",
        "judgment_dates": "15-05-2020",
        "clean_text": " ".join([f"word{i}" for i in range(800)])
    }
    chunks = structure_aware_chunk_document(record, CONFIG)
    assert chunks
    assert all(c["text"].strip() for c in chunks)


def test_historical_documents_are_supported():
    for year in [1950, 1970, 1990, 2010, 2020]:
        record = {
            "diary_no": f"{year}-test",
            "judgment_dates": f"01-01-{year}",
            "clean_text": "\n\nSUPREME COURT OF INDIA\nPage 1 of 3\nPETITIONER\nTest\nRESPONDENT\nAnother\n\nJUDGMENT\n1. This is a historical record.\n2. It should still be chunked.\n" if year < 2000 else "\n\nJUDGMENT\n1. This is a modern record.\n2. It should still be chunked.\n"
        }
        chunks = structure_aware_chunk_document(record, CONFIG)
        assert chunks
        assert all(c["year"] == year for c in chunks)


def test_reproducibility():
    record = {
        "diary_no": "repeat-1",
        "judgment_dates": "20-10-2015",
        "clean_text": " ".join([f"token{i}" for i in range(1200)])
    }
    first = generic_chunk_document(record, CONFIG, "generic")
    second = generic_chunk_document(record, CONFIG, "generic")
    assert [c["chunk_id"] for c in first] == [c["chunk_id"] for c in second]
    assert [c["text"] for c in first] == [c["text"] for c in second]


def test_summarize_duplicate_counts_are_fast_and_correct():
    chunks = []
    for i in range(60000):
        chunk_id = f"dup-{i % 5000}"
        text = f"sentence-{i % 2000}"
        chunks.append(
            {
                "diary_no": "case-1",
                "word_count": 100,
                "text": text,
                "chunk_id": chunk_id,
                "year": 2020,
                "decade": 2020,
                "start_position": 0,
                "end_position": 10,
                "metadata": {"provenance": "source"},
            }
        )

    start = time.perf_counter()
    stats = validate_chunks.summarize(chunks, "generic")
    elapsed = time.perf_counter() - start

    assert stats["duplicate_chunk_ids"] == 5000
    assert stats["duplicate_chunk_text"] == 2000
    assert elapsed < 2.0
