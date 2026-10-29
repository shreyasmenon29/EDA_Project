from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_METADATA_CACHE: dict[str, dict[str, Any]] | None = None


def _get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _build_metadata_dict(jsonl_path: Path) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}

    if not jsonl_path.exists():
        return metadata

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            diary_no = str(doc.get("diary_no") or "").strip()
            if not diary_no:
                continue

            pet = str(doc.get("pet") or "").strip()
            res = str(doc.get("res") or "").strip()
            if pet.lower() == "nan":
                pet = ""
            if res.lower() == "nan":
                res = ""

            if pet and res:
                case_name = f"{pet} v. {res}"
            elif pet:
                case_name = pet
            elif res:
                case_name = f"v. {res}"
            else:
                case_no = str(doc.get("case_no") or "").strip()
                if case_no and case_no.lower() != "nan" and case_no != "- 0":
                    case_name = f"Case {case_no}"
                else:
                    case_name = "Case name unavailable"

            dt = str(doc.get("judgment_dates") or doc.get("judgment_date") or "").strip()
            year = ""
            if dt and dt.lower() != "nan":
                parts = dt.split("-")
                if len(parts) == 3 and len(parts[2]) == 4:
                    year = parts[2]
                elif len(parts) == 1 and len(parts[0]) == 4:
                    year = parts[0]
            if not year:
                year = "Year unavailable"

            cit = str(doc.get("citation_extracted") or "").strip()
            if not cit or cit.lower() == "nan":
                citation = "Citation unavailable"
            else:
                citation = " ".join(cit.split())

            link = str(doc.get("temp_link") or "").strip()
            if not link or link.lower() == "nan":
                url = None
            else:
                url = f"https://main.sci.gov.in/{link.lstrip('/')}"

            metadata[diary_no] = {
                "diary_no": diary_no,
                "case_name": case_name,
                "year": year,
                "citation": citation,
                "url": url,
                "judgment_date": dt if dt and dt.lower() != "nan" else "Date unavailable",
                "case_no": str(doc.get("case_no") or ""),
            }

    return metadata


def load_metadata_cache() -> dict[str, dict[str, Any]]:
    global _METADATA_CACHE

    if _METADATA_CACHE is not None:
        return _METADATA_CACHE

    root = _get_project_root()
    json_path = root / "data" / "processed" / "document_metadata.json"
    jsonl_path = root / "data" / "processed" / "judgments_processed.jsonl"

    if json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as f:
                _METADATA_CACHE = json.load(f)
                return _METADATA_CACHE
        except Exception:
            pass

    # Build from processed jsonl if json cache is missing or corrupt
    _METADATA_CACHE = _build_metadata_dict(jsonl_path)
    if _METADATA_CACHE:
        try:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(_METADATA_CACHE, f, ensure_ascii=False)
        except Exception:
            pass

    return _METADATA_CACHE


def get_document_metadata(diary_no: str) -> dict[str, Any]:
    """
    Look up verified document metadata for a given diary number.
    Returns case name, year, verified citation (or 'Citation unavailable'),
    clickable URL (or None if unavailable), and judgment date.
    """
    cache = load_metadata_cache()
    if diary_no and diary_no in cache:
        return cache[diary_no]

    return {
        "diary_no": diary_no,
        "case_name": "Case name unavailable",
        "year": "Year unavailable",
        "citation": "Citation unavailable",
        "url": None,
        "judgment_date": "Date unavailable",
        "case_no": "",
    }
