import json
from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG_PATH = Path("config/rag_config.json")


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load the project configuration file."""
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
