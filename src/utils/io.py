"""I/O utilities: JSON save/load, directory helpers."""

import json
import os
from datetime import datetime
from typing import Any, Optional


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist. Return path."""
    os.makedirs(path, exist_ok=True)
    return path


def save_json(data: Any, path: str, indent: int = 2) -> None:
    """Save data as JSON with UTF-8 encoding."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)


def load_json(path: str) -> Optional[Any]:
    """Load JSON file. Return None if file missing."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def timestamped_dir(root: str) -> str:
    """Create and return root/YYYY-MM-DD_HHMMSS/ directory."""
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = os.path.join(root, ts)
    return ensure_dir(run_dir)
