"""Small shared helpers used across the package."""

from pathlib import Path


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
