"""Input loading utilities for raw text and files."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from .utils import normalize_whitespace


SUPPORTED_EXTENSIONS = {".txt", ".pdf"}


def load_text_from_path(file_path: str) -> str:
    """Read plain text from a .txt or .pdf file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}. Supported types: {sorted(SUPPORTED_EXTENSIONS)}")

    if suffix == ".txt":
        return normalize_whitespace(path.read_text(encoding="utf-8"))

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return normalize_whitespace("\n".join(pages))


def load_text(raw_text: str | None = None, file_path: str | None = None) -> str:
    """Load text from either a raw string or a file path."""
    if raw_text and raw_text.strip():
        return normalize_whitespace(raw_text)
    if file_path:
        return load_text_from_path(file_path)
    raise ValueError("Please provide either raw_text or file_path.")
