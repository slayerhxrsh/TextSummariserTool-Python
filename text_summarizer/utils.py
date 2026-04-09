"""Shared utility helpers."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, List


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace while preserving paragraph boundaries."""
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_divide(numerator: float, denominator: float) -> float:
    """Return 0.0 when division is not possible."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compression_ratio(original_length: int, summary_length: int) -> float:
    """Compute how much shorter the summary is compared to the source text."""
    return safe_divide(summary_length, original_length)


def sentence_count_from_ratio(
    total_sentences: int,
    ratio: float,
    minimum: int,
    maximum: int,
) -> int:
    """Convert a ratio into a bounded sentence count."""
    if total_sentences <= 0:
        return 0
    count = max(minimum, math.ceil(total_sentences * ratio))
    return min(maximum, total_sentences, count)


def ngrams(tokens: List[str], size: int) -> List[tuple[str, ...]]:
    """Generate contiguous token n-grams."""
    if size <= 0 or len(tokens) < size:
        return []
    return [tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)]


def counter_overlap(first: Iterable, second: Iterable) -> int:
    """Count multiset overlap between two token collections."""
    first_counter = Counter(first)
    second_counter = Counter(second)
    return sum((first_counter & second_counter).values())
