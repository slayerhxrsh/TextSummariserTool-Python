"""ROUGE evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .utils import counter_overlap, ngrams, safe_divide


TOKEN_PATTERN = r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?"


@dataclass
class RougeScore:
    precision: float
    recall: float
    f1: float


class RougeEvaluator:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L metrics."""

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        import re

        return [token.lower() for token in re.findall(TOKEN_PATTERN, text)]

    @staticmethod
    def _f1(precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _rouge_n(self, candidate: str, reference: str, n: int) -> RougeScore:
        candidate_tokens = self._tokenize(candidate)
        reference_tokens = self._tokenize(reference)
        candidate_ngrams = ngrams(candidate_tokens, n)
        reference_ngrams = ngrams(reference_tokens, n)
        overlap = counter_overlap(candidate_ngrams, reference_ngrams)
        precision = safe_divide(overlap, len(candidate_ngrams))
        recall = safe_divide(overlap, len(reference_ngrams))
        return RougeScore(precision=precision, recall=recall, f1=self._f1(precision, recall))

    @staticmethod
    def _lcs_length(first: List[str], second: List[str]) -> int:
        rows = len(first) + 1
        cols = len(second) + 1
        table = [[0] * cols for _ in range(rows)]

        for row in range(1, rows):
            for col in range(1, cols):
                if first[row - 1] == second[col - 1]:
                    table[row][col] = table[row - 1][col - 1] + 1
                else:
                    table[row][col] = max(table[row - 1][col], table[row][col - 1])

        return table[-1][-1]

    def _rouge_l(self, candidate: str, reference: str) -> RougeScore:
        candidate_tokens = self._tokenize(candidate)
        reference_tokens = self._tokenize(reference)
        lcs = self._lcs_length(candidate_tokens, reference_tokens)
        precision = safe_divide(lcs, len(candidate_tokens))
        recall = safe_divide(lcs, len(reference_tokens))
        return RougeScore(precision=precision, recall=recall, f1=self._f1(precision, recall))

    def evaluate(self, candidate: str, reference: str | None) -> Dict[str, Dict[str, float] | None]:
        """Evaluate a summary against an optional reference summary."""
        if not reference or not reference.strip():
            return {"rouge-1": None, "rouge-2": None, "rouge-l": None}

        rouge_1 = self._rouge_n(candidate, reference, 1)
        rouge_2 = self._rouge_n(candidate, reference, 2)
        rouge_l = self._rouge_l(candidate, reference)

        return {
            "rouge-1": rouge_1.__dict__,
            "rouge-2": rouge_2.__dict__,
            "rouge-l": rouge_l.__dict__,
        }
