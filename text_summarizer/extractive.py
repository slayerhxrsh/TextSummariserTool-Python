"""TextRank-style extractive summarization."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import ExtractiveConfig
from .preprocessing import TextPreprocessor
from .utils import sentence_count_from_ratio


class TextRankSummarizer:
    """Rank sentences using TF-IDF similarity and a PageRank-like update."""

    def __init__(self, preprocessor: TextPreprocessor, config: ExtractiveConfig) -> None:
        self.preprocessor = preprocessor
        self.config = config

    @staticmethod
    def _pagerank(similarity_matrix: np.ndarray, damping: float = 0.85, iterations: int = 50) -> np.ndarray:
        """Compute sentence importance scores."""
        if similarity_matrix.size == 0:
            return np.array([])

        matrix = similarity_matrix.copy()
        np.fill_diagonal(matrix, 0.0)
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        normalized = matrix / row_sums

        sentence_count = normalized.shape[0]
        scores = np.ones(sentence_count) / sentence_count

        for _ in range(iterations):
            scores = (1 - damping) / sentence_count + damping * normalized.T.dot(scores)

        return scores

    def summarize(self, text: str) -> dict:
        """Generate an extractive summary and sentence scores."""
        cleaned_text = self.preprocessor.clean_text(text)
        sentences = self.preprocessor.sentence_tokenize(cleaned_text)
        if not sentences:
            return {"summary": "", "sentences": [], "scores": []}
        if len(sentences) == 1:
            return {"summary": sentences[0], "sentences": [sentences[0]], "scores": [1.0]}

        processed_sentences = [self.preprocessor.preprocess_sentence(sentence) for sentence in sentences]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(processed_sentences)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        scores = self._pagerank(similarity_matrix)

        summary_size = sentence_count_from_ratio(
            total_sentences=len(sentences),
            ratio=self.config.summary_ratio,
            minimum=self.config.min_sentences,
            maximum=self.config.max_sentences,
        )

        ranked_indexes = np.argsort(scores)[::-1][:summary_size]
        selected_indexes = sorted(ranked_indexes.tolist())
        summary_sentences = [sentences[index] for index in selected_indexes]

        return {
            "summary": " ".join(summary_sentences),
            "sentences": summary_sentences,
            "scores": scores.tolist(),
        }
