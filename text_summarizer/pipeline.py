"""End-to-end summarization pipeline."""

from __future__ import annotations

from dataclasses import asdict

from .abstractive import BartSummarizer
from .config import PipelineConfig
from .evaluation import RougeEvaluator
from .extractive import TextRankSummarizer
from .input_handlers import load_text
from .preprocessing import TextPreprocessor
from .utils import compression_ratio


class SummarizationPipeline:
    """Coordinate loading, preprocessing, summarization, and evaluation."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.preprocessor = TextPreprocessor(
            use_lemmatization=self.config.extractive.use_lemmatization,
            use_stemming=False,
        )
        self.extractive_summarizer = TextRankSummarizer(self.preprocessor, self.config.extractive)
        self.abstractive_summarizer = BartSummarizer(self.preprocessor, self.config.abstractive)
        self.rouge_evaluator = RougeEvaluator()

    @staticmethod
    def _build_stats(original_text: str, summary_text: str) -> dict:
        original_words = len(original_text.split())
        summary_words = len(summary_text.split())
        return {
            "original_word_count": original_words,
            "summary_word_count": summary_words,
            "compression_ratio": compression_ratio(original_words, summary_words),
        }

    def summarize(
        self,
        raw_text: str | None = None,
        file_path: str | None = None,
        reference_summary: str | None = None,
        abstractive_min_length: int | None = None,
        abstractive_max_length: int | None = None,
    ) -> dict:
        """Run the full pipeline and return a serializable result."""
        source_text = load_text(raw_text=raw_text, file_path=file_path)
        cleaned_text = self.preprocessor.clean_text(source_text)

        extractive_result = self.extractive_summarizer.summarize(cleaned_text)
        if self.config.abstractive.enabled:
            abstractive_result = self.abstractive_summarizer.summarize(
                cleaned_text,
                min_length=abstractive_min_length,
                max_length=abstractive_max_length,
            )
            abstractive_payload = {
                **abstractive_result,
                "stats": self._build_stats(cleaned_text, abstractive_result["summary"]),
                "rouge": self.rouge_evaluator.evaluate(abstractive_result["summary"], reference_summary),
                "enabled": True,
            }
        else:
            abstractive_payload = {
                "summary": "",
                "chunks": [],
                "config": asdict(self.config.abstractive),
                "stats": self._build_stats(cleaned_text, ""),
                "rouge": {"rouge-1": None, "rouge-2": None, "rouge-l": None},
                "enabled": False,
                "used_fast_path": False,
            }

        return {
            "input": {
                "text": cleaned_text,
                "sentence_count": len(self.preprocessor.sentence_tokenize(cleaned_text)),
                "word_count": len(cleaned_text.split()),
            },
            "extractive": {
                **extractive_result,
                "stats": self._build_stats(cleaned_text, extractive_result["summary"]),
                "rouge": self.rouge_evaluator.evaluate(extractive_result["summary"], reference_summary),
            },
            "abstractive": abstractive_payload,
            "reference_summary": reference_summary,
            "config": asdict(self.config),
        }
