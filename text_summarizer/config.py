"""Configuration values for the summarization project."""

from dataclasses import dataclass, field


@dataclass
class ExtractiveConfig:
    """Settings for the TextRank-based extractive summarizer."""

    summary_ratio: float = 0.25
    min_sentences: int = 2
    max_sentences: int = 8
    use_lemmatization: bool = False


@dataclass
class AbstractiveConfig:
    """Settings for the transformer-based abstractive summarizer."""

    model_name: str = "facebook/bart-large-cnn"
    enabled: bool = True
    beam_width: int = 4
    length_penalty: float = 2.0
    min_length: int = 30
    max_length: int = 120
    no_repeat_ngram_size: int = 3
    max_input_tokens: int = 900
    max_chunk_overlap_sentences: int = 1
    fast_mode: bool = False
    short_text_word_threshold: int = 220
    fast_beam_width: int = 2
    fast_min_length: int = 20
    fast_max_length: int = 70


@dataclass
class PipelineConfig:
    """Top-level application settings."""

    extractive: ExtractiveConfig = field(default_factory=ExtractiveConfig)
    abstractive: AbstractiveConfig = field(default_factory=AbstractiveConfig)
