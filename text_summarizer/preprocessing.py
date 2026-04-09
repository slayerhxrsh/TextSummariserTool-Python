"""Text cleaning and tokenization helpers."""

from __future__ import annotations

import re
import zipfile
from typing import Iterable, List

import spacy
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .utils import normalize_whitespace


FALLBACK_STOPWORDS = set(ENGLISH_STOP_WORDS)
WORD_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


class TextPreprocessor:
    """Prepare text for extractive and evaluation workflows."""

    _shared_nlp = None

    def __init__(self, use_lemmatization: bool = False, use_stemming: bool = False) -> None:
        self.stop_words = self._load_stopwords()
        self.use_lemmatization = use_lemmatization
        self.use_stemming = use_stemming
        self.stemmer = PorterStemmer() if use_stemming else None
        self._nlp = self._load_spacy_model() if use_lemmatization else None

    @staticmethod
    def _load_stopwords() -> set[str]:
        """Use NLTK stopwords when available, otherwise fall back to sklearn's list."""
        try:
            from nltk.corpus import stopwords

            return set(stopwords.words("english"))
        except (LookupError, OSError, zipfile.BadZipFile):
            return FALLBACK_STOPWORDS

    @staticmethod
    def _load_spacy_model():
        """Load a small spaCy model when available; otherwise skip lemmatization."""
        if TextPreprocessor._shared_nlp is not None:
            return TextPreprocessor._shared_nlp

        try:
            TextPreprocessor._shared_nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        except OSError:
            TextPreprocessor._shared_nlp = None

        return TextPreprocessor._shared_nlp

    def clean_text(self, text: str) -> str:
        """Remove noisy whitespace and normalize punctuation spacing."""
        text = normalize_whitespace(text)
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        return text

    def sentence_tokenize(self, text: str) -> List[str]:
        """Split text into sentences without requiring external tokenizer data."""
        cleaned = self.clean_text(text)
        if not cleaned:
            return []
        if not re.search(r"[.!?]", cleaned):
            return [cleaned]
        return [sentence.strip() for sentence in SENTENCE_SPLIT_PATTERN.split(cleaned) if sentence.strip()]

    def word_tokenize(self, text: str) -> List[str]:
        """Split text into word-like tokens without external corpus files."""
        return WORD_PATTERN.findall(text)

    def normalize_tokens(self, tokens: Iterable[str], remove_stopwords: bool = True) -> List[str]:
        """Lowercase and optionally filter stop words and punctuation."""
        cleaned_tokens = []
        for token in tokens:
            normalized = token.lower()
            if not normalized.isalpha():
                continue
            if remove_stopwords and normalized in self.stop_words:
                continue
            if self.use_stemming and self.stemmer:
                normalized = self.stemmer.stem(normalized)
            cleaned_tokens.append(normalized)

        if self.use_lemmatization and self._nlp and cleaned_tokens:
            doc = self._nlp(" ".join(cleaned_tokens))
            return [token.lemma_.lower() for token in doc if token.lemma_.strip()]

        return cleaned_tokens

    def preprocess_sentence(self, sentence: str) -> str:
        """Create a cleaned sentence string for vectorization."""
        tokens = self.word_tokenize(sentence)
        processed_tokens = self.normalize_tokens(tokens, remove_stopwords=True)
        return " ".join(processed_tokens)
