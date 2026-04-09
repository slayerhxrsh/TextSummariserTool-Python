"""Abstractive summarization using a pretrained BART model."""

from __future__ import annotations

from dataclasses import asdict
from typing import List

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .config import AbstractiveConfig
from .preprocessing import TextPreprocessor


class BartSummarizer:
    """Generate abstractive summaries with chunking for long documents."""

    _shared_tokenizers = {}
    _shared_models = {}

    def __init__(self, preprocessor: TextPreprocessor, config: AbstractiveConfig) -> None:
        self.preprocessor = preprocessor
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def tokenizer(self):
        if self.config.model_name not in self._shared_tokenizers:
            self._shared_tokenizers[self.config.model_name] = AutoTokenizer.from_pretrained(self.config.model_name)
        return self._shared_tokenizers[self.config.model_name]

    @property
    def model(self):
        cache_key = (self.config.model_name, self.device)
        if cache_key not in self._shared_models:
            model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name)
            model.to(self.device)
            model.eval()
            self._shared_models[cache_key] = model
        return self._shared_models[cache_key]

    def _chunk_sentences(self, sentences: List[str]) -> List[str]:
        """Split long inputs into chunks that fit the model context window."""
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_token_count = 0

        for sentence in sentences:
            sentence_token_count = len(self.tokenizer.encode(sentence, add_special_tokens=False))
            if current_chunk and current_token_count + sentence_token_count > self.config.max_input_tokens:
                chunks.append(" ".join(current_chunk))
                overlap = current_chunk[-self.config.max_chunk_overlap_sentences :]
                current_chunk = overlap[:] if overlap else []
                current_token_count = (
                    len(self.tokenizer.encode(" ".join(current_chunk), add_special_tokens=False))
                    if current_chunk
                    else 0
                )

            current_chunk.append(sentence)
            current_token_count += sentence_token_count

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _generate(self, text: str, min_length: int | None = None, max_length: int | None = None) -> str:
        """Run the model with the requested generation settings."""
        word_count = len(text.split())
        use_fast_path = self.config.fast_mode or word_count <= self.config.short_text_word_threshold
        beam_width = self.config.fast_beam_width if use_fast_path else self.config.beam_width
        minimum_length = min_length or (
            self.config.fast_min_length if use_fast_path else self.config.min_length
        )
        maximum_length = max_length or (
            self.config.fast_max_length if use_fast_path else self.config.max_length
        )

        encoded = self.tokenizer(
            text,
            max_length=1024,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **encoded,
                num_beams=beam_width,
                length_penalty=self.config.length_penalty,
                min_length=minimum_length,
                max_length=maximum_length,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                early_stopping=True,
            )

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def summarize(self, text: str, min_length: int | None = None, max_length: int | None = None) -> dict:
        """Generate an abstractive summary and keep chunk-level outputs."""
        cleaned_text = self.preprocessor.clean_text(text)
        sentences = self.preprocessor.sentence_tokenize(cleaned_text)
        if not sentences:
            return {"summary": "", "chunks": [], "config": asdict(self.config)}

        chunks = self._chunk_sentences(sentences)
        partial_summaries = [self._generate(chunk, min_length=min_length, max_length=max_length) for chunk in chunks]

        if len(partial_summaries) == 1:
            final_summary = partial_summaries[0]
        else:
            combined = " ".join(partial_summaries)
            final_summary = self._generate(combined, min_length=min_length, max_length=max_length)

        return {
            "summary": final_summary,
            "chunks": partial_summaries,
            "config": asdict(self.config),
            "used_fast_path": self.config.fast_mode or len(cleaned_text.split()) <= self.config.short_text_word_threshold,
        }
