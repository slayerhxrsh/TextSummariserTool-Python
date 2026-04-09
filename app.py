"""Flask app and CLI for the text summarization tool."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from text_summarizer import SummarizationPipeline
from text_summarizer.config import PipelineConfig


app = Flask(__name__)
app.config["SECRET_KEY"] = "development-secret-key"
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


IS_VERCEL = env_flag("VERCEL", False)
DEFAULT_ABSTRACTIVE_ENABLED = env_flag("ENABLE_ABSTRACTIVE", not IS_VERCEL)
DEFAULT_FAST_MODE = env_flag("FAST_MODE_DEFAULT", True)
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_SUMMARIZER_MODEL", "sshleifer/distilbart-cnn-12-6")

MODEL_OPTIONS = {
    "facebook/bart-large-cnn": "BART Large CNN (best quality, slower)",
    "sshleifer/distilbart-cnn-12-6": "DistilBART CNN (faster)",
}


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in {".txt", ".pdf"}


@lru_cache(maxsize=16)
def build_pipeline(
    summary_ratio: float = 0.25,
    use_lemmatization: bool = False,
    abstractive_enabled: bool = True,
    model_name: str = "facebook/bart-large-cnn",
    fast_mode: bool = False,
) -> SummarizationPipeline:
    config = PipelineConfig()
    config.extractive.summary_ratio = summary_ratio
    config.extractive.use_lemmatization = use_lemmatization
    config.abstractive.enabled = abstractive_enabled
    config.abstractive.model_name = model_name
    config.abstractive.fast_mode = fast_mode
    return SummarizationPipeline(config=config)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    form_values = {
        "raw_text": "",
        "reference_summary": "",
        "summary_ratio": 0.25,
        "use_lemmatization": False,
        "min_length": 30,
        "max_length": 120,
        "abstractive_enabled": DEFAULT_ABSTRACTIVE_ENABLED,
        "model_name": DEFAULT_MODEL_NAME if DEFAULT_MODEL_NAME in MODEL_OPTIONS else "sshleifer/distilbart-cnn-12-6",
        "fast_mode": DEFAULT_FAST_MODE,
    }

    if request.method == "POST":
        raw_text = request.form.get("raw_text", "")
        reference_summary = request.form.get("reference_summary", "")
        summary_ratio = float(request.form.get("summary_ratio", 0.25))
        use_lemmatization = request.form.get("use_lemmatization") == "on"
        min_length = int(request.form.get("min_length", 30))
        max_length = int(request.form.get("max_length", 120))
        abstractive_enabled = request.form.get("abstractive_enabled") == "on"
        model_name = request.form.get("model_name", "sshleifer/distilbart-cnn-12-6")
        fast_mode = request.form.get("fast_mode") == "on"
        uploaded_file = request.files.get("document")

        if model_name not in MODEL_OPTIONS:
            model_name = "sshleifer/distilbart-cnn-12-6"

        form_values.update(
            {
                "raw_text": raw_text,
                "reference_summary": reference_summary,
                "summary_ratio": summary_ratio,
                "use_lemmatization": use_lemmatization,
                "min_length": min_length,
                "max_length": max_length,
                "abstractive_enabled": abstractive_enabled,
                "model_name": model_name,
                "fast_mode": fast_mode,
            }
        )

        if not raw_text.strip() and (uploaded_file is None or uploaded_file.filename == ""):
            flash("Please enter text or upload a .txt/.pdf file.")
            return redirect(url_for("index"))
        temp_path = None

        try:
            pipeline = build_pipeline(
                summary_ratio=summary_ratio,
                use_lemmatization=use_lemmatization,
                abstractive_enabled=abstractive_enabled,
                model_name=model_name,
                fast_mode=fast_mode,
            )

            if uploaded_file and uploaded_file.filename:
                if not allowed_file(uploaded_file.filename):
                    flash("Unsupported file type. Please upload a .txt or .pdf file.")
                    return redirect(url_for("index"))

                safe_name = secure_filename(uploaded_file.filename)
                suffix = Path(safe_name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    uploaded_file.save(temp_file.name)
                    temp_path = temp_file.name

            result = pipeline.summarize(
                raw_text=raw_text,
                file_path=temp_path,
                reference_summary=reference_summary,
                abstractive_min_length=min_length,
                abstractive_max_length=max_length,
            )
        except Exception as error:  # noqa: BLE001 - user-facing message
            flash(f"Summarization failed: {error}")
        finally:
            if temp_path and Path(temp_path).exists():
                Path(temp_path).unlink(missing_ok=True)

    return render_template(
        "index.html",
        result=result,
        form_values=form_values,
        model_options=MODEL_OPTIONS,
        is_vercel=IS_VERCEL,
    )


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Compare extractive and abstractive text summarization.")
    parser.add_argument("--text", type=str, help="Raw input text.")
    parser.add_argument("--file", type=str, help="Path to a .txt or .pdf document.")
    parser.add_argument("--reference", type=str, help="Optional reference summary for ROUGE evaluation.")
    parser.add_argument("--summary-ratio", type=float, default=0.25, help="Extractive summary ratio.")
    parser.add_argument("--use-lemmatization", action="store_true", help="Enable spaCy lemmatization when available.")
    parser.add_argument("--min-length", type=int, default=30, help="Minimum abstractive summary length.")
    parser.add_argument("--max-length", type=int, default=120, help="Maximum abstractive summary length.")
    parser.add_argument("--serve", action="store_true", help="Start the Flask web application.")
    args = parser.parse_args()

    if args.serve:
        app.run(debug=False)
        return

    if not args.text and not args.file:
        parser.error("Provide --text or --file, or use --serve for the web UI.")

    pipeline = build_pipeline(summary_ratio=args.summary_ratio, use_lemmatization=args.use_lemmatization)
    result = pipeline.summarize(
        raw_text=args.text,
        file_path=args.file,
        reference_summary=args.reference,
        abstractive_min_length=args.min_length,
        abstractive_max_length=args.max_length,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run_cli()
