# Text Summarization Tool

This project compares two summarization strategies on the same document:

- Extractive summarization with a TextRank-style sentence ranking pipeline
- Abstractive summarization with `facebook/bart-large-cnn`

It includes raw text input, `.txt` and `.pdf` upload support, optional spaCy lemmatization, ROUGE evaluation, a Flask web interface, and a CLI entry point.

The web UI also supports:

- Fast mode for short text
- Skipping abstractive summarization for extractive-only runs
- Choosing between `facebook/bart-large-cnn` and the faster `sshleifer/distilbart-cnn-12-6`
- A loading indicator while summaries are being generated

## Project Structure

```text
.
├── app.py
├── example_input.txt
├── requirements.txt
├── templates/
│   └── index.html
└── text_summarizer/
    ├── __init__.py
    ├── abstractive.py
    ├── config.py
    ├── evaluation.py
    ├── extractive.py
    ├── input_handlers.py
    ├── pipeline.py
    ├── preprocessing.py
    └── utils.py
```

## Setup

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

2. Install project dependencies:

```powershell
pip install -r requirements.txt
```

3. Optional: install the spaCy English model for lemmatization:

```powershell
python -m spacy download en_core_web_sm
```

## Run the Web App

```powershell
python app.py --serve
```

Open `http://127.0.0.1:5000` in your browser.

## Deploy To Vercel

This repository now includes:

- [`vercel.json`](./vercel.json) with Fluid Compute enabled
- [`.python-version`](./.python-version) pinned to Python `3.11`
- Vercel-friendly defaults in [`app.py`](./app.py)

### Recommended Vercel setup

1. Push the project to GitHub.
2. Import the repository into Vercel.
3. Keep the framework preset as `Other`.
4. Set these environment variables in Vercel if you want to control behavior:

```text
ENABLE_ABSTRACTIVE=false
FAST_MODE_DEFAULT=true
DEFAULT_SUMMARIZER_MODEL=sshleifer/distilbart-cnn-12-6
```

### Important deployment notes

- The safest Vercel deployment is extractive-only or DistilBART with fast mode.
- Large local transformer inference on Vercel can still be slow because serverless Python functions have memory, duration, and request-body limits.
- Uploaded files should stay small for serverless usage.

### Deploy using the Vercel CLI

```powershell
npm i -g vercel
vercel
```

For production:

```powershell
vercel --prod
```

## Run from the Command Line

Using raw text:

```powershell
python app.py --text "Your long article goes here."
```

Using a file:

```powershell
python app.py --file example_input.txt
```

With evaluation and custom settings:

```powershell
python app.py --file example_input.txt --reference "Short gold summary." --summary-ratio 0.3 --min-length 40 --max-length 110
```

## Example Input

The repository already includes [`example_input.txt`](./example_input.txt):

```text
Artificial intelligence is becoming a core part of modern software systems. Companies use AI for search, recommendation, document processing, and customer support. As the amount of digital text grows, summarization helps users understand large documents quickly. Extractive approaches select the most important original sentences, while abstractive approaches generate new language that compresses the main ideas. Each method has trade-offs in fluency, faithfulness, and compute cost.
```

## Example Output

Example extractive summary:

```text
As the amount of digital text grows, summarization helps users understand large documents quickly. Extractive approaches select the most important original sentences, while abstractive approaches generate new language that compresses the main ideas.
```

Example abstractive summary:

```text
Text summarization helps people understand large documents faster. Extractive methods reuse key sentences, while abstractive methods rewrite the content into a shorter explanation.
```

## Pipeline Overview

1. Input handling: accepts raw text or `.txt` / `.pdf` files.
2. Preprocessing: cleans whitespace, tokenizes text, removes stopwords for extractive ranking, and optionally lemmatizes text.
3. Extractive summarization: builds TF-IDF sentence vectors, computes cosine similarity, and ranks sentences with a PageRank-style loop.
4. Abstractive summarization: runs BART with beam search, length penalty, minimum and maximum length constraints, and `no_repeat_ngram_size=3`.
5. Evaluation: computes ROUGE-1, ROUGE-2, and ROUGE-L when a reference summary is provided.

## Performance Tips

- For the fastest response, disable abstractive summarization.
- For a balanced quality/speed trade-off in the UI, choose `DistilBART CNN (faster)`.
- Fast mode automatically uses shorter generation settings for short inputs.
- The first transformer run is still the slowest because the model must be loaded into memory.

## Notes

- The first BART run downloads model weights, so startup can take time.
- If no GPU is available, BART runs on CPU.
- ROUGE values are returned as `None` when no reference summary is supplied.
- Long documents are chunked before BART inference so the model stays within its input token limit.
