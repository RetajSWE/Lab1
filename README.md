https://github.com/RetajSWE/Nextstep-SDAIA
# Bayan | بيان

### SDA-AIE-211 — NLP with Transformers

**Bayan** is a bilingual Arabic-English NLP service for analyzing citizen feedback using Transformer-based models.

## Features

* Arabic & English preprocessing + PII masking
* Topic classification
* Named Entity Recognition (NER)
* Extractive Question Answering
* Arabic dialect-aware processing
* Semantic search with FAISS
* Model evaluation and error analysis
* ONNX / INT8 optimization
* FastAPI deployment

## Tech Stack

**Python · Hugging Face Transformers · spaCy · CAMeL Tools · FAISS · scikit-learn · ONNX · FastAPI · Pytest**

## Project Flow

```text
Raw Feedback
     ↓
Preprocessing
     ↓
Transformer Models
     ↓
Classification / NER / QA / Search
     ↓
Evaluation & Optimization
     ↓
FastAPI
```

## Setup

```bash
python -m venv .venv
pip install -r requirements.txt
pip install -e .
```

## Run

```bash
uvicorn bayan.serving.api:app --host 0.0.0.0 --port 8000
```

## Test

```bash
pytest -q
```

### Outcome

An end-to-end bilingual NLP system covering **preprocessing, Transformer models, Arabic NLP, semantic search, evaluation, optimization, and deployment**.
