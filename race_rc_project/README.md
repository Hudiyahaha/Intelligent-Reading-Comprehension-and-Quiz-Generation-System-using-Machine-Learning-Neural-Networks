# Intelligent Reading Comprehension and Quiz Generation (RACE)

Project layout follows the course specification: `data/`, `models/`, `src/`, `ui/`, `notebooks/`, `tests/`, `report/`.

## Setup

```bash
cd race_rc_project
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Place RACE CSV splits in `data/raw/` as `train.csv`, `test.csv`, and `dev.csv` (or `val.csv`).

If your **train and test files are duplicates** (or you want one shuffled split), use **`--combined-split`**: all CSVs present are merged, **deduplicated by `id`**, shuffled, then split so roughly **`train_fraction`** of unique rows form the train+validation pool and the rest are **test**. From that pool, **`val_fraction_of_train_pool`** is taken as validation (default **0.8 / 0.1** → about **72% train, 8% validation, 20% test**).

## Preprocessing

Builds cleaned tables, verification (long) labels, one-hot–style sparse features (binary `CountVectorizer`), optional TF–IDF, handcrafted lexical features, and cosine-similarity scalars. Fitted vectorizers are stored under `data/processed/artifacts/` (fit on **train** only).

```bash
python -m src.preprocessing --raw-dir data/raw --processed-dir data/processed
```

Duplicate-safe split example:

```bash
python -m src.preprocessing --raw-dir data/raw --processed-dir data/processed --combined-split --train-fraction 0.8 --val-fraction-of-train-pool 0.1
```

Options: `--max-ohe-features 50000`, `--max-tfidf-features 50000`, `--no-tfidf`, `--sample-train N` (debug).

## Model A (traditional only)

**No neural models.** Pipeline:

1. Template-based candidate questions from the passage (top sentences by overlap with the gold answer).
2. **Supervised:** `LogisticRegression` ranker (trained on candidate features; proxy label = candidate closest to reference question).
3. **Unsupervised:** `KMeans` on standardized candidate features; “good” cluster chosen on train; score = proximity to that centroid.
4. **Ensemble:** per-MCQ min–max normalization, then  
   `w * supervised + (1 - w) * unsupervised` (default `w = 0.5`).

**Evaluation (reporting):** mean **BLEU**, **ROUGE-1/2/L (F1)**, **METEOR** between generated question and reference question (no accuracy/precision for this stage).

```bash
python -m src.model_a_train --processed-dir data/processed --output-dir models/model_a/traditional
```

Useful options:

- `--generation-top-sentences 3`
- `--generation-max-train-mcq 20000` (limit train MCQs for speed)
- `--generation-max-val-mcq 5000` / `--generation-max-test-mcq 5000` (optional debug limits)
- `--ensemble-weight-supervised 0.5`
- `--max-eval-mcq 5000` (only limits rows in exported prediction CSVs; metrics use full val/test)

Outputs under `models/model_a/traditional/`:

- `generation_supervised.joblib`, `generation_kmeans.joblib`, `generation_unsupervised_scaler.joblib`
- `model_a_meta.json`, `generation_feature_columns.json`
- `metrics_summary.json`, `generation_*_predictions.csv`

Inference: `from src.inference import ModelAInference` then `ModelAInference().generate_question(article, answer_text)`.

## Model B (traditional only)

Model B trains two classical subsystems:

- **Distractor generation**: supervised logistic ranker + KMeans scorer + ensemble, selecting top-3 distractor tokens.
- **Hint generation**: supervised sentence ranker + KMeans scorer + ensemble, returning top-3 graduated hints.

```bash
python -m src.model_b_train --processed-dir data/processed --output-dir models/model_b/traditional
```

Useful options:

- `--max-train-mcq 20000`
- `--max-val-mcq 5000`
- `--max-test-mcq 5000`
- `--top-distractors 3`
- `--top-hints 3`

Outputs under `models/model_b/traditional/`:

- `distractor_*.joblib`, `hint_*.joblib`, `model_b_meta.json`
- `metrics_summary.json`
- `distractor_*_predictions.csv`, `hint_*_predictions.csv`

Inference: `from src.inference import ModelBInference` then `ModelBInference().generate(article, question, answer_text)`.

## BERT comparison baseline

`src/model_bert_train.py` is a reranker baseline that reuses the **same**
candidate generation that Model A and Model B build (template questions,
article-token distractors, article-sentence hints) and replaces the classical
LogisticRegression + KMeans ensemble scorer with mean-pooled BERT embeddings
(default: `bert-base-uncased`). For every candidate we compute the cosine
similarity to a task-specific target (e.g. *article sentence + answer* for
Q-gen) and pick the top-1 / top-3.

This keeps the comparison apples-to-apples:

- Same val / test MCQs as Model A and Model B (default `--max-val-mcq 2011
  --max-test-mcq 5027`, matching their `metrics_summary.json`).
- Same candidate pool (so improvements come purely from the BERT scorer, not
  from a new candidate generator).
- Same BLEU / ROUGE / METEOR metrics, written to a `metrics_summary.json`
  with the same structure used by Model A / Model B.

```bash
pip install -r requirements.txt   # installs torch (CPU) + transformers
python -m src.model_bert_train \
    --processed-dir data/processed \
    --output-dir models/model_bert \
    --model-name bert-base-uncased \
    --max-val-mcq 2011 --max-test-mcq 5027
```

Useful options:

- `--model-name distilbert-base-uncased` — roughly 2× faster on CPU.
- `--distractor-candidate-cap 30` — top-N most frequent tokens per MCQ to
  rerank (keeps full val/test tractable on CPU).
- `--batch-size 32`, `--max-length 128`, `--device cuda` (auto-detected),
  `--fp16` (GPU only), `--cache-dir <path>` (HF cache).
- Per-task flags: `--skip-question-generation`, `--skip-distractors`,
  `--skip-hints`.

Outputs under `models/model_bert/`:

- `metrics_summary.json`, `model_bert_meta.json`
- `generation_*_predictions.csv` (Model A comparison)
- `distractor_*_predictions.csv`, `hint_*_predictions.csv` (Model B comparison)

A combined `models/comparison_summary.json` is also written, with one block
per task (`question_generation`, `distractor_generation`, `hint_generation`),
each containing `validation` and `test` for both the classical model and BERT.
`notebooks/experiments.ipynb` renders this as side-by-side tables.

## Next steps

- `ui/app.py` — Streamlit UI integrating Model A + Model B
