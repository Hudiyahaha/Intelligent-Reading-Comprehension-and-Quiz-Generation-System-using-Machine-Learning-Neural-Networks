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

## Preprocessing

Builds cleaned tables, verification (long) labels, one-hot–style sparse features (binary `CountVectorizer`), optional TF–IDF, handcrafted lexical features, and cosine-similarity scalars. Fitted vectorizers are stored under `data/processed/artifacts/` (fit on **train** only).

```bash
python -m src.preprocessing --raw-dir data/raw --processed-dir data/processed
```

Options: `--max-ohe-features 50000`, `--max-tfidf-features 50000`, `--no-tfidf`, `--sample-train N` (debug).

## Next steps

- `src/model_a_train.py` — Model A (answer verification) training
- `src/model_b_train.py` — Model B training
- `src/inference.py` — unified inference API
- `ui/app.py` — Streamlit UI

## Model A training (Colab / VSCode notebooks)

Model A now implements:

- **Verification**: given `(article, question, option)` predict `label in {0,1}`
- **Template-based question generation + ML ranking**: generate candidate questions from top answer-relevant sentences and rank them with a trained logistic ranker

It trains:

- Supervised baselines: Logistic Regression, Linear SVM, BernoulliNB, RandomForest
- Ensemble: soft voting over probabilistic models
- Unsupervised: KMeans purity/silhouette on handcrafted features
- Semi-supervised: Label Spreading

Run:

```bash
python -m src.model_a_train --processed-dir data/processed --output-dir models/model_a/traditional
```

Useful options:

- `--sparse-feature-kind ohe|tfidf`
- `--max-train-rows 20000` (faster debug)
- `--skip-unsupervised`
- `--skip-semi-supervised`
- `--skip-generation`
- `--generation-top-sentences 3`
- `--generation-max-train-mcq 20000`

Main output:

- `models/model_a/traditional/*.joblib`
- `models/model_a/traditional/metrics_summary.json`
- `models/model_a/traditional/generation_ranker.joblib`
- `models/model_a/traditional/generation_feature_columns.json`
- `models/model_a/traditional/generation_preview.csv`
