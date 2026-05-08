"""
RACE dataset loading, cleaning, and feature engineering.

Primary text representation: binary bag-of-words via CountVectorizer (one-hot–style
word presence). Optional TF–IDF. Vectorizers are fit on the training split only.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

OPTION_COLS = ("A", "B", "C", "D")


@dataclass
class PreprocessConfig:
    max_ohe_features: int = 50_000
    max_tfidf_features: int = 50_000
    max_aq_tfidf_features: int = 8192
    min_df: int = 2
    ngram_max: int = 1
    build_tfidf: bool = True
    random_seed: int = 42


_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def clean_text(text: str | float | None) -> str:
    """Lowercase, strip, remove punctuation (keep alphanumerics and whitespace)."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = str(text).lower().strip()
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    drop_cols = [c for c in df.columns if c.startswith("Unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    for c in ["id", "article", "question", "A", "B", "C", "D", "answer"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    df["answer"] = df["answer"].astype(str).str.upper().str.strip()
    return df


def load_race_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalize_columns(df)
    for c in ["article", "question", "A", "B", "C", "D"]:
        df[c] = df[c].fillna("").astype(str)
    return df


def load_raw_splits(raw_dir: Path) -> dict[str, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    train_path = raw_dir / "train.csv"
    test_path = raw_dir / "test.csv"
    val_path = raw_dir / "val.csv"
    if not val_path.is_file():
        val_path = raw_dir / "dev.csv"
    if not train_path.is_file():
        raise FileNotFoundError(f"Missing {train_path}")
    if not test_path.is_file():
        raise FileNotFoundError(f"Missing {test_path}")
    if not val_path.is_file():
        raise FileNotFoundError(f"Missing validation file (val.csv or dev.csv)")

    return {
        "train": load_race_csv(train_path),
        "validation": load_race_csv(val_path),
        "test": load_race_csv(test_path),
    }


def clean_mcq_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One row per original MCQ with cleaned text fields."""
    out = df.copy()
    out["article_clean"] = out["article"].map(clean_text)
    out["question_clean"] = out["question"].map(clean_text)
    for col in OPTION_COLS:
        out[f"{col}_clean"] = out[col].map(clean_text)
    return out


def mcq_to_verification_long(mcq: pd.DataFrame) -> pd.DataFrame:
    """
    Four rows per MCQ: one per option. Label 1 if option letter matches `answer`.
    """
    mcq = mcq.reset_index(drop=True)
    gold = mcq["answer"].astype(str).str.upper().str.strip()
    parts: list[pd.DataFrame] = []
    for letter in OPTION_COLS:
        parts.append(
            pd.DataFrame(
                {
                    "mcq_row_id": mcq.index.to_numpy(),
                    "id": mcq["id"].to_numpy(),
                    "article_clean": mcq["article_clean"].to_numpy(),
                    "question_clean": mcq["question_clean"].to_numpy(),
                    "option_letter": letter,
                    "option_clean": mcq[f"{letter}_clean"].to_numpy(),
                    "label": (gold == letter).astype(np.int8).to_numpy(),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _token_set(text: str) -> set[str]:
    return set(text.split()) if text else set()


def jaccard(a: str, b: str) -> float:
    sa, sb = _token_set(a), _token_set(b)
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def add_handcrafted_lexical_features(verify: pd.DataFrame) -> pd.DataFrame:
    """Dense lexical features per verification row."""
    v = verify.copy()
    ac, qc, oc = v["article_clean"], v["question_clean"], v["option_clean"]
    v["feat_article_char_len"] = ac.str.len()
    v["feat_question_char_len"] = qc.str.len()
    v["feat_option_char_len"] = oc.str.len()
    v["feat_article_word_count"] = ac.str.split().str.len().fillna(0).astype(int)
    v["feat_question_word_count"] = qc.str.split().str.len().fillna(0).astype(int)
    v["feat_option_word_count"] = oc.str.split().str.len().fillna(0).astype(int)
    v["feat_jaccard_question_article"] = [jaccard(q, a) for q, a in zip(qc, ac)]
    v["feat_jaccard_option_article"] = [jaccard(o, a) for o, a in zip(oc, ac)]
    v["feat_jaccard_option_question"] = [jaccard(o, q) for o, q in zip(oc, qc)]
    return v


def _verification_concat_texts(df: pd.DataFrame) -> list[str]:
    return (
        df["article_clean"] + " " + df["question_clean"] + " " + df["option_clean"]
    ).tolist()


def batch_cosine_pairs(a: sparse.csr_matrix, b: sparse.csr_matrix, batch: int = 2048) -> np.ndarray:
    """Row-wise cosine similarity between matching rows of a and b."""
    n = a.shape[0]
    out = np.zeros(n, dtype=np.float32)
    for start in range(0, n, batch):
        end = min(start + batch, n)
        out[start:end] = cosine_similarity(a[start:end], b[start:end]).diagonal().astype(np.float32)
    return out


def fit_transform_verification_features(
    train_verify: pd.DataFrame,
    other_verify: dict[str, pd.DataFrame],
    config: PreprocessConfig,
    artifacts_dir: Path,
) -> tuple[sparse.csr_matrix, dict[str, sparse.csr_matrix], dict[str, sparse.csr_matrix | None]]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_texts = _verification_concat_texts(train_verify)

    ohe = CountVectorizer(
        binary=True,
        max_features=config.max_ohe_features,
        min_df=config.min_df,
        ngram_range=(1, config.ngram_max),
        dtype=np.float32,
    )
    X_train_ohe = ohe.fit_transform(tqdm(train_texts, desc="Fit OHE (train)"))

    X_ohe: dict[str, sparse.csr_matrix] = {"train": X_train_ohe}
    for name, df in other_verify.items():
        texts = _verification_concat_texts(df)
        X_ohe[name] = ohe.transform(tqdm(texts, desc=f"Transform OHE ({name})"))

    joblib.dump(ohe, artifacts_dir / "vectorizer_ohe.joblib")

    X_tfidf: dict[str, sparse.csr_matrix | None] = {k: None for k in X_ohe}
    if config.build_tfidf:
        tfidf = TfidfVectorizer(
            max_features=config.max_tfidf_features,
            min_df=config.min_df,
            ngram_range=(1, config.ngram_max),
            dtype=np.float32,
        )
        X_train_tf = tfidf.fit_transform(train_texts)
        X_tfidf["train"] = X_train_tf
        for name, df in other_verify.items():
            texts = _verification_concat_texts(df)
            X_tfidf[name] = tfidf.transform(texts)
        joblib.dump(tfidf, artifacts_dir / "vectorizer_tfidf.joblib")

    return X_train_ohe, X_ohe, X_tfidf


def fit_transform_article_question_cosine(
    train_mcq: pd.DataFrame,
    mcq_splits: dict[str, pd.DataFrame],
    config: PreprocessConfig,
    artifacts_dir: Path,
) -> dict[str, np.ndarray]:
    """Per-MCQ cosine similarity between article and question TF-IDF vectors (train-fit)."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    corpus = pd.concat(
        [train_mcq["article_clean"], train_mcq["question_clean"]],
        ignore_index=True,
    ).tolist()
    vect = TfidfVectorizer(
        max_features=config.max_aq_tfidf_features,
        min_df=config.min_df,
        ngram_range=(1, config.ngram_max),
        dtype=np.float32,
    )
    vect.fit(tqdm(corpus, desc="Fit article+question TF-IDF"))
    joblib.dump(vect, artifacts_dir / "vectorizer_article_question_tfidf.joblib")

    out: dict[str, np.ndarray] = {}
    for name, df in mcq_splits.items():
        xa = vect.transform(df["article_clean"].tolist())
        xq = vect.transform(df["question_clean"].tolist())
        out[name] = batch_cosine_pairs(xa, xq)
    return out


def broadcast_mcq_cosine_to_verify(verify: pd.DataFrame, mcq_cosine: np.ndarray) -> np.ndarray:
    """Map mcq-level cosine (one per original question) to four verification rows."""
    # mcq_row_id indexes into mcq frame index used when building verify
    ids = verify["mcq_row_id"].to_numpy()
    return mcq_cosine[ids].astype(np.float32)


def save_sparse(path: Path, m: sparse.csr_matrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(path, m)


def run_preprocessing(
    raw_dir: Path | str,
    processed_dir: Path | str,
    config: PreprocessConfig | None = None,
    sample_train: int | None = None,
) -> None:
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    artifacts_dir = processed_dir / "artifacts"
    config = config or PreprocessConfig()

    splits = load_raw_splits(raw_dir)
    if sample_train is not None:
        splits["train"] = splits["train"].sample(
            n=min(sample_train, len(splits["train"])),
            random_state=config.random_seed,
        ).reset_index(drop=True)

    mcq_splits = {name: clean_mcq_frame(df) for name, df in splits.items()}

    verify_splits = {name: mcq_to_verification_long(mcq) for name, mcq in mcq_splits.items()}

    aq_cosine = fit_transform_article_question_cosine(
        mcq_splits["train"],
        mcq_splits,
        config,
        artifacts_dir,
    )

    for name, v in verify_splits.items():
        v = add_handcrafted_lexical_features(v)
        v["feat_cosine_article_question"] = broadcast_mcq_cosine_to_verify(v, aq_cosine[name])
        verify_splits[name] = v

    _, X_ohe, X_tfidf = fit_transform_verification_features(
        verify_splits["train"],
        {k: v for k, v in verify_splits.items() if k != "train"},
        config,
        artifacts_dir,
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    for name, mcq in mcq_splits.items():
        mcq.to_parquet(processed_dir / f"mcq_{name}.parquet", index=False)

    feature_cols = [c for c in verify_splits["train"].columns if c.startswith("feat_")]
    for name, v in verify_splits.items():
        v.to_parquet(processed_dir / f"verify_{name}.parquet", index=False)
        save_sparse(processed_dir / f"verify_{name}_X_ohe.npz", X_ohe[name])
        if X_tfidf.get(name) is not None:
            save_sparse(processed_dir / f"verify_{name}_X_tfidf.npz", X_tfidf[name])  # type: ignore[arg-type]

    manifest = {
        "config": asdict(config),
        "splits": {k: int(len(v)) for k, v in verify_splits.items()},
        "mcq_splits": {k: int(len(v)) for k, v in mcq_splits.items()},
        "ohe_feature_dim": int(X_ohe["train"].shape[1]),
        "tfidf_feature_dim": int(X_tfidf["train"].shape[1]) if X_tfidf["train"] is not None else None,
        "handcrafted_feature_columns": feature_cols,
        "artifacts": {
            "vectorizer_ohe": str(artifacts_dir / "vectorizer_ohe.joblib"),
            "vectorizer_tfidf": str(artifacts_dir / "vectorizer_tfidf.joblib") if config.build_tfidf else None,
            "vectorizer_article_question_tfidf": str(artifacts_dir / "vectorizer_article_question_tfidf.joblib"),
        },
    }
    (processed_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Preprocess RACE CSVs into cleaned tables and features.")
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--max-ohe-features", type=int, default=50_000)
    p.add_argument("--max-tfidf-features", type=int, default=50_000)
    p.add_argument("--max-aq-tfidf-features", type=int, default=8192)
    p.add_argument("--min-df", type=int, default=2)
    p.add_argument("--ngram-max", type=int, default=1)
    p.add_argument("--no-tfidf", action="store_true")
    p.add_argument("--sample-train", type=int, default=None, help="Use only N train MCQs (debug).")
    args = p.parse_args(list(argv) if argv is not None else None)

    cfg = PreprocessConfig(
        max_ohe_features=args.max_ohe_features,
        max_tfidf_features=args.max_tfidf_features,
        max_aq_tfidf_features=args.max_aq_tfidf_features,
        min_df=args.min_df,
        ngram_max=args.ngram_max,
        build_tfidf=not args.no_tfidf,
    )
    run_preprocessing(args.raw_dir, args.processed_dir, cfg, sample_train=args.sample_train)


if __name__ == "__main__":
    main()
