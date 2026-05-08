"""
Model A training pipeline.

Implements the *answer verification* sub-task from the project spec:
given (article, question, option) -> predict if option is correct (0/1).

Also includes:
- Required supervised baselines (LogReg, Linear SVM, Naive Bayes, RF)
- Unsupervised experiment (KMeans on handcrafted features)
- Semi-supervised experiment (Label Spreading on handcrafted features)
- Simple soft-voting ensemble over probabilistic models
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import re

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading
from sklearn.svm import LinearSVC


@dataclass
class TrainConfig:
    processed_dir: Path = Path("data/processed")
    output_dir: Path = Path("models/model_a/traditional")
    sparse_feature_kind: str = "ohe"  # ohe or tfidf
    use_handcrafted: bool = True
    max_train_rows: int | None = None
    random_seed: int = 42
    run_unsupervised: bool = True
    run_semi_supervised: bool = True
    semi_supervised_label_frac: float = 0.10
    run_generation: bool = True
    generation_top_sentences: int = 3
    generation_max_train_mcq: int | None = 20000


def _load_manifest(processed_dir: Path) -> dict:
    path = processed_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_split(processed_dir: Path, split: str, sparse_feature_kind: str) -> tuple[pd.DataFrame, sparse.csr_matrix]:
    table_path = processed_dir / f"verify_{split}.parquet"
    matrix_path = processed_dir / f"verify_{split}_X_{sparse_feature_kind}.npz"
    if not table_path.is_file():
        raise FileNotFoundError(f"Missing split table: {table_path}")
    if not matrix_path.is_file():
        raise FileNotFoundError(
            f"Missing sparse feature matrix: {matrix_path}. "
            f"Did you run preprocessing with this feature kind?"
        )
    df = pd.read_parquet(table_path)
    X = sparse.load_npz(matrix_path).tocsr()
    return df, X


def _load_mcq_split(processed_dir: Path, split: str) -> pd.DataFrame:
    path = processed_dir / f"mcq_{split}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Missing MCQ split file: {path}")
    return pd.read_parquet(path)


def _extract_xy(
    df: pd.DataFrame,
    X_sparse: sparse.csr_matrix,
    handcrafted_cols: list[str],
    use_handcrafted: bool,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    y = df["label"].to_numpy().astype(np.int8)
    if not use_handcrafted:
        return X_sparse, y
    dense_handcrafted = df[handcrafted_cols].to_numpy(dtype=np.float32)
    X_dense_sparse = sparse.csr_matrix(dense_handcrafted)
    X = sparse.hstack([X_sparse, X_dense_sparse], format="csr")
    return X, y


def _basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "exact_match": float(np.mean(y_true == y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def _fit_and_eval_model(
    name: str,
    model,
    X_train: sparse.csr_matrix,
    y_train: np.ndarray,
    X_val: sparse.csr_matrix,
    y_val: np.ndarray,
    X_test: sparse.csr_matrix,
    y_test: np.ndarray,
) -> tuple[dict, object]:
    t0 = perf_counter()
    model.fit(X_train, y_train)
    train_seconds = perf_counter() - t0

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    return (
        {
            "model_name": name,
            "train_seconds": train_seconds,
            "validation": _basic_metrics(y_val, val_pred),
            "test": _basic_metrics(y_test, test_pred),
        },
        model,
    )


def _cluster_purity(y_true: np.ndarray, clusters: np.ndarray) -> float:
    total = 0
    for c in np.unique(clusters):
        idx = np.where(clusters == c)[0]
        if len(idx) == 0:
            continue
        labels = y_true[idx]
        counts = np.bincount(labels, minlength=2)
        total += counts.max()
    return float(total / len(y_true))


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _token_set(s: str) -> set[str]:
    return set(_normalize_text(s).split()) if s else set()


def _jaccard(a: str, b: str) -> float:
    sa, sb = _token_set(a), _token_set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def _guess_wh_word(answer: str) -> str:
    a = _normalize_text(answer)
    if re.search(r"\b(he|she|him|her|person|people|man|woman|teacher|student)\b", a):
        return "who"
    if re.search(r"\b(city|country|school|park|street|place|room|home)\b", a):
        return "where"
    if re.search(r"\b(\d{4}|monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|september|october|november|december)\b", a):
        return "when"
    if re.search(r"\b(because|reason|cause)\b", a):
        return "why"
    return "what"


def _split_sentences(article: str) -> list[str]:
    text = (article or "").strip()
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    out = [p.strip() for p in parts if p and p.strip()]
    return out if out else [text]


def _template_question(sentence: str, answer: str, wh_word: str) -> tuple[str, int]:
    sent = sentence.strip()
    if not sent:
        return f"{wh_word.capitalize()} is the correct answer?", 0
    ans = answer.strip()
    if ans and re.search(re.escape(ans), sent, flags=re.IGNORECASE):
        masked = re.sub(re.escape(ans), "____", sent, count=1, flags=re.IGNORECASE)
        return f"{wh_word.capitalize()} fits in the blank: {masked}?", 1
    return f"{wh_word.capitalize()} is best supported by this statement: {sent}?", 0


def _build_generation_candidates(mcq_df: pd.DataFrame, top_sentences: int) -> pd.DataFrame:
    records: list[dict] = []
    for idx, row in mcq_df.reset_index(drop=True).iterrows():
        answer_letter = str(row["answer"]).strip().upper()
        answer_text = str(row.get(answer_letter, "")) if answer_letter in ("A", "B", "C", "D") else ""
        article = str(row.get("article", ""))
        gold_question = str(row.get("question", ""))
        wh = _guess_wh_word(answer_text)
        sentences = _split_sentences(article)
        if not sentences:
            continue
        scored = []
        for s in sentences:
            scored.append((s, _jaccard(s, answer_text), _jaccard(s, gold_question)))
        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
        for rank, (sent, overlap_ans, overlap_q) in enumerate(scored[: max(1, top_sentences)]):
            gen_q, has_blank = _template_question(sent, answer_text, wh)
            sim_gold = _jaccard(gen_q, gold_question)
            records.append(
                {
                    "mcq_row_id": idx,
                    "id": row.get("id", str(idx)),
                    "answer_text": answer_text,
                    "gold_question": gold_question,
                    "candidate_question": gen_q,
                    "sentence_text": sent,
                    "sent_answer_overlap": overlap_ans,
                    "sent_question_overlap": overlap_q,
                    "candidate_gold_similarity": sim_gold,
                    "candidate_len": len(_token_set(gen_q)),
                    "sentence_len": len(_token_set(sent)),
                    "has_blank": has_blank,
                    "wh_what": int(wh == "what"),
                    "wh_who": int(wh == "who"),
                    "wh_where": int(wh == "where"),
                    "wh_when": int(wh == "when"),
                    "wh_why": int(wh == "why"),
                    "candidate_rank": rank,
                }
            )
    cands = pd.DataFrame(records)
    if cands.empty:
        return cands
    cands["label"] = 0
    best_idx = cands.groupby("mcq_row_id")["candidate_gold_similarity"].idxmax()
    cands.loc[best_idx, "label"] = 1
    return cands


def _run_generation_training(
    cfg: TrainConfig,
    train_mcq: pd.DataFrame,
    val_mcq: pd.DataFrame,
    test_mcq: pd.DataFrame,
) -> dict:
    if cfg.generation_max_train_mcq is not None:
        n = min(cfg.generation_max_train_mcq, len(train_mcq))
        train_mcq = train_mcq.iloc[:n].reset_index(drop=True)

    train_cands = _build_generation_candidates(train_mcq, cfg.generation_top_sentences)
    val_cands = _build_generation_candidates(val_mcq, cfg.generation_top_sentences)
    test_cands = _build_generation_candidates(test_mcq, cfg.generation_top_sentences)

    if train_cands.empty or val_cands.empty or test_cands.empty:
        return {"status": "skipped", "reason": "No generation candidates were produced."}

    feat_cols = [
        "sent_answer_overlap",
        "sent_question_overlap",
        "candidate_gold_similarity",
        "candidate_len",
        "sentence_len",
        "has_blank",
        "wh_what",
        "wh_who",
        "wh_where",
        "wh_when",
        "wh_why",
        "candidate_rank",
    ]
    X_train = train_cands[feat_cols].to_numpy(dtype=np.float32)
    y_train = train_cands["label"].to_numpy(dtype=np.int8)
    X_val = val_cands[feat_cols].to_numpy(dtype=np.float32)
    X_test = test_cands[feat_cols].to_numpy(dtype=np.float32)

    ranker = LogisticRegression(
        solver="liblinear",
        max_iter=400,
        random_state=cfg.random_seed,
    )
    ranker.fit(X_train, y_train)
    joblib.dump(ranker, cfg.output_dir / "generation_ranker.joblib")
    (cfg.output_dir / "generation_feature_columns.json").write_text(json.dumps(feat_cols, indent=2), encoding="utf-8")

    def top1_accuracy(cands: pd.DataFrame, probs: np.ndarray) -> float:
        tmp = cands.copy()
        tmp["score"] = probs
        pred_best = tmp.groupby("mcq_row_id")["score"].idxmax()
        true_best = tmp.groupby("mcq_row_id")["label"].idxmax()
        return float(np.mean(pred_best.to_numpy() == true_best.to_numpy()))

    val_probs = ranker.predict_proba(X_val)[:, 1]
    test_probs = ranker.predict_proba(X_test)[:, 1]
    val_top1 = top1_accuracy(val_cands, val_probs)
    test_top1 = top1_accuracy(test_cands, test_probs)

    # Save a quick qualitative sample for reporting/demo.
    sample = val_cands.copy()
    sample["score"] = val_probs
    sample = sample.sort_values(["mcq_row_id", "score"], ascending=[True, False]).groupby("mcq_row_id").head(1)
    sample[["id", "gold_question", "candidate_question", "score"]].head(30).to_csv(
        cfg.output_dir / "generation_preview.csv", index=False
    )

    return {
        "status": "ok",
        "train_candidates": int(len(train_cands)),
        "validation_candidates": int(len(val_cands)),
        "test_candidates": int(len(test_cands)),
        "top1_accuracy_validation": val_top1,
        "top1_accuracy_test": test_top1,
        "feature_columns": feat_cols,
    }


def run_training(cfg: TrainConfig) -> None:
    manifest = _load_manifest(cfg.processed_dir)
    handcrafted_cols: list[str] = manifest["handcrafted_feature_columns"]

    train_df, train_sparse = _load_split(cfg.processed_dir, "train", cfg.sparse_feature_kind)
    val_df, val_sparse = _load_split(cfg.processed_dir, "validation", cfg.sparse_feature_kind)
    test_df, test_sparse = _load_split(cfg.processed_dir, "test", cfg.sparse_feature_kind)
    train_mcq = _load_mcq_split(cfg.processed_dir, "train")
    val_mcq = _load_mcq_split(cfg.processed_dir, "validation")
    test_mcq = _load_mcq_split(cfg.processed_dir, "test")

    if cfg.max_train_rows is not None:
        n = min(cfg.max_train_rows, len(train_df))
        train_df = train_df.iloc[:n].reset_index(drop=True)
        train_sparse = train_sparse[:n]

    X_train, y_train = _extract_xy(train_df, train_sparse, handcrafted_cols, cfg.use_handcrafted)
    X_val, y_val = _extract_xy(val_df, val_sparse, handcrafted_cols, cfg.use_handcrafted)
    X_test, y_test = _extract_xy(test_df, test_sparse, handcrafted_cols, cfg.use_handcrafted)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "logistic_regression": LogisticRegression(
            solver="saga",
            max_iter=300,
            n_jobs=-1,
            random_state=cfg.random_seed,
        ),
        "linear_svm": LinearSVC(
            C=1.0,
            random_state=cfg.random_seed,
        ),
        "bernoulli_nb": BernoulliNB(),
    }

    # RF is trained only on handcrafted features for speed/stability.
    rf_uses_handcrafted = True
    rf = RandomForestClassifier(
        n_estimators=250,
        max_depth=20,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=cfg.random_seed,
    )

    all_results: dict[str, dict] = {}
    trained_models: dict[str, object] = {}

    for name, model in models.items():
        result, fitted = _fit_and_eval_model(
            name,
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
        )
        all_results[name] = result
        trained_models[name] = fitted
        joblib.dump(fitted, cfg.output_dir / f"{name}.joblib")

    if rf_uses_handcrafted:
        Xh_train = train_df[handcrafted_cols].to_numpy(dtype=np.float32)
        Xh_val = val_df[handcrafted_cols].to_numpy(dtype=np.float32)
        Xh_test = test_df[handcrafted_cols].to_numpy(dtype=np.float32)
        rf_result, rf_fitted = _fit_and_eval_model(
            "random_forest_handcrafted",
            rf,
            sparse.csr_matrix(Xh_train),
            y_train,
            sparse.csr_matrix(Xh_val),
            y_val,
            sparse.csr_matrix(Xh_test),
            y_test,
        )
        all_results["random_forest_handcrafted"] = rf_result
        trained_models["random_forest_handcrafted"] = rf_fitted
        joblib.dump(rf_fitted, cfg.output_dir / "random_forest_handcrafted.joblib")

    # Soft-voting ensemble over probabilistic models.
    ensemble_members = []
    for k in ("logistic_regression", "bernoulli_nb"):
        m = trained_models.get(k)
        if m is not None and hasattr(m, "predict_proba"):
            ensemble_members.append(k)
    if ensemble_members:
        val_probs = np.mean(
            [trained_models[k].predict_proba(X_val)[:, 1] for k in ensemble_members],  # type: ignore[attr-defined]
            axis=0,
        )
        test_probs = np.mean(
            [trained_models[k].predict_proba(X_test)[:, 1] for k in ensemble_members],  # type: ignore[attr-defined]
            axis=0,
        )
        val_pred = (val_probs >= 0.5).astype(np.int8)
        test_pred = (test_probs >= 0.5).astype(np.int8)
        all_results["soft_voting_ensemble"] = {
            "model_name": "soft_voting_ensemble",
            "members": ensemble_members,
            "validation": _basic_metrics(y_val, val_pred),
            "test": _basic_metrics(y_test, test_pred),
        }

    # Unsupervised and semi-supervised on handcrafted features (dense, scalable).
    Xh_train = train_df[handcrafted_cols].to_numpy(dtype=np.float32)
    Xh_val = val_df[handcrafted_cols].to_numpy(dtype=np.float32)
    Xh_test = test_df[handcrafted_cols].to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    Xh_train_scaled = scaler.fit_transform(Xh_train)
    Xh_val_scaled = scaler.transform(Xh_val)
    Xh_test_scaled = scaler.transform(Xh_test)
    joblib.dump(scaler, cfg.output_dir / "handcrafted_scaler.joblib")

    if cfg.run_unsupervised:
        km = KMeans(n_clusters=2, random_state=cfg.random_seed, n_init=10)
        train_clusters = km.fit_predict(Xh_train_scaled)
        val_clusters = km.predict(Xh_val_scaled)
        test_clusters = km.predict(Xh_test_scaled)
        joblib.dump(km, cfg.output_dir / "kmeans.joblib")
        all_results["kmeans_unsupervised"] = {
            "model_name": "kmeans_unsupervised",
            "train_silhouette": float(silhouette_score(Xh_train_scaled, train_clusters)),
            "train_purity": _cluster_purity(y_train, train_clusters),
            "validation_purity": _cluster_purity(y_val, val_clusters),
            "test_purity": _cluster_purity(y_test, test_clusters),
        }

    if cfg.run_semi_supervised:
        rng = np.random.default_rng(cfg.random_seed)
        y_semi = y_train.copy().astype(int)
        keep = rng.random(len(y_semi)) < cfg.semi_supervised_label_frac
        y_semi[~keep] = -1
        semi = LabelSpreading(kernel="knn", n_neighbors=10, alpha=0.2, max_iter=30)
        semi.fit(Xh_train_scaled, y_semi)
        val_pred = semi.predict(Xh_val_scaled)
        test_pred = semi.predict(Xh_test_scaled)
        joblib.dump(semi, cfg.output_dir / "label_spreading.joblib")
        all_results["label_spreading_semi_supervised"] = {
            "model_name": "label_spreading_semi_supervised",
            "labeled_fraction": float(cfg.semi_supervised_label_frac),
            "validation": _basic_metrics(y_val, val_pred),
            "test": _basic_metrics(y_test, test_pred),
        }

    generation_results = None
    if cfg.run_generation:
        generation_results = _run_generation_training(cfg, train_mcq, val_mcq, test_mcq)

    # Save run metadata + metrics.
    summary = {
        "config": {
            "processed_dir": str(cfg.processed_dir),
            "output_dir": str(cfg.output_dir),
            "sparse_feature_kind": cfg.sparse_feature_kind,
            "use_handcrafted": cfg.use_handcrafted,
            "max_train_rows": cfg.max_train_rows,
            "random_seed": cfg.random_seed,
            "run_unsupervised": cfg.run_unsupervised,
            "run_semi_supervised": cfg.run_semi_supervised,
            "semi_supervised_label_frac": cfg.semi_supervised_label_frac,
            "run_generation": cfg.run_generation,
            "generation_top_sentences": cfg.generation_top_sentences,
            "generation_max_train_mcq": cfg.generation_max_train_mcq,
        },
        "split_rows": {
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "results": all_results,
    }
    if generation_results is not None:
        summary["generation"] = generation_results
    (cfg.output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Train Model A answer verification baselines.")
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--output-dir", type=Path, default=Path("models/model_a/traditional"))
    p.add_argument(
        "--sparse-feature-kind",
        choices=("ohe", "tfidf"),
        default="ohe",
        help="Which sparse feature matrix to use from preprocessing outputs.",
    )
    p.add_argument("--no-handcrafted", action="store_true", help="Disable handcrafted lexical features.")
    p.add_argument("--max-train-rows", type=int, default=None, help="Debug mode: limit training rows.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-unsupervised", action="store_true")
    p.add_argument("--skip-semi-supervised", action="store_true")
    p.add_argument("--semi-label-frac", type=float, default=0.10)
    p.add_argument("--skip-generation", action="store_true")
    p.add_argument("--generation-top-sentences", type=int, default=3)
    p.add_argument("--generation-max-train-mcq", type=int, default=20000)
    args = p.parse_args()

    cfg = TrainConfig(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        sparse_feature_kind=args.sparse_feature_kind,
        use_handcrafted=not args.no_handcrafted,
        max_train_rows=args.max_train_rows,
        random_seed=args.seed,
        run_unsupervised=not args.skip_unsupervised,
        run_semi_supervised=not args.skip_semi_supervised,
        semi_supervised_label_frac=args.semi_label_frac,
        run_generation=not args.skip_generation,
        generation_top_sentences=args.generation_top_sentences,
        generation_max_train_mcq=args.generation_max_train_mcq,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
