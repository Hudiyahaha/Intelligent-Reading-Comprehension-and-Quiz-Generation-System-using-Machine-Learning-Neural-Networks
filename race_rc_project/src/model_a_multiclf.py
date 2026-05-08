"""
Model A — multi-classifier extension and soft-voting ensemble.

This module addresses the rubric items that ask for ``>= 2 traditional ML
classifiers`` and a named ensemble strategy (soft / hard voting or stacking).
It deliberately does **not** replace ``src.model_a_train`` — that module is
the supervised + unsupervised (LR + KMeans) ensemble baseline. Here we add
three named classifiers (LR, Linear SVM with Platt calibration, Random
Forest) trained on the same 12 candidate features, plus a
``VotingClassifier`` (soft voting) over the three, and report BLEU / ROUGE /
METEOR for each.

Outputs (saved next to the original Model A artifacts so the UI dashboard
and `src.evaluate` automatically pick them up):

- ``models/model_a/traditional/classifier_{lr,svm,rf,soft_vote}.joblib``
- ``models/model_a/traditional/multiclf_metrics.json``
- The ``classifiers`` block is also patched into the existing
  ``metrics_summary.json`` so downstream code paths don't need to know about
  this script.

Run::

    python -m src.model_a_multiclf

Useful flags::

    --processed-dir data/processed
    --output-dir    models/model_a/traditional
    --top-sentences 3
    --rf-n-estimators 100
    --rf-max-depth   None
    --max-eval-mcq   None   # cap evaluation rows; None = use full val/test
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.model_a_train import (
    GENERATION_FEAT_COLS,
    _bleu,
    _ensure_nltk,
    _evaluate_generation,
    _load_mcq_split,
    _minmax_by_group,
    _meteor,
    _rouge_agg,
    build_generation_candidates,
)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


@dataclass
class MultiClfConfig:
    processed_dir: Path = Path("data/processed")
    output_dir: Path = Path("models/model_a/traditional")
    random_seed: int = 42
    top_sentences: int = 3
    max_train_mcq: int | None = None
    max_val_mcq: int | None = None
    max_test_mcq: int | None = None
    max_eval_mcq: int | None = None

    rf_n_estimators: int = 100
    rf_max_depth: int | None = None
    svm_C: float = 1.0


def _patch_metrics_summary(
    path: Path, multiclf_block: dict, ensemble_block: dict | None
) -> None:
    """Add ``classifiers`` and optionally ``ensemble`` keys to metrics_summary.json."""
    if not path.is_file():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    data["classifiers"] = multiclf_block
    if ensemble_block is not None:
        data["ensemble"] = ensemble_block
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _rank_by_classifier(
    cands: pd.DataFrame,
    scaler: StandardScaler,
    clf,
    feature_cols: list[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return ensemble_score (top-1 picker) for each candidate row.

    The ensemble_score is the per-MCQ min-max normalized probability that the
    candidate is the gold-question match. Per-MCQ normalization keeps the
    scoring comparable to ``model_a_train``'s ensemble logic.
    """
    X = cands[feature_cols].to_numpy(dtype=np.float32)
    Z = scaler.transform(X)
    proba = clf.predict_proba(Z)[:, 1]
    gids = cands["mcq_row_id"].to_numpy()
    score = _minmax_by_group(proba, gids)
    return score, cands


def _build_classifiers(cfg: MultiClfConfig) -> dict:
    """Return un-fit classifier instances keyed by short name."""
    return {
        "lr": LogisticRegression(
            solver="liblinear", max_iter=500, random_state=cfg.random_seed
        ),
        "svm": CalibratedClassifierCV(
            LinearSVC(C=cfg.svm_C, random_state=cfg.random_seed, dual=False),
            cv=3,
            method="sigmoid",
        ),
        "rf": RandomForestClassifier(
            n_estimators=cfg.rf_n_estimators,
            max_depth=cfg.rf_max_depth,
            n_jobs=-1,
            random_state=cfg.random_seed,
        ),
    }


def run_multiclf(cfg: MultiClfConfig) -> dict:
    _ensure_nltk()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    train_mcq = _load_mcq_split(cfg.processed_dir, "train")
    val_mcq = _load_mcq_split(cfg.processed_dir, "validation")
    test_mcq = _load_mcq_split(cfg.processed_dir, "test")

    if cfg.max_train_mcq is not None:
        train_mcq = train_mcq.iloc[: cfg.max_train_mcq].reset_index(drop=True)
    if cfg.max_val_mcq is not None:
        val_mcq = val_mcq.iloc[: cfg.max_val_mcq].reset_index(drop=True)
    if cfg.max_test_mcq is not None:
        test_mcq = test_mcq.iloc[: cfg.max_test_mcq].reset_index(drop=True)

    print("Building candidate sets ...")
    t0 = perf_counter()
    train_cands = build_generation_candidates(train_mcq, cfg.top_sentences)
    val_cands = build_generation_candidates(val_mcq, cfg.top_sentences)
    test_cands = build_generation_candidates(test_mcq, cfg.top_sentences)
    print(f"  candidates: train={len(train_cands)} val={len(val_cands)} test={len(test_cands)}")
    print(f"  build time: {perf_counter() - t0:.1f}s")

    if train_cands.empty or val_cands.empty or test_cands.empty:
        raise RuntimeError("Empty candidate set; check the processed parquet files.")

    feat_cols = list(GENERATION_FEAT_COLS)
    X_train = train_cands[feat_cols].to_numpy(dtype=np.float32)
    y_train = train_cands["label"].to_numpy(dtype=np.int8)

    scaler = StandardScaler()
    Z_train = scaler.fit_transform(X_train)
    joblib.dump(scaler, cfg.output_dir / "classifier_scaler.joblib")

    base_clfs = _build_classifiers(cfg)

    print("Training individual classifiers ...")
    train_times: dict[str, float] = {}
    fitted: dict = {}
    for name, clf in base_clfs.items():
        t = perf_counter()
        clf.fit(Z_train, y_train)
        train_times[name] = perf_counter() - t
        joblib.dump(clf, cfg.output_dir / f"classifier_{name}.joblib")
        print(f"  {name}: {train_times[name]:.1f}s")
        fitted[name] = clf

    print("Training soft-voting ensemble (LR + SVM + RF) ...")
    voting = VotingClassifier(
        estimators=[
            ("lr", _build_classifiers(cfg)["lr"]),
            ("svm", _build_classifiers(cfg)["svm"]),
            ("rf", _build_classifiers(cfg)["rf"]),
        ],
        voting="soft",
        n_jobs=None,
    )
    t = perf_counter()
    voting.fit(Z_train, y_train)
    train_times["soft_vote"] = perf_counter() - t
    joblib.dump(voting, cfg.output_dir / "classifier_soft_vote.joblib")
    print(f"  soft_vote: {train_times['soft_vote']:.1f}s")
    fitted["soft_vote"] = voting

    rouge_s = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    def _score_and_evaluate(name: str, clf) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
        ens_val, _ = _rank_by_classifier(val_cands, scaler, clf, feat_cols)
        ens_test, _ = _rank_by_classifier(test_cands, scaler, clf, feat_cols)
        val_metrics, val_detail = _evaluate_generation(val_cands, ens_val, rouge_s)
        test_metrics, test_detail = _evaluate_generation(test_cands, ens_test, rouge_s)
        return val_metrics, test_metrics, val_detail, test_detail

    print("Evaluating classifiers on val + test ...")
    classifiers_block: dict[str, dict] = {}
    detail_paths: dict[str, dict] = {}
    for name, clf in fitted.items():
        t = perf_counter()
        val_m, test_m, val_d, test_d = _score_and_evaluate(name, clf)
        eval_seconds = perf_counter() - t
        if cfg.max_eval_mcq is not None:
            val_d = val_d.head(cfg.max_eval_mcq)
            test_d = test_d.head(cfg.max_eval_mcq)
        val_path = cfg.output_dir / f"classifier_{name}_val_predictions.csv"
        test_path = cfg.output_dir / f"classifier_{name}_test_predictions.csv"
        val_d.to_csv(val_path, index=False)
        test_d.to_csv(test_path, index=False)
        classifiers_block[name] = {
            "validation": val_m,
            "test": test_m,
            "train_seconds": train_times[name],
            "eval_seconds": eval_seconds,
        }
        detail_paths[name] = {"val_csv": str(val_path), "test_csv": str(test_path)}
        print(
            f"  {name}: test BLEU={test_m['bleu_mean']:.4f} "
            f"ROUGE-1={test_m['rouge1_f_mean']:.4f} METEOR={test_m['meteor_mean']:.4f}"
        )

    ensemble_block = {
        "name": "soft_voting(LR + SVM + RF)",
        "voting": "soft",
        "base_estimators": ["LogisticRegression", "LinearSVC+Platt", "RandomForestClassifier"],
        "validation": classifiers_block["soft_vote"]["validation"],
        "test": classifiers_block["soft_vote"]["test"],
        "best_individual_on_test": max(
            (c for c in ("lr", "svm", "rf")),
            key=lambda n: classifiers_block[n]["test"]["meteor_mean"],
        ),
    }

    summary = {
        "model": "Model A multi-classifier rerank",
        "evaluation": "BLEU / ROUGE / METEOR vs reference question text",
        "feature_columns": feat_cols,
        "config": {
            "processed_dir": str(cfg.processed_dir),
            "output_dir": str(cfg.output_dir),
            "random_seed": cfg.random_seed,
            "top_sentences": cfg.top_sentences,
            "max_train_mcq": cfg.max_train_mcq,
            "max_val_mcq": cfg.max_val_mcq,
            "max_test_mcq": cfg.max_test_mcq,
            "rf_n_estimators": cfg.rf_n_estimators,
            "rf_max_depth": cfg.rf_max_depth,
            "svm_C": cfg.svm_C,
        },
        "classifiers": classifiers_block,
        "ensemble": ensemble_block,
        "detail_paths": detail_paths,
    }
    (cfg.output_dir / "multiclf_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _patch_metrics_summary(
        cfg.output_dir / "metrics_summary.json", classifiers_block, ensemble_block
    )
    return summary


def _print_comparison(summary: dict) -> None:
    rows = []
    for name, block in summary["classifiers"].items():
        for split in ("validation", "test"):
            m = block[split]
            rows.append(
                {
                    "classifier": name,
                    "split": split,
                    "n": m["n_examples"],
                    "BLEU": m["bleu_mean"],
                    "ROUGE-1": m["rouge1_f_mean"],
                    "ROUGE-2": m["rouge2_f_mean"],
                    "ROUGE-L": m["rougeL_f_mean"],
                    "METEOR": m["meteor_mean"],
                }
            )
    df = pd.DataFrame(rows)
    print("\nModel A — individual classifiers and soft-vote ensemble:\n")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(
        "\nBest individual classifier on test METEOR:",
        summary["ensemble"]["best_individual_on_test"],
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train multiple Model A classifiers and a soft-voting ensemble."
    )
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--output-dir", type=Path, default=Path("models/model_a/traditional"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-sentences", type=int, default=3)
    p.add_argument("--max-train-mcq", type=int, default=None)
    p.add_argument("--max-val-mcq", type=int, default=None)
    p.add_argument("--max-test-mcq", type=int, default=None)
    p.add_argument("--max-eval-mcq", type=int, default=None)
    p.add_argument("--rf-n-estimators", type=int, default=100)
    p.add_argument("--rf-max-depth", type=int, default=None)
    p.add_argument("--svm-c", type=float, default=1.0)
    args = p.parse_args()

    cfg = MultiClfConfig(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        random_seed=args.seed,
        top_sentences=args.top_sentences,
        max_train_mcq=args.max_train_mcq,
        max_val_mcq=args.max_val_mcq,
        max_test_mcq=args.max_test_mcq,
        max_eval_mcq=args.max_eval_mcq,
        rf_n_estimators=args.rf_n_estimators,
        rf_max_depth=args.rf_max_depth,
        svm_C=args.svm_c,
    )
    summary = run_multiclf(cfg)
    _print_comparison(summary)


if __name__ == "__main__":
    main()
