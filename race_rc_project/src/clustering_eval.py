"""Silhouette score, cluster purity, and cluster-size diagnostics for the
unsupervised K-Means components in Model A and Model B.

This script does **not** retrain anything. It loads the already-persisted
KMeans + StandardScaler artifacts produced by ``src.model_a_train`` and
``src.model_b_train``, rebuilds the train-split candidate features using the
same builder functions, and computes:

- ``silhouette_score`` on a (sub-sampled, for tractability) standardized
  feature matrix.
- **Cluster purity** with respect to the weak supervised label
  (``label == 1`` for "best" / "good" candidates). Purity is computed as
  ``sum(max per-cluster class count) / total``, and is reported alongside
  per-cluster class proportions so the "good cluster" choice is auditable.
- Cluster sizes.

Outputs:

- ``models/model_a/traditional/clustering_metrics.json``
- ``models/model_b/traditional/clustering_metrics.json``
- The ``clustering`` key is also patched into the corresponding
  ``metrics_summary.json`` so downstream notebooks / the UI dashboard pick
  the numbers up without code changes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from src.model_a_train import (
    GENERATION_FEAT_COLS,
    _ensure_nltk,
    build_generation_candidates,
)
from src.model_b_train import (
    DISTRACTOR_FEATS,
    HINT_FEATS,
    _build_distractor_candidates,
    _build_hint_candidates,
    _load_mcq_split,
)


@dataclass
class ClusterReport:
    name: str
    n_samples: int
    n_features: int
    n_clusters: int
    silhouette: float
    silhouette_sample_size: int
    purity: float
    cluster_sizes: dict[int, int]
    cluster_label_proportions: dict[int, dict[int, float]]
    good_cluster_id: int


def _purity(cluster_assignments: np.ndarray, labels: np.ndarray) -> float:
    """Sum of max per-cluster class count, divided by total.

    Standard external-validity measure for clustering. 1.0 means each cluster
    is class-pure; 0.5 is the chance level for two balanced classes.
    """
    total = len(cluster_assignments)
    if total == 0:
        return 0.0
    correct = 0
    for c in np.unique(cluster_assignments):
        mask = cluster_assignments == c
        if not mask.any():
            continue
        in_cluster_labels = labels[mask]
        counts = np.bincount(in_cluster_labels.astype(np.int64))
        correct += int(counts.max())
    return float(correct) / float(total)


def _cluster_diagnostics(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    scaler,
    kmeans,
    good_cluster_id: int,
    silhouette_sample: int,
    random_seed: int,
) -> ClusterReport:
    Z = scaler.transform(X)
    assignments = kmeans.predict(Z)

    n = len(assignments)
    sample = min(silhouette_sample, n)
    if sample < 2 or len(np.unique(assignments)) < 2:
        sil = float("nan")
        sample_used = sample
    else:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(n, size=sample, replace=False)
        sil = float(silhouette_score(Z[idx], assignments[idx], metric="euclidean"))
        sample_used = sample

    pur = _purity(assignments, y)

    sizes = {int(c): int((assignments == c).sum()) for c in np.unique(assignments)}
    props: dict[int, dict[int, float]] = {}
    for c in np.unique(assignments):
        mask = assignments == c
        if not mask.any():
            props[int(c)] = {}
            continue
        in_lbl = y[mask]
        counts = np.bincount(in_lbl.astype(np.int64))
        total = int(counts.sum())
        props[int(c)] = {
            int(lbl): float(counts[lbl]) / float(total) for lbl in range(len(counts))
        }

    return ClusterReport(
        name=name,
        n_samples=int(n),
        n_features=int(Z.shape[1]),
        n_clusters=int(kmeans.n_clusters),
        silhouette=sil,
        silhouette_sample_size=int(sample_used),
        purity=pur,
        cluster_sizes=sizes,
        cluster_label_proportions=props,
        good_cluster_id=int(good_cluster_id),
    )


def _report_to_dict(r: ClusterReport) -> dict:
    return {
        "n_samples": r.n_samples,
        "n_features": r.n_features,
        "n_clusters": r.n_clusters,
        "silhouette_score": r.silhouette,
        "silhouette_sample_size": r.silhouette_sample_size,
        "purity": r.purity,
        "cluster_sizes": r.cluster_sizes,
        "cluster_label_proportions": r.cluster_label_proportions,
        "good_cluster_id": r.good_cluster_id,
    }


def _patch_metrics_summary(metrics_path: Path, clustering_block: dict) -> None:
    if not metrics_path.is_file():
        return
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    data["clustering"] = clustering_block
    metrics_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def evaluate_model_a(
    processed_dir: Path,
    model_dir: Path,
    silhouette_sample: int,
    random_seed: int,
) -> dict:
    """Re-build train candidates, load Model A KMeans + scaler, compute metrics."""
    _ensure_nltk()
    meta = json.loads((model_dir / "model_a_meta.json").read_text(encoding="utf-8"))
    good_cluster_id = int(meta.get("good_kmeans_cluster_id", 0))
    top_sents = int(meta.get("generation_top_sentences", 3))

    train_mcq = _load_mcq_split(processed_dir, "train")
    train_cands = build_generation_candidates(train_mcq, top_sents)
    X = train_cands[GENERATION_FEAT_COLS].to_numpy(dtype=np.float32)
    y = train_cands["label"].to_numpy(dtype=np.int8)

    scaler = joblib.load(model_dir / "generation_unsupervised_scaler.joblib")
    kmeans = joblib.load(model_dir / "generation_kmeans.joblib")

    report = _cluster_diagnostics(
        name="model_a_question_generation",
        X=X,
        y=y,
        scaler=scaler,
        kmeans=kmeans,
        good_cluster_id=good_cluster_id,
        silhouette_sample=silhouette_sample,
        random_seed=random_seed,
    )
    return {"question_generation": _report_to_dict(report)}


def evaluate_model_b(
    processed_dir: Path,
    model_dir: Path,
    silhouette_sample: int,
    random_seed: int,
) -> dict:
    """Re-build train candidates and score both Model B KMeans components."""
    meta = json.loads((model_dir / "model_b_meta.json").read_text(encoding="utf-8"))
    d_good = int(meta.get("distractor_good_cluster_id", 0))
    h_good = int(meta.get("hint_good_cluster_id", 0))

    train_mcq = _load_mcq_split(processed_dir, "train")

    out: dict = {}

    d_cands = _build_distractor_candidates(train_mcq)
    if not d_cands.empty:
        Xd = d_cands[DISTRACTOR_FEATS].to_numpy(dtype=np.float32)
        yd = d_cands["label"].to_numpy(dtype=np.int8)
        d_scaler = joblib.load(model_dir / "distractor_scaler.joblib")
        d_km = joblib.load(model_dir / "distractor_kmeans.joblib")
        d_report = _cluster_diagnostics(
            name="model_b_distractor_generation",
            X=Xd,
            y=yd,
            scaler=d_scaler,
            kmeans=d_km,
            good_cluster_id=d_good,
            silhouette_sample=silhouette_sample,
            random_seed=random_seed,
        )
        out["distractor_generation"] = _report_to_dict(d_report)

    h_cands = _build_hint_candidates(train_mcq)
    if not h_cands.empty:
        Xh = h_cands[HINT_FEATS].to_numpy(dtype=np.float32)
        yh = h_cands["label"].to_numpy(dtype=np.int8)
        h_scaler = joblib.load(model_dir / "hint_scaler.joblib")
        h_km = joblib.load(model_dir / "hint_kmeans.joblib")
        h_report = _cluster_diagnostics(
            name="model_b_hint_generation",
            X=Xh,
            y=yh,
            scaler=h_scaler,
            kmeans=h_km,
            good_cluster_id=h_good,
            silhouette_sample=silhouette_sample,
            random_seed=random_seed,
        )
        out["hint_generation"] = _report_to_dict(h_report)

    return out


def _print_block(title: str, block: dict) -> None:
    print(f"\n=== {title} ===")
    rows = []
    for sub_name, sub in block.items():
        rows.append(
            {
                "task": sub_name,
                "n_samples": sub["n_samples"],
                "n_clusters": sub["n_clusters"],
                "silhouette": sub["silhouette_score"],
                "purity": sub["purity"],
                "good_cluster_id": sub["good_cluster_id"],
                "cluster_sizes": sub["cluster_sizes"],
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        print("(no clusterers found)")
        return
    with pd.option_context("display.width", 140, "display.max_colwidth", 60):
        print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute silhouette + purity for the saved KMeans components."
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--model-a-dir", type=Path, default=Path("models/model_a/traditional")
    )
    parser.add_argument(
        "--model-b-dir", type=Path, default=Path("models/model_b/traditional")
    )
    parser.add_argument(
        "--silhouette-sample",
        type=int,
        default=5000,
        help="Random sample size for silhouette (full silhouette is O(n^2)).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    a_block: dict = {}
    if (args.model_a_dir / "generation_kmeans.joblib").is_file():
        a_block = evaluate_model_a(
            args.processed_dir, args.model_a_dir, args.silhouette_sample, args.seed
        )
        (args.model_a_dir / "clustering_metrics.json").write_text(
            json.dumps(a_block, indent=2), encoding="utf-8"
        )
        _patch_metrics_summary(args.model_a_dir / "metrics_summary.json", a_block)
        _print_block("Model A clustering", a_block)
    else:
        print(f"[skip] Model A KMeans not found in {args.model_a_dir}")

    b_block: dict = {}
    if (args.model_b_dir / "distractor_kmeans.joblib").is_file():
        b_block = evaluate_model_b(
            args.processed_dir, args.model_b_dir, args.silhouette_sample, args.seed
        )
        (args.model_b_dir / "clustering_metrics.json").write_text(
            json.dumps(b_block, indent=2), encoding="utf-8"
        )
        _patch_metrics_summary(args.model_b_dir / "metrics_summary.json", b_block)
        _print_block("Model B clustering", b_block)
    else:
        print(f"[skip] Model B KMeans not found in {args.model_b_dir}")


if __name__ == "__main__":
    main()
