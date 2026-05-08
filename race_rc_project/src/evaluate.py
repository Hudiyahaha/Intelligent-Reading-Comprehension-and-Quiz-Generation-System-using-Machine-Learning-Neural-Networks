"""Shared evaluation helpers and a comparison CLI.

This module gives the rest of the project two things:

1. **Shared metric helpers.** ``bleu``, ``meteor``, ``rouge_agg``, and
   ``evaluate_text_generation`` are thin re-exports of the BLEU / ROUGE /
   METEOR implementations used internally by Model A and Model B, so a notebook
   or downstream script can call ``from src.evaluate import bleu`` without
   needing to know which trainer module owns each helper.

2. **A side-by-side CLI.** ``python -m src.evaluate`` loads the
   ``metrics_summary.json`` files produced by Model A, Model B, and the BERT
   rerank baseline (when present) and prints a tidy comparison table per task
   (question generation / distractor generation / hint generation). This is the
   command-line equivalent of section 3 in ``notebooks/experiments.ipynb`` and
   gives the user a quick way to inspect the latest run without opening Jupyter.

Usage::

    python -m src.evaluate                 # print model summaries + comparison
    python -m src.evaluate --models-dir models
    python -m src.evaluate --json          # emit the merged comparison as JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.model_a_train import _bleu as _bleu_a
from src.model_a_train import _meteor as _meteor_a
from src.model_a_train import _rouge_agg as _rouge_agg_a
from src.model_b_train import _evaluate_text_generation as _eval_text_b

bleu = _bleu_a
meteor = _meteor_a
rouge_agg = _rouge_agg_a
evaluate_text_generation = _eval_text_b


METRIC_COLS: tuple[str, ...] = (
    "bleu_mean",
    "rouge1_f_mean",
    "rouge2_f_mean",
    "rougeL_f_mean",
    "meteor_mean",
)


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_all_metrics(models_dir: Path) -> dict[str, dict | None]:
    """Load metrics_summary.json for Model A, Model B, and the BERT baseline."""
    return {
        "model_a": _read_json(models_dir / "model_a" / "traditional" / "metrics_summary.json"),
        "model_b": _read_json(models_dir / "model_b" / "traditional" / "metrics_summary.json"),
        "bert": _read_json(models_dir / "model_bert" / "metrics_summary.json"),
    }


def _pick(d: dict | None, *keys: str) -> dict | None:
    """Walk a nested dict, returning None if any key is missing."""
    cur: dict | None = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur if isinstance(cur, dict) else None


def build_comparison(models: dict[str, dict | None]) -> dict:
    """Produce the same shape as ``models/comparison_summary.json``.

    The output groups metrics by task (question / distractor / hint) and then
    by split (validation / test), with one entry per model that has metrics for
    that task.
    """
    a, b, bert = models.get("model_a"), models.get("model_b"), models.get("bert")
    return {
        "tasks": {
            "question_generation": {
                "validation": {
                    "model_a": _pick(a, "validation"),
                    "bert": _pick(bert, "validation", "question_generation"),
                },
                "test": {
                    "model_a": _pick(a, "test"),
                    "bert": _pick(bert, "test", "question_generation"),
                },
            },
            "distractor_generation": {
                "validation": {
                    "model_b": _pick(b, "validation", "distractor_generation"),
                    "bert": _pick(bert, "validation", "distractor_generation"),
                },
                "test": {
                    "model_b": _pick(b, "test", "distractor_generation"),
                    "bert": _pick(bert, "test", "distractor_generation"),
                },
            },
            "hint_generation": {
                "validation": {
                    "model_b": _pick(b, "validation", "hint_generation"),
                    "bert": _pick(bert, "validation", "hint_generation"),
                },
                "test": {
                    "model_b": _pick(b, "test", "hint_generation"),
                    "bert": _pick(bert, "test", "hint_generation"),
                },
            },
        }
    }


def task_table(task_block: dict) -> pd.DataFrame:
    """Convert a single task block into a tidy DataFrame for printing."""
    rows: dict[str, dict[str, float | int | None]] = {m: {} for m in METRIC_COLS + ("n_examples",)}
    for split, models in task_block.items():
        if not isinstance(models, dict):
            continue
        for name, metrics in models.items():
            if not metrics:
                continue
            col = f"{name}.{split}"
            for m in METRIC_COLS + ("n_examples",):
                rows[m][col] = metrics.get(m)
    df = pd.DataFrame(rows).T
    df.index.name = "metric"
    return df


def print_comparison(comparison: dict) -> None:
    any_data = False
    for task, block in comparison.get("tasks", {}).items():
        df = task_table(block)
        if df.empty or df.dropna(how="all").empty:
            continue
        any_data = True
        print(f"\n=== {task} ===")
        print(df.to_string(float_format=lambda v: f"{v:.4f}", na_rep="—"))
    if not any_data:
        print(
            "No metrics found. Train Model A, Model B, and (optionally) the BERT "
            "baseline first, then re-run this command."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Print a side-by-side BLEU/ROUGE/METEOR comparison across Model A, "
            "Model B, and the BERT rerank baseline."
        )
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Root directory containing model_a/, model_b/, model_bert/.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the merged comparison object as JSON instead of human tables.",
    )
    args = parser.parse_args()

    models = load_all_metrics(args.models_dir)
    comparison = build_comparison(models)

    if args.json:
        print(json.dumps(comparison, indent=2))
    else:
        print_comparison(comparison)


if __name__ == "__main__":
    main()
