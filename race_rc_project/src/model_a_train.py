"""
Model A — classical (non-neural) question generation from a passage.

Components (per updated project brief):
- Supervised: logistic ranker over template-generated candidates
- Unsupervised: KMeans on candidate features; score by proximity to the "high-quality" cluster
- Ensemble: convex combination of normalized supervised and unsupervised scores

Evaluation (generation): BLEU, ROUGE, METEOR vs reference question text (no accuracy/precision reporting).
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning, module="nltk")


@dataclass
class TrainConfig:
    processed_dir: Path = Path("data/processed")
    output_dir: Path = Path("models/model_a/traditional")
    random_seed: int = 42
    generation_top_sentences: int = 3
    generation_max_train_mcq: int | None = None
    generation_max_val_mcq: int | None = None
    generation_max_test_mcq: int | None = None
    ensemble_weight_supervised: float = 0.5
    max_eval_mcq: int | None = None


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _ensure_nltk() -> None:
    import nltk

    for pkg in ("punkt", "punkt_tab", "wordnet", "omw-1.4"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass


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
    if re.search(
        r"\b(\d{4}|monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|september|october|november|december)\b",
        a,
    ):
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


def _load_mcq_split(processed_dir: Path, split: str) -> pd.DataFrame:
    path = processed_dir / f"mcq_{split}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Missing MCQ split file: {path}")
    return pd.read_parquet(path)


GENERATION_FEAT_COLS = [
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


def build_generation_candidates(mcq_df: pd.DataFrame, top_sentences: int) -> pd.DataFrame:
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


def _minmax_by_group(scores: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
    out = np.zeros_like(scores, dtype=np.float64)
    for g in np.unique(group_ids):
        m = group_ids == g
        s = scores[m]
        lo, hi = float(s.min()), float(s.max())
        if hi - lo < 1e-12:
            out[m] = 1.0
        else:
            out[m] = (s - lo) / (hi - lo)
    return out


def _fit_unsupervised(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_seed: int,
) -> tuple[KMeans, StandardScaler, int]:
    scaler = StandardScaler()
    Z = scaler.fit_transform(X_train)
    km = KMeans(n_clusters=2, random_state=random_seed, n_init=10)
    clusters = km.fit_predict(Z)
    quality = []
    for c in (0, 1):
        m = clusters == c
        quality.append(float(y_train[m].mean()) if m.any() else 0.0)
    good_cluster = int(np.argmax(quality))
    return km, scaler, good_cluster


def _unsupervised_scores(Z: np.ndarray, km: KMeans, good_cluster: int) -> np.ndarray:
    centroid = km.cluster_centers_[good_cluster]
    dist = np.linalg.norm(Z - centroid, axis=1)
    return 1.0 / (1.0 + dist)


def _bleu(ref: str, hyp: str) -> float:
    ref_t = word_tokenize(ref.lower())
    hyp_t = word_tokenize(hyp.lower())
    if not hyp_t:
        return 0.0
    smooth = SmoothingFunction().method1
    return float(sentence_bleu([ref_t], hyp_t, smoothing_function=smooth))


def _rouge_agg(ref: str, hyp: str, scorer: rouge_scorer.RougeScorer) -> dict[str, float]:
    if not ref.strip() or not hyp.strip():
        return {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}
    r = scorer.score(ref, hyp)
    return {
        "rouge1_f": float(r["rouge1"].fmeasure),
        "rouge2_f": float(r["rouge2"].fmeasure),
        "rougeL_f": float(r["rougeL"].fmeasure),
    }


def _meteor(ref: str, hyp: str) -> float:
    ref_t = word_tokenize(ref.lower())
    hyp_t = word_tokenize(hyp.lower())
    if not ref_t or not hyp_t:
        return 0.0
    return float(meteor_score([ref_t], hyp_t))


def _evaluate_generation(
    cands: pd.DataFrame,
    ensemble_score: np.ndarray,
    rouge_s: rouge_scorer.RougeScorer,
) -> tuple[dict, pd.DataFrame]:
    tmp = cands.copy()
    tmp["_ens"] = ensemble_score
    picked = tmp.sort_values(["mcq_row_id", "_ens"], ascending=[True, False]).groupby("mcq_row_id").head(1)

    rows = []
    bleu_vals, met_vals = [], []
    r1, r2, rl = [], [], []
    for _, r in picked.iterrows():
        ref_q = str(r["gold_question"])
        hyp_q = str(r["candidate_question"])
        b = _bleu(ref_q, hyp_q)
        rg = _rouge_agg(ref_q, hyp_q, rouge_s)
        m = _meteor(ref_q, hyp_q)
        bleu_vals.append(b)
        met_vals.append(m)
        r1.append(rg["rouge1_f"])
        r2.append(rg["rouge2_f"])
        rl.append(rg["rougeL_f"])
        rows.append(
            {
                "id": r["id"],
                "gold_question": ref_q,
                "predicted_question": hyp_q,
                "bleu": b,
                "rouge1_f": rg["rouge1_f"],
                "rouge2_f": rg["rouge2_f"],
                "rougeL_f": rg["rougeL_f"],
                "meteor": m,
            }
        )

    detail = pd.DataFrame(rows)
    agg = {
        "n_examples": int(len(detail)),
        "bleu_mean": float(np.mean(bleu_vals)) if bleu_vals else 0.0,
        "rouge1_f_mean": float(np.mean(r1)) if r1 else 0.0,
        "rouge2_f_mean": float(np.mean(r2)) if r2 else 0.0,
        "rougeL_f_mean": float(np.mean(rl)) if rl else 0.0,
        "meteor_mean": float(np.mean(met_vals)) if met_vals else 0.0,
    }
    return agg, detail


def run_training(cfg: TrainConfig) -> None:
    _ensure_nltk()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    train_mcq = _load_mcq_split(cfg.processed_dir, "train")
    val_mcq = _load_mcq_split(cfg.processed_dir, "validation")
    test_mcq = _load_mcq_split(cfg.processed_dir, "test")

    if cfg.generation_max_train_mcq is not None:
        n = min(cfg.generation_max_train_mcq, len(train_mcq))
        train_mcq = train_mcq.iloc[:n].reset_index(drop=True)
    if cfg.generation_max_val_mcq is not None:
        n = min(cfg.generation_max_val_mcq, len(val_mcq))
        val_mcq = val_mcq.iloc[:n].reset_index(drop=True)
    if cfg.generation_max_test_mcq is not None:
        n = min(cfg.generation_max_test_mcq, len(test_mcq))
        test_mcq = test_mcq.iloc[:n].reset_index(drop=True)

    train_cands = build_generation_candidates(train_mcq, cfg.generation_top_sentences)
    val_cands = build_generation_candidates(val_mcq, cfg.generation_top_sentences)
    test_cands = build_generation_candidates(test_mcq, cfg.generation_top_sentences)

    if train_cands.empty or val_cands.empty or test_cands.empty:
        raise RuntimeError("No generation candidates produced. Check MCQ parquet and article text.")

    X_train = train_cands[GENERATION_FEAT_COLS].to_numpy(dtype=np.float32)
    y_train = train_cands["label"].to_numpy(dtype=np.int8)
    X_val = val_cands[GENERATION_FEAT_COLS].to_numpy(dtype=np.float32)
    X_test = test_cands[GENERATION_FEAT_COLS].to_numpy(dtype=np.float32)

    t0 = perf_counter()
    ranker = LogisticRegression(solver="liblinear", max_iter=500, random_state=cfg.random_seed)
    ranker.fit(X_train, y_train)
    sup_train_sec = perf_counter() - t0

    km, scaler, good_cluster = _fit_unsupervised(X_train, y_train, cfg.random_seed)

    joblib.dump(ranker, cfg.output_dir / "generation_supervised.joblib")
    joblib.dump(km, cfg.output_dir / "generation_kmeans.joblib")
    joblib.dump(scaler, cfg.output_dir / "generation_unsupervised_scaler.joblib")
    (cfg.output_dir / "generation_feature_columns.json").write_text(
        json.dumps(GENERATION_FEAT_COLS, indent=2), encoding="utf-8"
    )

    w = float(cfg.ensemble_weight_supervised)
    w = max(0.0, min(1.0, w))

    def ensemble_scores(cands: pd.DataFrame, X: np.ndarray) -> np.ndarray:
        p_sup = ranker.predict_proba(X)[:, 1]
        Z = scaler.transform(X)
        raw_u = _unsupervised_scores(Z, km, good_cluster)
        gids = cands["mcq_row_id"].to_numpy()
        sup_n = _minmax_by_group(p_sup, gids)
        uns_n = _minmax_by_group(raw_u, gids)
        return w * sup_n + (1.0 - w) * uns_n

    ens_val = ensemble_scores(val_cands, X_val)
    ens_test = ensemble_scores(test_cands, X_test)

    rouge_s = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    val_metrics, val_detail = _evaluate_generation(val_cands, ens_val, rouge_s)
    test_metrics, test_detail = _evaluate_generation(test_cands, ens_test, rouge_s)

    val_out, test_out = val_detail, test_detail
    if cfg.max_eval_mcq is not None:
        val_out = val_detail.head(cfg.max_eval_mcq)
        test_out = test_detail.head(cfg.max_eval_mcq)
    val_out.to_csv(cfg.output_dir / "generation_val_predictions.csv", index=False)
    test_out.to_csv(cfg.output_dir / "generation_test_predictions.csv", index=False)

    meta = {
        "ensemble_weight_supervised": w,
        "good_kmeans_cluster_id": good_cluster,
        "generation_top_sentences": cfg.generation_top_sentences,
        "generation_max_train_mcq": cfg.generation_max_train_mcq,
        "feature_columns": GENERATION_FEAT_COLS,
        "supervised_train_seconds": sup_train_sec,
        "train_candidates": int(len(train_cands)),
    }
    (cfg.output_dir / "model_a_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    summary = {
        "model": "Model A (traditional only)",
        "evaluation": "BLEU, ROUGE, METEOR vs reference question (no classification metrics)",
        "config": {
            "processed_dir": str(cfg.processed_dir),
            "output_dir": str(cfg.output_dir),
            "random_seed": cfg.random_seed,
            "generation_top_sentences": cfg.generation_top_sentences,
            "generation_max_train_mcq": cfg.generation_max_train_mcq,
            "generation_max_val_mcq": cfg.generation_max_val_mcq,
            "generation_max_test_mcq": cfg.generation_max_test_mcq,
            "ensemble_weight_supervised": w,
        },
        "validation": val_metrics,
        "test": test_metrics,
        "artifacts": {
            "supervised": str(cfg.output_dir / "generation_supervised.joblib"),
            "kmeans": str(cfg.output_dir / "generation_kmeans.joblib"),
            "scaler": str(cfg.output_dir / "generation_unsupervised_scaler.joblib"),
            "meta": str(cfg.output_dir / "model_a_meta.json"),
        },
    }
    (cfg.output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Train Model A: classical question generation + BLEU/ROUGE/METEOR.")
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--output-dir", type=Path, default=Path("models/model_a/traditional"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--generation-top-sentences", type=int, default=3)
    p.add_argument("--generation-max-train-mcq", type=int, default=None)
    p.add_argument("--generation-max-val-mcq", type=int, default=None)
    p.add_argument("--generation-max-test-mcq", type=int, default=None)
    p.add_argument("--ensemble-weight-supervised", type=float, default=0.5)
    p.add_argument("--max-eval-mcq", type=int, default=None, help="Cap rows written to prediction CSVs.")
    args = p.parse_args()

    cfg = TrainConfig(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        random_seed=args.seed,
        generation_top_sentences=args.generation_top_sentences,
        generation_max_train_mcq=args.generation_max_train_mcq,
        generation_max_val_mcq=args.generation_max_val_mcq,
        generation_max_test_mcq=args.generation_max_test_mcq,
        ensemble_weight_supervised=args.ensemble_weight_supervised,
        max_eval_mcq=args.max_eval_mcq,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
