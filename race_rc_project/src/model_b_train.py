"""
Model B — traditional distractor + hint generation.

Pipeline:
- Distractors:
  - Supervised logistic ranker on candidate terms
  - Unsupervised KMeans quality scorer
  - Weighted ensemble to pick top-3 distractors
- Hints:
  - Supervised logistic sentence ranker
  - Unsupervised KMeans sentence scorer
  - Weighted ensemble to pick top-3 hint sentences
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
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning, module="nltk")

OPTION_COLS = ("A", "B", "C", "D")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9\-']{1,}")


@dataclass
class TrainConfig:
    processed_dir: Path = Path("data/processed")
    output_dir: Path = Path("models/model_b/traditional")
    random_seed: int = 42
    top_distractors: int = 3
    top_hints: int = 3
    max_train_mcq: int | None = None
    max_val_mcq: int | None = None
    max_test_mcq: int | None = None
    ensemble_weight_supervised: float = 0.5


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


def _tokenize(s: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(s or "")]


def _token_set(s: str) -> set[str]:
    return set(_tokenize(s))


def _jaccard(a: str, b: str) -> float:
    sa, sb = _token_set(a), _token_set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def _split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    out = [p.strip() for p in parts if p.strip()]
    return out if out else [text]


def _load_mcq_split(processed_dir: Path, split: str) -> pd.DataFrame:
    p = processed_dir / f"mcq_{split}.parquet"
    if not p.is_file():
        raise FileNotFoundError(f"Missing split file: {p}")
    return pd.read_parquet(p)


def _minmax_by_group(scores: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
    out = np.zeros_like(scores, dtype=np.float64)
    for g in np.unique(group_ids):
        m = group_ids == g
        s = scores[m]
        lo, hi = float(s.min()), float(s.max())
        out[m] = 1.0 if hi - lo < 1e-12 else (s - lo) / (hi - lo)
    return out


def _fit_unsupervised(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[KMeans, StandardScaler, int]:
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    km = KMeans(n_clusters=2, random_state=seed, n_init=10)
    c = km.fit_predict(Z)
    quality = []
    for i in (0, 1):
        m = c == i
        quality.append(float(y[m].mean()) if m.any() else 0.0)
    good = int(np.argmax(quality))
    return km, scaler, good


def _unsupervised_scores(X: np.ndarray, km: KMeans, scaler: StandardScaler, good_cluster: int) -> np.ndarray:
    Z = scaler.transform(X)
    centroid = km.cluster_centers_[good_cluster]
    dist = np.linalg.norm(Z - centroid, axis=1)
    return 1.0 / (1.0 + dist)


def _limit(df: pd.DataFrame, n: int | None) -> pd.DataFrame:
    if n is None:
        return df
    return df.iloc[: min(n, len(df))].reset_index(drop=True)


DISTRACTOR_FEATS = [
    "freq_norm",
    "len_token",
    "answer_overlap",
    "question_overlap",
    "in_question",
    "in_answer",
]


def _build_distractor_candidates(mcq: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for idx, r in mcq.reset_index(drop=True).iterrows():
        article = str(r.get("article", ""))
        question = str(r.get("question", ""))
        answer_letter = str(r.get("answer", "")).strip().upper()
        if answer_letter not in OPTION_COLS:
            continue
        answer_text = str(r.get(answer_letter, ""))
        wrong_texts = [str(r.get(c, "")) for c in OPTION_COLS if c != answer_letter]
        wrong_norm = {_normalize_text(w) for w in wrong_texts if _normalize_text(w)}

        toks = [t for t in _tokenize(article) if t not in ENGLISH_STOP_WORDS and len(t) > 2]
        if not toks:
            continue
        uniq, counts = np.unique(np.array(toks), return_counts=True)
        maxc = int(counts.max()) if len(counts) else 1
        qset, aset = _token_set(question), _token_set(answer_text)
        for tok, c in zip(uniq.tolist(), counts.tolist()):
            if tok in aset:
                continue
            freq_norm = float(c) / max(1, maxc)
            label = int(tok in wrong_norm)  # weak supervision from incorrect options
            rows.append(
                {
                    "mcq_row_id": idx,
                    "id": r.get("id", str(idx)),
                    "question": question,
                    "answer_text": answer_text,
                    "gold_wrong_options": wrong_texts,
                    "candidate": tok,
                    "freq_norm": freq_norm,
                    "len_token": float(len(tok)),
                    "answer_overlap": 1.0 if tok in aset else 0.0,
                    "question_overlap": 1.0 if tok in qset else 0.0,
                    "in_question": 1.0 if tok in qset else 0.0,
                    "in_answer": 1.0 if tok in aset else 0.0,
                    "label": label,
                }
            )
    return pd.DataFrame(rows)


def _bleu(ref: str, hyp: str) -> float:
    ref_t = word_tokenize(ref.lower())
    hyp_t = word_tokenize(hyp.lower())
    if not hyp_t:
        return 0.0
    smooth = SmoothingFunction().method1
    return float(sentence_bleu([ref_t], hyp_t, smoothing_function=smooth))


def _meteor(ref: str, hyp: str) -> float:
    ref_t = word_tokenize(ref.lower())
    hyp_t = word_tokenize(hyp.lower())
    if not ref_t or not hyp_t:
        return 0.0
    return float(meteor_score([ref_t], hyp_t))


def _rouge_agg(ref: str, hyp: str, scorer: rouge_scorer.RougeScorer) -> dict[str, float]:
    if not ref.strip() or not hyp.strip():
        return {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}
    r = scorer.score(ref, hyp)
    return {
        "rouge1_f": float(r["rouge1"].fmeasure),
        "rouge2_f": float(r["rouge2"].fmeasure),
        "rougeL_f": float(r["rougeL"].fmeasure),
    }


def _evaluate_text_generation(
    pred_df: pd.DataFrame,
    pred_col: str,
    ref_col: str,
    rouge_s: rouge_scorer.RougeScorer,
) -> dict:
    bleu_vals, meteor_vals, r1, r2, rl = [], [], [], [], []
    for _, r in pred_df.iterrows():
        pred = str(r[pred_col]).strip()
        ref = str(r[ref_col]).strip()
        b = _bleu(ref, pred)
        m = _meteor(ref, pred)
        rg = _rouge_agg(ref, pred, rouge_s)
        bleu_vals.append(b)
        meteor_vals.append(m)
        r1.append(rg["rouge1_f"])
        r2.append(rg["rouge2_f"])
        rl.append(rg["rougeL_f"])
    return {
        "n_examples": int(len(pred_df)),
        "bleu_mean": float(np.mean(bleu_vals)) if bleu_vals else 0.0,
        "rouge1_f_mean": float(np.mean(r1)) if r1 else 0.0,
        "rouge2_f_mean": float(np.mean(r2)) if r2 else 0.0,
        "rougeL_f_mean": float(np.mean(rl)) if rl else 0.0,
        "meteor_mean": float(np.mean(meteor_vals)) if meteor_vals else 0.0,
    }


HINT_FEATS = [
    "q_overlap",
    "a_overlap",
    "qa_overlap",
    "sent_len",
    "pos_norm",
]


def _build_hint_candidates(mcq: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for idx, r in mcq.reset_index(drop=True).iterrows():
        article = str(r.get("article", ""))
        question = str(r.get("question", ""))
        answer_letter = str(r.get("answer", "")).strip().upper()
        if answer_letter not in OPTION_COLS:
            continue
        answer_text = str(r.get(answer_letter, ""))
        sents = _split_sentences(article)
        if not sents:
            continue
        # weak label: sentence with max overlap to answer+question
        oq = []
        for pos, s in enumerate(sents):
            qov = _jaccard(s, question)
            aov = _jaccard(s, answer_text)
            qa = 0.6 * aov + 0.4 * qov
            oq.append((pos, s, qov, aov, qa))
        best_pos = max(oq, key=lambda x: x[4])[0]
        n = len(sents)
        for pos, s, qov, aov, qa in oq:
            rows.append(
                {
                    "mcq_row_id": idx,
                    "id": r.get("id", str(idx)),
                    "question": question,
                    "answer_text": answer_text,
                    "sentence": s,
                    "q_overlap": qov,
                    "a_overlap": aov,
                    "qa_overlap": qa,
                    "sent_len": float(len(_token_set(s))),
                    "pos_norm": float(pos) / max(1, n - 1),
                    "label": int(pos == best_pos),
                }
            )
    return pd.DataFrame(rows)


def _make_hints(sentences: list[str], top_k: int) -> list[str]:
    if not sentences:
        return []
    # graduated hints: low -> medium -> high specificity from the ranked set
    if len(sentences) == 1:
        return [sentences[0]]
    if len(sentences) == 2:
        return [sentences[1], sentences[0]]
    picked = sentences[:top_k]
    return [picked[-1], picked[len(picked) // 2], picked[0]][:top_k]


def run_training(cfg: TrainConfig) -> None:
    _ensure_nltk()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    train_mcq = _limit(_load_mcq_split(cfg.processed_dir, "train"), cfg.max_train_mcq)
    val_mcq = _limit(_load_mcq_split(cfg.processed_dir, "validation"), cfg.max_val_mcq)
    test_mcq = _limit(_load_mcq_split(cfg.processed_dir, "test"), cfg.max_test_mcq)

    # ---------------- Distractors ----------------
    tr_d = _build_distractor_candidates(train_mcq)
    va_d = _build_distractor_candidates(val_mcq)
    te_d = _build_distractor_candidates(test_mcq)
    if tr_d.empty or va_d.empty or te_d.empty:
        raise RuntimeError("Distractor candidate generation produced empty splits.")

    Xtr = tr_d[DISTRACTOR_FEATS].to_numpy(dtype=np.float32)
    ytr = tr_d["label"].to_numpy(dtype=np.int8)
    Xva = va_d[DISTRACTOR_FEATS].to_numpy(dtype=np.float32)
    Xte = te_d[DISTRACTOR_FEATS].to_numpy(dtype=np.float32)

    t0 = perf_counter()
    d_sup = LogisticRegression(solver="liblinear", max_iter=400, random_state=cfg.random_seed)
    d_sup.fit(Xtr, ytr)
    d_km, d_scaler, d_good = _fit_unsupervised(Xtr, ytr, cfg.random_seed)
    d_train_sec = perf_counter() - t0

    joblib.dump(d_sup, cfg.output_dir / "distractor_supervised.joblib")
    joblib.dump(d_km, cfg.output_dir / "distractor_kmeans.joblib")
    joblib.dump(d_scaler, cfg.output_dir / "distractor_scaler.joblib")

    w = float(max(0.0, min(1.0, cfg.ensemble_weight_supervised)))

    def _rank_distr(cands: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
        p_sup = d_sup.predict_proba(X)[:, 1]
        p_uns = _unsupervised_scores(X, d_km, d_scaler, d_good)
        gids = cands["mcq_row_id"].to_numpy()
        s_sup = _minmax_by_group(p_sup, gids)
        s_uns = _minmax_by_group(p_uns, gids)
        score = w * s_sup + (1.0 - w) * s_uns
        tmp = cands.copy()
        tmp["score"] = score
        pred_rows = []
        for _, g in tmp.sort_values(["mcq_row_id", "score"], ascending=[True, False]).groupby("mcq_row_id"):
            unique = []
            for t in g["candidate"].tolist():
                if t not in unique:
                    unique.append(t)
                if len(unique) == cfg.top_distractors:
                    break
            pred_rows.append(
                {
                    "id": g.iloc[0]["id"],
                    "question": g.iloc[0]["question"],
                    "answer_text": g.iloc[0]["answer_text"],
                    "gold_wrong_options": g.iloc[0]["gold_wrong_options"],
                    "pred_distractors": unique,
                    "pred_distractors_text": " ; ".join(unique),
                    "ref_distractors_text": " ; ".join([str(x) for x in g.iloc[0]["gold_wrong_options"] if str(x).strip()]),
                }
            )
        return pd.DataFrame(pred_rows)

    val_pred_d = _rank_distr(va_d, Xva)
    test_pred_d = _rank_distr(te_d, Xte)
    val_pred_d.to_csv(cfg.output_dir / "distractor_val_predictions.csv", index=False)
    test_pred_d.to_csv(cfg.output_dir / "distractor_test_predictions.csv", index=False)
    rouge_s = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    val_d_metrics = _evaluate_text_generation(val_pred_d, "pred_distractors_text", "ref_distractors_text", rouge_s)
    test_d_metrics = _evaluate_text_generation(test_pred_d, "pred_distractors_text", "ref_distractors_text", rouge_s)

    # ---------------- Hints ----------------
    tr_h = _build_hint_candidates(train_mcq)
    va_h = _build_hint_candidates(val_mcq)
    te_h = _build_hint_candidates(test_mcq)
    if tr_h.empty or va_h.empty or te_h.empty:
        raise RuntimeError("Hint candidate generation produced empty splits.")

    Xtrh = tr_h[HINT_FEATS].to_numpy(dtype=np.float32)
    ytrh = tr_h["label"].to_numpy(dtype=np.int8)
    Xvah = va_h[HINT_FEATS].to_numpy(dtype=np.float32)
    Xteh = te_h[HINT_FEATS].to_numpy(dtype=np.float32)

    t0 = perf_counter()
    h_sup = LogisticRegression(solver="liblinear", max_iter=400, random_state=cfg.random_seed)
    h_sup.fit(Xtrh, ytrh)
    h_km, h_scaler, h_good = _fit_unsupervised(Xtrh, ytrh, cfg.random_seed)
    h_train_sec = perf_counter() - t0

    joblib.dump(h_sup, cfg.output_dir / "hint_supervised.joblib")
    joblib.dump(h_km, cfg.output_dir / "hint_kmeans.joblib")
    joblib.dump(h_scaler, cfg.output_dir / "hint_scaler.joblib")

    def _rank_hints(cands: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
        p_sup = h_sup.predict_proba(X)[:, 1]
        p_uns = _unsupervised_scores(X, h_km, h_scaler, h_good)
        gids = cands["mcq_row_id"].to_numpy()
        s_sup = _minmax_by_group(p_sup, gids)
        s_uns = _minmax_by_group(p_uns, gids)
        score = w * s_sup + (1.0 - w) * s_uns
        tmp = cands.copy()
        tmp["score"] = score
        pred_rows = []
        for _, g in tmp.sort_values(["mcq_row_id", "score"], ascending=[True, False]).groupby("mcq_row_id"):
            top_sents = g["sentence"].head(cfg.top_hints).tolist()
            pred_rows.append(
                {
                    "id": g.iloc[0]["id"],
                    "question": g.iloc[0]["question"],
                    "answer_text": g.iloc[0]["answer_text"],
                    "pred_hints": _make_hints(top_sents, cfg.top_hints),
                    "pred_hints_text": " ; ".join(_make_hints(top_sents, cfg.top_hints)),
                    "ref_hints_text": " ; ".join(_make_hints(
                        g.sort_values("qa_overlap", ascending=False)["sentence"].head(cfg.top_hints).tolist(),
                        cfg.top_hints,
                    )),
                }
            )
        return pd.DataFrame(pred_rows)

    val_pred_h = _rank_hints(va_h, Xvah)
    test_pred_h = _rank_hints(te_h, Xteh)
    val_pred_h.to_csv(cfg.output_dir / "hint_val_predictions.csv", index=False)
    test_pred_h.to_csv(cfg.output_dir / "hint_test_predictions.csv", index=False)
    val_h_metrics = _evaluate_text_generation(val_pred_h, "pred_hints_text", "ref_hints_text", rouge_s)
    test_h_metrics = _evaluate_text_generation(test_pred_h, "pred_hints_text", "ref_hints_text", rouge_s)

    meta = {
        "ensemble_weight_supervised": w,
        "top_distractors": cfg.top_distractors,
        "top_hints": cfg.top_hints,
        "distractor_good_cluster_id": int(d_good),
        "hint_good_cluster_id": int(h_good),
        "distractor_features": DISTRACTOR_FEATS,
        "hint_features": HINT_FEATS,
        "distractor_train_seconds": d_train_sec,
        "hint_train_seconds": h_train_sec,
    }
    (cfg.output_dir / "model_b_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    summary = {
        "model": "Model B (traditional only)",
        "config": {
            "processed_dir": str(cfg.processed_dir),
            "output_dir": str(cfg.output_dir),
            "random_seed": cfg.random_seed,
            "top_distractors": cfg.top_distractors,
            "top_hints": cfg.top_hints,
            "max_train_mcq": cfg.max_train_mcq,
            "max_val_mcq": cfg.max_val_mcq,
            "max_test_mcq": cfg.max_test_mcq,
            "ensemble_weight_supervised": w,
        },
        "validation": {
            "distractor_generation": val_d_metrics,
            "hint_generation": val_h_metrics,
        },
        "test": {
            "distractor_generation": test_d_metrics,
            "hint_generation": test_h_metrics,
        },
        "artifacts": {
            "distractor_supervised": str(cfg.output_dir / "distractor_supervised.joblib"),
            "distractor_kmeans": str(cfg.output_dir / "distractor_kmeans.joblib"),
            "distractor_scaler": str(cfg.output_dir / "distractor_scaler.joblib"),
            "hint_supervised": str(cfg.output_dir / "hint_supervised.joblib"),
            "hint_kmeans": str(cfg.output_dir / "hint_kmeans.joblib"),
            "hint_scaler": str(cfg.output_dir / "hint_scaler.joblib"),
            "meta": str(cfg.output_dir / "model_b_meta.json"),
        },
    }
    (cfg.output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Train Model B (traditional distractor + hint generation).")
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--output-dir", type=Path, default=Path("models/model_b/traditional"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-distractors", type=int, default=3)
    p.add_argument("--top-hints", type=int, default=3)
    p.add_argument("--max-train-mcq", type=int, default=None)
    p.add_argument("--max-val-mcq", type=int, default=None)
    p.add_argument("--max-test-mcq", type=int, default=None)
    p.add_argument("--ensemble-weight-supervised", type=float, default=0.5)
    args = p.parse_args()

    cfg = TrainConfig(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        random_seed=args.seed,
        top_distractors=args.top_distractors,
        top_hints=args.top_hints,
        max_train_mcq=args.max_train_mcq,
        max_val_mcq=args.max_val_mcq,
        max_test_mcq=args.max_test_mcq,
        ensemble_weight_supervised=args.ensemble_weight_supervised,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
