"""
BERT baseline — reranker comparison against Model A and Model B.

This module reuses the *same* candidate generation that Model A and Model B
produce (template questions, article-token distractors, and article-sentence
hints) and replaces the classical Logistic / KMeans ensemble scorer with a
BERT encoder (default: ``bert-base-uncased``).

For each (candidate, target) pair we compute a mean-pooled BERT sentence
embedding and rank candidates by cosine similarity to the target. This keeps
the evaluation directly comparable: same val / test MCQs, same candidate pool,
same BLEU / ROUGE / METEOR metrics — only the scorer changes.

Outputs (mirroring ``models/model_a`` and ``models/model_b``):
    models/model_bert/
        metrics_summary.json
        model_bert_meta.json
        generation_val_predictions.csv
        generation_test_predictions.csv
        distractor_val_predictions.csv
        distractor_test_predictions.csv
        hint_val_predictions.csv
        hint_test_predictions.csv
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer

from src.model_a_train import (
    GENERATION_FEAT_COLS,
    _bleu,
    _ensure_nltk,
    _meteor,
    _rouge_agg,
    build_generation_candidates,
)
from src.model_b_train import (
    DISTRACTOR_FEATS,
    HINT_FEATS,
    _build_distractor_candidates,
    _build_hint_candidates,
    _evaluate_text_generation,
    _make_hints,
    _load_mcq_split,
)

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


@dataclass
class BertConfig:
    processed_dir: Path = Path("data/processed")
    output_dir: Path = Path("models/model_bert")
    model_name: str = "bert-base-uncased"
    random_seed: int = 42

    generation_top_sentences: int = 3
    top_distractors: int = 3
    top_hints: int = 3

    max_train_mcq: int | None = None
    max_val_mcq: int | None = None
    max_test_mcq: int | None = None

    distractor_candidate_cap: int = 30
    batch_size: int = 32
    max_length: int = 128
    device: str | None = None
    cache_dir: Path | None = None
    fp16: bool = False

    skip_distractors: bool = False
    skip_hints: bool = False
    skip_question_generation: bool = False

    feature_columns: dict[str, list[str]] = field(
        default_factory=lambda: {
            "question_generation": list(GENERATION_FEAT_COLS),
            "distractor_generation": list(DISTRACTOR_FEATS),
            "hint_generation": list(HINT_FEATS),
        }
    )


class _BertEncoder:
    """Mean-pooled BERT sentence encoder with an in-memory cache.

    The encoder is intentionally lightweight: it loads a HuggingFace BERT
    checkpoint, runs forward passes in batches, and returns L2-normalised
    sentence vectors so cosine similarity reduces to a dot product.
    """

    def __init__(
        self,
        model_name: str,
        device: str | None,
        batch_size: int,
        max_length: int,
        cache_dir: Path | None,
        fp16: bool,
    ) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised at runtime
            raise SystemExit(
                "torch and transformers are required for the BERT baseline. "
                "Install them via `pip install -r requirements.txt`."
            ) from exc
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch = torch
        self.device = device
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=str(cache_dir) if cache_dir else None
        )
        self.model = AutoModel.from_pretrained(
            model_name, cache_dir=str(cache_dir) if cache_dir else None
        )
        self.model.eval()
        self.model.to(self.device)
        if fp16 and self.device != "cpu":
            self.model.half()
            self._dtype = torch.float16
        else:
            self._dtype = torch.float32
        self._cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _key(text: str) -> str:
        return (text or "").strip()

    def encode(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        unique: list[str] = []
        seen: set[str] = set()
        for t in texts:
            k = self._key(t)
            if k and k not in self._cache and k not in seen:
                unique.append(k)
                seen.add(k)

        for start in range(0, len(unique), self.batch_size):
            batch = unique[start : start + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**enc)
            last = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(last.dtype)
            summed = (last * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            mean = summed / counts
            mean = torch.nn.functional.normalize(mean, p=2, dim=1)
            vecs = mean.detach().to(torch.float32).cpu().numpy()
            for text, vec in zip(batch, vecs):
                self._cache[text] = vec

        out_vecs = []
        for t in texts:
            k = self._key(t)
            if not k:
                vec = np.zeros(self.model.config.hidden_size, dtype=np.float32)
            else:
                vec = self._cache[k]
            out_vecs.append(vec)
        return np.vstack(out_vecs).astype(np.float32)

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)


def _limit(df: pd.DataFrame, n: int | None) -> pd.DataFrame:
    if n is None:
        return df
    return df.iloc[: min(n, len(df))].reset_index(drop=True)


def _cap_distractor_candidates(cands: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Limit distractors to the top-N most frequent tokens per MCQ.

    Encoding every article token with BERT is too slow on CPU; using
    ``freq_norm`` keeps the candidate set faithful to Model B's pool while
    making BERT rerank tractable on full val + test.
    """
    if cap is None or cap <= 0:
        return cands
    out = (
        cands.sort_values(["mcq_row_id", "freq_norm"], ascending=[True, False])
        .groupby("mcq_row_id", group_keys=False)
        .head(cap)
        .reset_index(drop=True)
    )
    return out


def _rank_question_generation(
    cands: pd.DataFrame,
    encoder: _BertEncoder,
) -> pd.DataFrame:
    """Pick the candidate question with the highest BERT cosine to its target.

    Target text = ``article answer`` (approximated by the *sentence* the
    candidate came from plus the gold answer, which is what Model A had access
    to). We embed each candidate question and each target, then choose the
    top-1 candidate per MCQ.
    """
    target_texts = (
        cands["sentence_text"].astype(str) + " " + cands["answer_text"].astype(str)
    ).tolist()
    cand_texts = cands["candidate_question"].astype(str).tolist()
    cand_vecs = encoder.encode(cand_texts)
    tgt_vecs = encoder.encode(target_texts)
    sims = np.sum(cand_vecs * tgt_vecs, axis=1)
    tmp = cands.copy()
    tmp["bert_score"] = sims
    picked = (
        tmp.sort_values(["mcq_row_id", "bert_score"], ascending=[True, False])
        .groupby("mcq_row_id")
        .head(1)
    )
    return picked.reset_index(drop=True)


def _evaluate_question_generation(
    picked: pd.DataFrame,
    rouge_s: rouge_scorer.RougeScorer,
) -> tuple[dict, pd.DataFrame]:
    rows = []
    bleu_vals, met_vals, r1, r2, rl = [], [], [], [], []
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
                "bert_score": float(r.get("bert_score", 0.0)),
                "bleu": b,
                "rouge1_f": rg["rouge1_f"],
                "rouge2_f": rg["rouge2_f"],
                "rougeL_f": rg["rougeL_f"],
                "meteor": m,
            }
        )
    agg = {
        "n_examples": int(len(rows)),
        "bleu_mean": float(np.mean(bleu_vals)) if bleu_vals else 0.0,
        "rouge1_f_mean": float(np.mean(r1)) if r1 else 0.0,
        "rouge2_f_mean": float(np.mean(r2)) if r2 else 0.0,
        "rougeL_f_mean": float(np.mean(rl)) if rl else 0.0,
        "meteor_mean": float(np.mean(met_vals)) if met_vals else 0.0,
    }
    return agg, pd.DataFrame(rows)


def _rank_distractors(
    cands: pd.DataFrame,
    encoder: _BertEncoder,
    top_k: int,
) -> pd.DataFrame:
    """Pick top-K distractor tokens by BERT cosine to a (question, answer) cue.

    Target text = ``question`` + the answer (so we keep candidates that are
    topically related to the question, similarly to how Model B's KMeans
    quality cluster behaves). Per MCQ we deduplicate tokens and return the
    top-K.
    """
    target_texts = (
        cands["question"].astype(str) + " " + cands["answer_text"].astype(str)
    ).tolist()
    cand_texts = cands["candidate"].astype(str).tolist()
    cand_vecs = encoder.encode(cand_texts)
    tgt_vecs = encoder.encode(target_texts)
    sims = np.sum(cand_vecs * tgt_vecs, axis=1)
    tmp = cands.copy()
    tmp["bert_score"] = sims
    pred_rows = []
    for _, g in tmp.sort_values(
        ["mcq_row_id", "bert_score"], ascending=[True, False]
    ).groupby("mcq_row_id"):
        unique: list[str] = []
        for t in g["candidate"].tolist():
            if t not in unique:
                unique.append(t)
            if len(unique) == top_k:
                break
        pred_rows.append(
            {
                "id": g.iloc[0]["id"],
                "question": g.iloc[0]["question"],
                "answer_text": g.iloc[0]["answer_text"],
                "gold_wrong_options": g.iloc[0]["gold_wrong_options"],
                "pred_distractors": unique,
                "pred_distractors_text": " ; ".join(unique),
                "ref_distractors_text": " ; ".join(
                    [str(x) for x in g.iloc[0]["gold_wrong_options"] if str(x).strip()]
                ),
            }
        )
    return pd.DataFrame(pred_rows)


def _rank_hints(
    cands: pd.DataFrame,
    encoder: _BertEncoder,
    top_k: int,
) -> pd.DataFrame:
    """Pick top-K hint sentences by BERT cosine to (question, answer).

    We mirror Model B's evaluation: the reference hint text is built from the
    top-K sentences ranked by ``qa_overlap`` (the supervised weak label), so
    a BERT-ranked subset that overlaps the same gold pool scores higher.
    """
    target_texts = (
        cands["question"].astype(str) + " " + cands["answer_text"].astype(str)
    ).tolist()
    sent_texts = cands["sentence"].astype(str).tolist()
    sent_vecs = encoder.encode(sent_texts)
    tgt_vecs = encoder.encode(target_texts)
    sims = np.sum(sent_vecs * tgt_vecs, axis=1)
    tmp = cands.copy()
    tmp["bert_score"] = sims
    pred_rows = []
    for _, g in tmp.sort_values(
        ["mcq_row_id", "bert_score"], ascending=[True, False]
    ).groupby("mcq_row_id"):
        top_sents = g["sentence"].head(top_k).tolist()
        ref_top = (
            g.sort_values("qa_overlap", ascending=False)["sentence"].head(top_k).tolist()
        )
        pred_rows.append(
            {
                "id": g.iloc[0]["id"],
                "question": g.iloc[0]["question"],
                "answer_text": g.iloc[0]["answer_text"],
                "pred_hints": _make_hints(top_sents, top_k),
                "pred_hints_text": " ; ".join(_make_hints(top_sents, top_k)),
                "ref_hints_text": " ; ".join(_make_hints(ref_top, top_k)),
            }
        )
    return pd.DataFrame(pred_rows)


def run_bert_baseline(cfg: BertConfig) -> dict:
    _ensure_nltk()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    val_mcq = _limit(_load_mcq_split(cfg.processed_dir, "validation"), cfg.max_val_mcq)
    test_mcq = _limit(_load_mcq_split(cfg.processed_dir, "test"), cfg.max_test_mcq)

    t_load = perf_counter()
    encoder = _BertEncoder(
        model_name=cfg.model_name,
        device=cfg.device,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
        cache_dir=cfg.cache_dir,
        fp16=cfg.fp16,
    )
    load_sec = perf_counter() - t_load

    rouge_s = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    summary: dict = {
        "model": f"BERT rerank baseline ({cfg.model_name})",
        "evaluation": (
            "BLEU, ROUGE, METEOR vs reference text — same val/test MCQs as Model A & B."
        ),
        "config": {
            "processed_dir": str(cfg.processed_dir),
            "output_dir": str(cfg.output_dir),
            "model_name": cfg.model_name,
            "device": encoder.device,
            "batch_size": cfg.batch_size,
            "max_length": cfg.max_length,
            "generation_top_sentences": cfg.generation_top_sentences,
            "top_distractors": cfg.top_distractors,
            "top_hints": cfg.top_hints,
            "max_val_mcq": cfg.max_val_mcq,
            "max_test_mcq": cfg.max_test_mcq,
            "distractor_candidate_cap": cfg.distractor_candidate_cap,
            "fp16": cfg.fp16,
        },
        "timings": {"model_load_seconds": load_sec},
        "validation": {},
        "test": {},
    }

    if not cfg.skip_question_generation:
        t = perf_counter()
        val_qcands = build_generation_candidates(val_mcq, cfg.generation_top_sentences)
        test_qcands = build_generation_candidates(
            test_mcq, cfg.generation_top_sentences
        )
        if val_qcands.empty or test_qcands.empty:
            raise RuntimeError("No question-generation candidates produced.")

        val_picked = _rank_question_generation(val_qcands, encoder)
        test_picked = _rank_question_generation(test_qcands, encoder)
        val_q_metrics, val_q_detail = _evaluate_question_generation(val_picked, rouge_s)
        test_q_metrics, test_q_detail = _evaluate_question_generation(
            test_picked, rouge_s
        )
        val_q_detail.to_csv(
            cfg.output_dir / "generation_val_predictions.csv", index=False
        )
        test_q_detail.to_csv(
            cfg.output_dir / "generation_test_predictions.csv", index=False
        )
        summary["validation"]["question_generation"] = val_q_metrics
        summary["test"]["question_generation"] = test_q_metrics
        summary["timings"]["question_generation_seconds"] = perf_counter() - t

    if not cfg.skip_distractors:
        t = perf_counter()
        val_dcands = _cap_distractor_candidates(
            _build_distractor_candidates(val_mcq), cfg.distractor_candidate_cap
        )
        test_dcands = _cap_distractor_candidates(
            _build_distractor_candidates(test_mcq), cfg.distractor_candidate_cap
        )
        if val_dcands.empty or test_dcands.empty:
            raise RuntimeError("No distractor candidates produced.")

        val_pred_d = _rank_distractors(val_dcands, encoder, cfg.top_distractors)
        test_pred_d = _rank_distractors(test_dcands, encoder, cfg.top_distractors)
        val_pred_d.to_csv(
            cfg.output_dir / "distractor_val_predictions.csv", index=False
        )
        test_pred_d.to_csv(
            cfg.output_dir / "distractor_test_predictions.csv", index=False
        )
        summary["validation"]["distractor_generation"] = _evaluate_text_generation(
            val_pred_d, "pred_distractors_text", "ref_distractors_text", rouge_s
        )
        summary["test"]["distractor_generation"] = _evaluate_text_generation(
            test_pred_d, "pred_distractors_text", "ref_distractors_text", rouge_s
        )
        summary["timings"]["distractor_generation_seconds"] = perf_counter() - t

    if not cfg.skip_hints:
        t = perf_counter()
        val_hcands = _build_hint_candidates(val_mcq)
        test_hcands = _build_hint_candidates(test_mcq)
        if val_hcands.empty or test_hcands.empty:
            raise RuntimeError("No hint candidates produced.")

        val_pred_h = _rank_hints(val_hcands, encoder, cfg.top_hints)
        test_pred_h = _rank_hints(test_hcands, encoder, cfg.top_hints)
        val_pred_h.to_csv(cfg.output_dir / "hint_val_predictions.csv", index=False)
        test_pred_h.to_csv(cfg.output_dir / "hint_test_predictions.csv", index=False)
        summary["validation"]["hint_generation"] = _evaluate_text_generation(
            val_pred_h, "pred_hints_text", "ref_hints_text", rouge_s
        )
        summary["test"]["hint_generation"] = _evaluate_text_generation(
            test_pred_h, "pred_hints_text", "ref_hints_text", rouge_s
        )
        summary["timings"]["hint_generation_seconds"] = perf_counter() - t

    meta = {
        "model_name": cfg.model_name,
        "device": encoder.device,
        "hidden_size": encoder.hidden_size,
        "batch_size": cfg.batch_size,
        "max_length": cfg.max_length,
        "generation_top_sentences": cfg.generation_top_sentences,
        "top_distractors": cfg.top_distractors,
        "top_hints": cfg.top_hints,
        "distractor_candidate_cap": cfg.distractor_candidate_cap,
        "feature_columns": cfg.feature_columns,
    }
    (cfg.output_dir / "model_bert_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    (cfg.output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _write_comparison(
    repo_models_dir: Path, bert_summary: dict
) -> dict:
    """Combine Model A, Model B, and BERT metrics into a single comparison JSON."""

    def _load(p: Path) -> dict | None:
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
        return None

    a = _load(repo_models_dir / "model_a" / "traditional" / "metrics_summary.json")
    b = _load(repo_models_dir / "model_b" / "traditional" / "metrics_summary.json")

    def _pick(d: dict | None, *keys: str) -> dict | None:
        cur: dict | None = d
        for k in keys:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(k)
        return cur if isinstance(cur, dict) else None

    comparison = {
        "tasks": {
            "question_generation": {
                "validation": {
                    "model_a": _pick(a, "validation"),
                    "bert": _pick(bert_summary, "validation", "question_generation"),
                },
                "test": {
                    "model_a": _pick(a, "test"),
                    "bert": _pick(bert_summary, "test", "question_generation"),
                },
            },
            "distractor_generation": {
                "validation": {
                    "model_b": _pick(b, "validation", "distractor_generation"),
                    "bert": _pick(bert_summary, "validation", "distractor_generation"),
                },
                "test": {
                    "model_b": _pick(b, "test", "distractor_generation"),
                    "bert": _pick(bert_summary, "test", "distractor_generation"),
                },
            },
            "hint_generation": {
                "validation": {
                    "model_b": _pick(b, "validation", "hint_generation"),
                    "bert": _pick(bert_summary, "validation", "hint_generation"),
                },
                "test": {
                    "model_b": _pick(b, "test", "hint_generation"),
                    "bert": _pick(bert_summary, "test", "hint_generation"),
                },
            },
        }
    }
    out = repo_models_dir / "comparison_summary.json"
    out.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return comparison


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Train/score the BERT rerank baseline against Model A and Model B "
            "candidates (same val/test MCQs, BLEU/ROUGE/METEOR)."
        )
    )
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--output-dir", type=Path, default=Path("models/model_bert"))
    p.add_argument("--model-name", type=str, default="bert-base-uncased")
    p.add_argument("--generation-top-sentences", type=int, default=3)
    p.add_argument("--top-distractors", type=int, default=3)
    p.add_argument("--top-hints", type=int, default=3)
    p.add_argument("--max-val-mcq", type=int, default=None)
    p.add_argument("--max-test-mcq", type=int, default=None)
    p.add_argument("--distractor-candidate-cap", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--device", type=str, default=None, help="cpu / cuda (auto if None)")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--skip-question-generation", action="store_true")
    p.add_argument("--skip-distractors", action="store_true")
    p.add_argument("--skip-hints", action="store_true")
    p.add_argument(
        "--no-comparison",
        action="store_true",
        help="Do not write models/comparison_summary.json next to the BERT outputs.",
    )
    args = p.parse_args()

    cfg = BertConfig(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        generation_top_sentences=args.generation_top_sentences,
        top_distractors=args.top_distractors,
        top_hints=args.top_hints,
        max_val_mcq=args.max_val_mcq,
        max_test_mcq=args.max_test_mcq,
        distractor_candidate_cap=args.distractor_candidate_cap,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        cache_dir=args.cache_dir,
        fp16=args.fp16,
        skip_distractors=args.skip_distractors,
        skip_hints=args.skip_hints,
        skip_question_generation=args.skip_question_generation,
    )
    summary = run_bert_baseline(cfg)
    if not args.no_comparison:
        _write_comparison(args.output_dir.parent, summary)


if __name__ == "__main__":
    main()
