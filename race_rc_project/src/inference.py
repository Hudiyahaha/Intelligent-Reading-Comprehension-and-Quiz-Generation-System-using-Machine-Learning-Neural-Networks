"""Inference for Model A and Model B (traditional pipelines)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.model_a_train import (
    GENERATION_FEAT_COLS,
    _guess_wh_word,
    _jaccard,
    _split_sentences,
    _template_question,
    _token_set,
    _unsupervised_scores,
    _minmax_by_group,
)
from src.model_b_train import (
    DISTRACTOR_FEATS,
    HINT_FEATS,
    _jaccard as _jaccard_b,
    _make_hints,
    _split_sentences as _split_sentences_b,
    _token_set as _token_set_b,
    _tokenize,
)


class ModelAInference:
    """Generate a question from (article, answer_text) using trained ensemble."""

    def __init__(
        self,
        model_dir: Path | str = Path("models/model_a/traditional"),
    ) -> None:
        self.model_dir = Path(model_dir)
        self.ranker = joblib.load(self.model_dir / "generation_supervised.joblib")
        self.km = joblib.load(self.model_dir / "generation_kmeans.joblib")
        self.scaler = joblib.load(self.model_dir / "generation_unsupervised_scaler.joblib")
        meta = json.loads((self.model_dir / "model_a_meta.json").read_text(encoding="utf-8"))
        self.ensemble_w = float(meta.get("ensemble_weight_supervised", 0.5))
        self.good_cluster = int(meta.get("good_kmeans_cluster_id", 0))
        self.top_sentences = int(meta.get("generation_top_sentences", 3))

    def generate_question(self, article: str, answer_text: str) -> dict:
        wh = _guess_wh_word(answer_text)
        sentences = _split_sentences(article)
        if not sentences:
            return {"question": "What is the correct answer?", "ensemble_score": 0.0}

        records = []
        scored = []
        for s in sentences:
            scored.append((s, _jaccard(s, answer_text), 0.0))
        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
        for rank, (sent, overlap_ans, overlap_q) in enumerate(scored[: max(1, self.top_sentences)]):
            gen_q, has_blank = _template_question(sent, answer_text, wh)
            sent_tokens = _token_set(sent)
            q_tokens = _token_set(gen_q)
            overlap_q2 = len(sent_tokens & q_tokens) / max(1, len(sent_tokens | q_tokens))
            records.append(
                {
                    "candidate_question": gen_q,
                    "sent_answer_overlap": overlap_ans,
                    "sent_question_overlap": overlap_q,
                    "candidate_gold_similarity": 0.0,
                    "candidate_len": len(q_tokens),
                    "sentence_len": len(sent_tokens),
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
        X = cands[GENERATION_FEAT_COLS].to_numpy(dtype=np.float32)
        p_sup = self.ranker.predict_proba(X)[:, 1]
        Z = self.scaler.transform(X)
        raw_u = _unsupervised_scores(Z, self.km, self.good_cluster)
        gids = np.zeros(len(cands), dtype=np.int64)
        sup_n = _minmax_by_group(p_sup, gids)
        uns_n = _minmax_by_group(raw_u, gids)
        ens = self.ensemble_w * sup_n + (1.0 - self.ensemble_w) * uns_n
        best_i = int(np.argmax(ens))
        return {
            "question": str(cands.iloc[best_i]["candidate_question"]),
            "ensemble_score": float(ens[best_i]),
        }


class ModelBInference:
    """Generate distractors and hints for a question-answer pair."""

    def __init__(self, model_dir: Path | str = Path("models/model_b/traditional")) -> None:
        self.model_dir = Path(model_dir)
        self.d_sup = joblib.load(self.model_dir / "distractor_supervised.joblib")
        self.d_km = joblib.load(self.model_dir / "distractor_kmeans.joblib")
        self.d_scaler = joblib.load(self.model_dir / "distractor_scaler.joblib")
        self.h_sup = joblib.load(self.model_dir / "hint_supervised.joblib")
        self.h_km = joblib.load(self.model_dir / "hint_kmeans.joblib")
        self.h_scaler = joblib.load(self.model_dir / "hint_scaler.joblib")
        meta = json.loads((self.model_dir / "model_b_meta.json").read_text(encoding="utf-8"))
        self.w = float(meta.get("ensemble_weight_supervised", 0.5))
        self.top_d = int(meta.get("top_distractors", 3))
        self.top_h = int(meta.get("top_hints", 3))
        self.d_good = int(meta.get("distractor_good_cluster_id", 0))
        self.h_good = int(meta.get("hint_good_cluster_id", 0))

    def generate(self, article: str, question: str, answer_text: str) -> dict:
        # Distractor ranking
        qset = _token_set_b(question)
        aset = _token_set_b(answer_text)
        toks = [t for t in _tokenize(article) if len(t) > 2 and t not in aset]
        uniq, counts = np.unique(np.array(toks), return_counts=True) if toks else (np.array([]), np.array([]))
        d_rows = []
        maxc = int(counts.max()) if len(counts) else 1
        for tok, c in zip(uniq.tolist(), counts.tolist()):
            d_rows.append(
                {
                    "candidate": tok,
                    "freq_norm": float(c) / max(1, maxc),
                    "len_token": float(len(tok)),
                    "answer_overlap": 1.0 if tok in aset else 0.0,
                    "question_overlap": 1.0 if tok in qset else 0.0,
                    "in_question": 1.0 if tok in qset else 0.0,
                    "in_answer": 1.0 if tok in aset else 0.0,
                }
            )
        distractors: list[str] = []
        if d_rows:
            cands = pd.DataFrame(d_rows)
            X = cands[DISTRACTOR_FEATS].to_numpy(dtype=np.float32)
            s_sup = self.d_sup.predict_proba(X)[:, 1]
            Z = self.d_scaler.transform(X)
            s_uns = _unsupervised_scores(Z, self.d_km, self.d_good)
            cands["score"] = self.w * s_sup + (1.0 - self.w) * s_uns
            distractors = cands.sort_values("score", ascending=False)["candidate"].head(self.top_d).tolist()

        # Hint ranking
        h_rows = []
        sents = _split_sentences_b(article)
        for i, sent in enumerate(sents):
            qov = _jaccard_b(sent, question)
            aov = _jaccard_b(sent, answer_text)
            h_rows.append(
                {
                    "sentence": sent,
                    "q_overlap": qov,
                    "a_overlap": aov,
                    "qa_overlap": 0.6 * aov + 0.4 * qov,
                    "sent_len": float(len(_token_set_b(sent))),
                    "pos_norm": float(i) / max(1, len(sents) - 1),
                }
            )
        hints: list[str] = []
        if h_rows:
            hdf = pd.DataFrame(h_rows)
            Xh = hdf[HINT_FEATS].to_numpy(dtype=np.float32)
            hs_sup = self.h_sup.predict_proba(Xh)[:, 1]
            Zh = self.h_scaler.transform(Xh)
            hs_uns = _unsupervised_scores(Zh, self.h_km, self.h_good)
            hdf["score"] = self.w * hs_sup + (1.0 - self.w) * hs_uns
            top_sents = hdf.sort_values("score", ascending=False)["sentence"].head(self.top_h).tolist()
            hints = _make_hints(top_sents, self.top_h)

        return {"distractors": distractors, "hints": hints}


def run_pipeline(*_args, **_kwargs):
    raise NotImplementedError("Use ModelAInference / ModelBInference directly.")


if __name__ == "__main__":
    run_pipeline()
