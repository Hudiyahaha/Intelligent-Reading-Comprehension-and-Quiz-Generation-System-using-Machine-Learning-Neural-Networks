"""Inference for Model A (classical generation: supervised + unsupervised ensemble)."""

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


def run_pipeline(*_args, **_kwargs):
    raise NotImplementedError("Model B pipeline is pending; use ModelAInference for Model A.")


if __name__ == "__main__":
    run_pipeline()
